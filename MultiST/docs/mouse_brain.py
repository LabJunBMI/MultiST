import scanpy as sc
import pandas as pd
import numpy as np
import torch
import os
import time
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder

import MultiST
from MultiST.MultiST_model import MultiST as MultiSTModel
from MultiST.image_process import EnhancedDualModalMultiST, train_enhanced_dual_modal_MultiST

# ===================== CONFIG =====================
data_path = '/home/wang3wa/Spatial/paper/data/mouse_brain_coronal'
COUNT_FILE = 'V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5'  
base_output = '/data/bai/Spatial/PLOS/multiST_mouseBrainCoronal_no_ground_truth_20'

SEEDS = [2024] 
# BIC search range for the number of clusters (no ground truth available,
G_MIN, G_MAX = 2, 20

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.environ["R_HOME"]      = "/home/wang3wa/.conda/envs/wei/lib/R"
os.environ["R_PATH"]      = "/home/wang3wa/.conda/envs/wei/bin/R"
os.environ["R_LIBS_USER"] = "/home/wang3wa/.conda/envs/wei/lib/R/library"
os.environ["PATH"]        = "/home/wang3wa/.conda/envs/wei/bin:" + os.environ["PATH"]


# ===================== HELPERS =====================
def fix_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    MultiST.fix_seed(seed)
    import rpy2.robjects as ro
    ro.r(f'set.seed({seed})')


def load_and_preprocess(seed):
    adata = sc.read_visium(data_path, count_file=COUNT_FILE)
    adata.var_names_make_unique()

    adata.layers['count'] = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True].copy()
    sc.pp.scale(adata)

    adata_X = PCA(n_components=200, random_state=seed).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    return adata


def mclust_R_auto(adata, use_rep='MultiST_pca', key_added='mclust',
                   G_range=(2, 15), random_seed=2024):
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()

    np.random.seed(random_seed)
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    rmclust = robjects.r['Mclust']
    modelNames = 'EEE'

    g_min, g_max = G_range
    G_vector = robjects.IntVector(list(range(g_min, g_max + 1)))

    data_r = rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep])
    res = rmclust(data_r, G_vector, modelNames)

    try:
        optimal_G = int(np.array(res.rx2('G'))[0])
    except Exception as e:
        print(f"[WARNING] Could not extract 'G' via res.rx2('G'): {e}")
        print("  Falling back to inferring G from the number of unique "
              "cluster labels in the classification vector.")
        optimal_G = None

    mclust_res = np.array(res[-2]) 

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int').astype('category')

    if optimal_G is None:
        optimal_G = adata.obs[key_added].nunique()

    print(f"[mclust_R_auto] BIC-selected optimal number of clusters: {optimal_G} "
          f"(searched G in [{g_min}, {g_max}])")

    return adata, optimal_G


def weighted_label_propagation_gaussian(
    adata, label_key='mclust', spatial_key='spatial',
    k=8, anchor_keep_ratio=0.01, sigma=None, n_iter=10, out_key='refined_label'
):
    coords = adata.obsm[spatial_key]
    labels = adata.obs[label_key].astype(str).values
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    n_classes = len(le.classes_)
    N = len(labels)

    A = kneighbors_graph(coords, n_neighbors=k, mode='distance', include_self=False)
    dist_matrix = A.toarray()
    dist_matrix[dist_matrix == 0] = np.inf
    if sigma is None:
        sigma = np.median(dist_matrix[dist_matrix != np.inf])
    W = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
    W[dist_matrix == np.inf] = 0

    agree_scores = np.zeros(N)
    for c in range(n_classes):
        mask_c = (label_ids == c)
        agree_scores[mask_c] = (W[mask_c] * (label_ids == c).astype(float)).sum(axis=1)

    anchor_mask = np.zeros(N, dtype=bool)
    for c in range(n_classes):
        mask_c = (label_ids == c)
        scores_c = agree_scores[mask_c]
        cutoff = np.quantile(scores_c, 1 - anchor_keep_ratio)
        anchor_mask[mask_c] = scores_c >= cutoff

    Y = np.zeros((N, n_classes))
    Y[np.arange(N), label_ids] = 1
    for _ in range(n_iter):
        Y_new = W.dot(Y)
        Y_new[anchor_mask] = Y[anchor_mask]
        Y = Y_new / (Y_new.sum(axis=1, keepdims=True) + 1e-9)

    refined_ids = np.argmax(Y, axis=1)
    adata.obs[out_key] = pd.Series(
        le.inverse_transform(refined_ids), index=adata.obs.index
    ).astype("category")
    return adata


def save_predictions_csv(adata, pred_key, out_path, true_key='layer_guess'):
    """Save (spot, true, pred); 'true' is NaN since no ground truth exists
    for this dataset, keeping the same schema as ground-truth experiments."""
    if true_key in adata.obs.columns:
        true_vals = adata.obs[true_key].values
    else:
        true_vals = np.full(adata.n_obs, np.nan)

    pred_df = pd.DataFrame({
        'spot': adata.obs_names,
        'true': true_vals,
        'pred': adata.obs[pred_key].astype(str).values,
    })
    pred_df.to_csv(out_path, index=False)
    return pred_df


# ===================== MAIN LOOP =====================
out_dir = Path(base_output)
out_dir.mkdir(parents=True, exist_ok=True)
all_records = []

for seed in SEEDS:
    print(f"\n{'=' * 60}")
    print(f"seed={seed}")
    print(f"{'=' * 60}")

    run_dir = out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    start_time = time.time()
    record = {'seed': seed, 'status': 'success'}

    try:
        fix_all_seeds(seed)
        adata = load_and_preprocess(seed)
        print(f"  Loaded {adata.n_obs} spots, {adata.n_vars} genes (after HVG filtering)")

        graph_dict = MultiST.graph_construction(adata, 12)
        MultiST_net = MultiSTModel(
            X=adata.obsm['X_pca'], graph_dict=graph_dict,
            use_gan=True, gan_w=0.3, mmd_w=0.2, rec_w=10, gcn_w=0.1,
            self_w=1, dec_kl_w=1, mode='clustering', device=device,
        )
        MultiST_net.train_with_dec(preepoch=400, epochs=500, N=1)
        MultiST_feat, _, _, _ = MultiST_net.process()
        adata.obsm['MultiST'] = MultiST_feat

        image_path = os.path.join(data_path, 'spatial', 'tissue_hires_image.png')
        optimized_model = EnhancedDualModalMultiST(
            trained_MultiST_model=MultiST_net.model,
            num_clusters=G_MAX, 
            image_dim=512, hidden_dim=256, spatial_k=7,
            patch_size=64, image_size=224, num_workers=8,
            sdm_temperature=0.1, use_contrastive=True,
            attention_mode='img2gene', gene_weight=0.7,
            apply_normalization=True, save_images=False,
            normalization_strategy='optimal_diverse',
        )

        spatial_coords = adata.obsm['spatial']
        library_id = list(adata.uns['spatial'].keys())[0]
        scale_factor = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']

        trainer, eval_results, _ = train_enhanced_dual_modal_MultiST(
            model=optimized_model,
            gene_features=MultiST_feat,
            image_path=image_path,
            spatial_coords=spatial_coords,
            scale_factor=scale_factor,
            epochs=50, batch_size=128, device=device,
            log_dir=str(run_dir),
            log_name=f"mouseBrainCoronal_seed{seed}",
            adata=adata, save_images=False,
        )

        fusion_features = eval_results['fused_features']
        adata.obsm['features'] = fusion_features
        adata.obsm['MultiST_pca'] = PCA(n_components=50, random_state=seed).fit_transform(
            adata.obsm['features']
        )

        # ---- Automatic cluster number selection via BIC (no ground truth) ----
        adata, optimal_G = mclust_R_auto(
            adata, use_rep='MultiST_pca', key_added='mclust',
            G_range=(G_MIN, G_MAX), random_seed=seed
        )
        record['bic_selected_G'] = optimal_G

        adata = weighted_label_propagation_gaussian(
            adata, label_key='mclust', spatial_key='spatial',
            k=8, anchor_keep_ratio=0.01, sigma=None, n_iter=10, out_key='refined_mclust'
        )
        final_key = 'refined_mclust'

        elapsed = time.time() - start_time
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  BIC-selected G={optimal_G}  Time={elapsed:.1f}s")

        pred_path = run_dir / f"mouseBrainCoronal_seed{seed}_predictions.csv"
        save_predictions_csv(adata, final_key, pred_path, true_key='layer_guess')

        record.update({
            'n_clusters_final': adata.obs[final_key].nunique(),
            'runtime_sec': elapsed, 'mem_peak_mb': mem_peak / 1024 / 1024,
            'predictions_csv': str(pred_path),
        })

    except Exception as e:
        tracemalloc.stop()
        elapsed = time.time() - start_time
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        record.update({
            'bic_selected_G': np.nan, 'n_clusters_final': np.nan,
            'runtime_sec': elapsed, 'mem_peak_mb': np.nan,
            'status': f'error: {str(e)[:150]}',
            'predictions_csv': None,
        })

    all_records.append(record)
    pd.DataFrame(all_records).to_csv(out_dir / "mouseBrainCoronal_results.csv", index=False)


df_all = pd.DataFrame(all_records)
df_valid = df_all[df_all['status'] == 'success']

print("\n\n========== MOUSE BRAIN CORONAL (NO GROUND TRUTH) COMPLETE ==========")
print(f"n_seeds successful: {len(df_valid)}/{len(SEEDS)}")
if not df_valid.empty:
    print(f"BIC-selected G across seeds: {df_valid['bic_selected_G'].tolist()}")
print(f"\nResults saved to: {out_dir}")