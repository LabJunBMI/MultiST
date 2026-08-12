import os
import json
import time
import tracemalloc
import gc
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData

import torch
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import scipy.sparse as sp
from tqdm import tqdm

import MultiST
from MultiST.MultiST_model import MultiST as MultiSTModel
from MultiST.image_process import EnhancedDualModalMultiST, train_enhanced_dual_modal_MultiST

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0',
                     help="GPU index, e.g. 0 or 1; maps to cuda:0 / cuda:1")
parser.add_argument('--seeds', type=str, default='42,123,456,789,2024',
                     help="Comma-separated seed list, e.g. '42,123' or '456,789,2024'")
parser.add_argument('--tag', type=str, default='',
                     help="Output directory suffix, avoids parallel processes overwriting each other's CSV, e.g. 'gpu0' / 'gpu1'")
args, _ = parser.parse_known_args()

# ===================== CONFIG =====================
qc_dir       = '/data/bai/Spatial/PLOS/dataset/crop/visium_hd_cancer_colon_square_016um/qc'
labels_path  = '/data/bai/Spatial/PLOS/dataset/crop/visium_hd_cancer_colon_square_016um/labels.tsv'
spatial_dir  = '/data/bai/Spatial/PLOS/dataset/binned_outputs/square_016um/spatial'
tissue_image_path = os.path.join(spatial_dir, 'tissue_hires_image.png')

base_output = '/data/bai/Spatial/multiST_visiumHD_colon_result'
if args.tag:
    base_output = f"{base_output}_{args.tag}"   
section_id = 'colon_cancer_16um'   


EXCLUDE_LABELS = ['Outside']

random_seeds = [int(s) for s in args.seeds.split(',')]

USE_GAN = True
USE_IMAGE = True
USE_LABEL_PROP = True
RUN_NAME = 'ColonCancer'

os.environ["R_HOME"]      = "/home/wang3wa/.conda/envs/wei/lib/R"
os.environ["R_PATH"]      = "/home/wang3wa/.conda/envs/wei/bin/R"
os.environ["R_LIBS_USER"] = "/home/wang3wa/.conda/envs/wei/lib/R/library"
os.environ["PATH"]        = "/home/wang3wa/.conda/envs/wei/bin:" + os.environ["PATH"]

device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} | seeds: {random_seeds} | base_output: {base_output}")


# ===================== DATA LOADING (Visium HD) =====================

def load_visium_hd_section(qc_dir, spatial_dir, labels_path, exclude_labels=None):
    qc_dir = Path(qc_dir)
    spatial_dir = Path(spatial_dir)

    obs = pd.read_csv(qc_dir / 'observations.tsv', sep='\t', index_col=0)
    var = pd.read_csv(qc_dir / 'features.tsv', sep='\t', index_col=0)
    coords = pd.read_csv(qc_dir / 'coordinates.tsv', sep=r'\s+', index_col=0, engine="python")
    labels = pd.read_csv(labels_path, sep='\t', index_col=0)

    X = sio.mmread(qc_dir / 'counts.mtx')
    X = sp.csr_matrix(X)

    n_obs, n_var = len(obs), len(var)
    if X.shape == (n_obs, n_var):
        pass
    elif X.shape == (n_var, n_obs):
        X = X.T.tocsr()
    else:
        raise ValueError(f"counts.mtx shape {X.shape} does not match obs/var row counts "
                          f"({n_obs}, {n_var}); please check whether the files match")

    adata = AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()

    # Keep only spots that passed QC and are within tissue
    if 'selected' in adata.obs.columns:
        adata = adata[adata.obs['selected'] == True].copy()
    if 'in_tissue' in adata.obs.columns:
        adata = adata[adata.obs['in_tissue'] == True].copy()

    coords.columns = coords.columns.astype(str).str.strip()

    if {'x', 'y'}.issubset(coords.columns):
        xy_cols = ['x', 'y']
    elif {'pxl_col_in_fullres', 'pxl_row_in_fullres'}.issubset(coords.columns):
        xy_cols = ['pxl_col_in_fullres', 'pxl_row_in_fullres']
    elif coords.shape[1] >= 2:
        xy_cols = coords.columns[:2].tolist()
        print(f"  [coords] No x/y columns found, falling back to the first two columns as coordinates: {xy_cols}")
    else:
        raise ValueError(f"coordinates.tsv has too few coordinate columns, current columns: {coords.columns.tolist()}")

    coords = coords.reindex(adata.obs_names)

    missing = coords[xy_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"  [WARNING] {missing} spots have no matching coordinates and will be dropped")
        keep = ~coords[xy_cols].isna().any(axis=1)
        adata = adata[keep.values].copy()
        coords = coords.loc[adata.obs_names]

    adata.obsm['spatial'] = coords[xy_cols].astype(float).values
    adata.obs['layer_guess'] = labels.reindex(adata.obs_names)['label'].values

    if exclude_labels:
        before_n = adata.n_obs
        keep_mask = ~adata.obs['layer_guess'].isin(exclude_labels)
        adata = adata[keep_mask].copy()
        print(f"  [exclude_labels={exclude_labels}] {before_n} -> {adata.n_obs} spots "
              f"(dropped {before_n - adata.n_obs})")

    library_id = 'colon_cancer_16um'
    with open(spatial_dir / 'scalefactors_json.json') as f:
        scalef = json.load(f)

    hires = np.array(Image.open(spatial_dir / 'tissue_hires_image.png')) / 255.0
    lowres = np.array(Image.open(spatial_dir / 'tissue_lowres_image.png')) / 255.0

    adata.uns['spatial'] = {
        library_id: {
            'images': {'hires': hires, 'lowres': lowres},
            'scalefactors': scalef,
            'metadata': {}
        }
    }

    return adata

def get_n_clusters():
    """Automatically determine the number of clusters from the class count
    in labels.tsv, excluding EXCLUDE_LABELS (e.g. Outside)."""
    labels = pd.read_csv(labels_path, sep='\t', index_col=0)
    valid_labels = labels[~labels['label'].isin(EXCLUDE_LABELS)]
    return valid_labels['label'].nunique()


def safe_cluster_fused_features(
    adata, n_clusters, seed,
    input_key="features", pca_key="MultiST_pca", cluster_key="mclust",
    n_pca_components=30
):
    if input_key not in adata.obsm:
        raise KeyError(
            f"'{input_key}' not found in adata.obsm, "
            f"available keys: {list(adata.obsm.keys())}"
        )

    features = np.asarray(adata.obsm[input_key])
    if features.ndim != 2:
        raise ValueError(f"Fused features must be a 2D matrix, current shape={features.shape}")
    if features.shape[0] != adata.n_obs:
        raise ValueError(
            f"Fused feature sample count {features.shape[0]} does not match adata.n_obs={adata.n_obs}"
        )

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- PCA ----
    max_components = min(n_pca_components, features.shape[0] - 1, features.shape[1])
    if max_components < 2:
        raise ValueError(f"PCA available dimensionality too low, at most {max_components} components can be used")

    pca_model = PCA(n_components=max_components, random_state=seed)
    fusion_pca = np.asarray(pca_model.fit_transform(features), dtype=np.float64)
    fusion_pca = np.nan_to_num(fusion_pca, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"PCA done: n_components={max_components}, "
          f"explained_variance_ratio_sum={pca_model.explained_variance_ratio_.sum():.4f}")
    component_variance = np.var(fusion_pca, axis=0)
    valid_components = component_variance > 1e-12
    if valid_components.sum() < fusion_pca.shape[1]:
        removed = fusion_pca.shape[1] - valid_components.sum()
        print(f"[WARNING] Removing {removed} near-zero-variance PCA components")
        fusion_pca = fusion_pca[:, valid_components]
    if fusion_pca.shape[1] < 2:
        raise ValueError("After removing zero-variance components, PCA has fewer than 2 valid dimensions")

    adata.obsm[pca_key] = fusion_pca.astype(np.float32)

    # ---- R mclust ----
    print("Running R mclust...")
    MultiST.mclust_R(adata, n_clusters, use_rep=pca_key, key_added=cluster_key, random_seed=seed)

    if cluster_key not in adata.obs.columns:
        raise RuntimeError(f"'{cluster_key}' not found in adata.obs after mclust ran")
    if adata.obs[cluster_key].isna().any():
        raise RuntimeError("mclust returned labels containing NaN")

    n_found = adata.obs[cluster_key].nunique()
    if n_found != n_clusters:
        raise RuntimeError(f"mclust returned {n_found} clusters, expected {n_clusters}")

    print(f"R mclust succeeded: {n_found} clusters")
    adata.uns["clustering_method"] = "R_mclust"
    return cluster_key


def weighted_label_propagation_gaussian(
    adata, label_key='mclust', spatial_key='spatial',
    k=7, anchor_keep_ratio=0.01, sigma=None, n_iter=10, out_key='refined_label'
):
    coords = adata.obsm[spatial_key]
    labels = adata.obs[label_key].astype(str).values
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    n_classes = len(le.classes_)
    N = len(labels)

    A = kneighbors_graph(coords, n_neighbors=k, mode='distance', include_self=False).tocsr()

    if sigma is None:
        sigma = np.median(A.data)

    W = A.copy()
    W.data = np.exp(-(A.data ** 2) / (sigma ** 2))

    W_coo = W.tocoo()
    rows, cols, wdata = W_coo.row, W_coo.col, W_coo.data
    same_label = (label_ids[rows] == label_ids[cols])

    agree_scores = np.zeros(N)
    np.add.at(agree_scores, rows[same_label], wdata[same_label])

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


def compute_metrics(adata, pred_key, true_key='layer_guess'):
    sub = adata[~pd.isnull(adata.obs[true_key])]
    y_true = sub.obs[true_key].values
    y_pred = sub.obs[pred_key].values
    ari  = metrics.adjusted_rand_score(y_true, y_pred)
    ami  = metrics.adjusted_mutual_info_score(y_true, y_pred)
    comp = metrics.completeness_score(y_true, y_pred)
    return ari, ami, comp


def save_clustering_fig(adata, pred_key, true_key, ari, save_path, title_suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    sc.pl.spatial(adata, color=true_key, ax=axes[0], show=False)
    sc.pl.spatial(adata, color=pred_key, ax=axes[1], show=False)
    axes[0].set_title('Manual Annotation')
    axes[1].set_title(f'Clustering {title_suffix} (ARI={ari:.4f})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def mask_generator_safe(adj_label, N=1):
    """PyTorch 2.x compatible negative-edge sampler. adj_label must be a sparse COO tensor."""
    adj_label = adj_label.coalesce()
    idx = adj_label.indices()
    values = adj_label.values()
    cell_num = adj_label.size(0)

    row_list, col_list = [], []
    for i in range(cell_num):
        neighbor = idx[1, idx[0] == i]
        n_selected = int(len(neighbor) * N)
        if n_selected == 0:
            continue

        total_idx = torch.arange(cell_num, dtype=torch.long)
        invalid = torch.cat([neighbor, torch.tensor([i], dtype=torch.long)])
        non_neighbor = total_idx[~torch.isin(total_idx, invalid)]
        if len(non_neighbor) == 0:
            continue

        n_selected = min(n_selected, len(non_neighbor))
        perm = torch.randperm(len(non_neighbor))[:n_selected]
        selected = non_neighbor[perm]

        row_list.append(torch.full((n_selected,), i, dtype=torch.long))
        col_list.append(selected)

    if len(row_list) == 0:
        return adj_label

    neg_rows = torch.cat(row_list)
    neg_cols = torch.cat(col_list)
    neg_indices = torch.stack([neg_rows, neg_cols], dim=0)

    all_indices = torch.cat([idx, neg_indices], dim=1)
    all_values = torch.cat([values.float(), torch.zeros(len(neg_rows), dtype=torch.float32)])

    return torch.sparse_coo_tensor(
        all_indices, all_values, size=adj_label.shape, dtype=torch.float32
    ).coalesce()


def scipy_sparse_to_torch_coo(matrix):
    """scipy sparse matrix -> torch sparse COO tensor"""
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, size=matrix.shape, dtype=torch.float32).coalesce()


def preprocess_graph_safe(adj):
    """A_hat = D^(-1/2) (A + I) D^(-1/2)"""
    adj = adj.tocsr().astype(np.float32)
    adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")

    rowsum = np.asarray(adj_.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(rowsum, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    degree_mat = sp.diags(degree_inv_sqrt)

    adj_normalized = (degree_mat @ adj_ @ degree_mat).tocoo()
    return scipy_sparse_to_torch_coo(adj_normalized)


def graph_construction_visium_hd(adata, n=12, negative_ratio=1):
    """
    Sparse KNN graph construction for Visium HD. Does not build an N x N
    distance matrix, avoiding O(N^2) memory blowup at 50k+ spot scale.
    Output format matches the original MultiST.graph_construction.
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    print(f"  [graph] Building sparse KNN graph: {coords.shape[0]} spots, k={n}")

    adj = kneighbors_graph(
        coords, n_neighbors=n, mode="connectivity", include_self=False, n_jobs=1
    ).astype(np.float32)

    adj = adj.maximum(adj.T).tocsr()  # symmetrize
    adj.setdiag(0)
    adj.eliminate_zeros()
    print(f"  [graph] Sparse adjacency: shape={adj.shape}, edges={adj.nnz}")

    adj_norm = preprocess_graph_safe(adj)

    # label adjacency includes self-loops, keeping the original logic
    adj_label_scipy = (adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")).tocoo()
    adj_label = scipy_sparse_to_torch_coo(adj_label_scipy)

    total_pairs = float(adj.shape[0] * adj.shape[0])
    positive_pairs = float(adj_label_scipy.sum())
    denominator = (total_pairs - positive_pairs) * 2.0
    norm_value = total_pairs / denominator if denominator > 0 else 1.0

    adj_mask = mask_generator_safe(adj_label, N=negative_ratio)

    return {
        "adj_norm": adj_norm,
        "adj_label": adj_label,
        "norm_value": norm_value,
        "mask": adj_mask,
    }


def load_and_preprocess_section():
    adata = load_visium_hd_section(qc_dir, spatial_dir, labels_path, exclude_labels=EXCLUDE_LABELS)

    n_labeled = adata.obs['layer_guess'].notna().sum()
    tqdm.write(f"     Loaded: {adata.n_obs} spots, {adata.n_vars} genes, "
               f"of which {n_labeled} are labeled ({n_labeled/adata.n_obs*100:.1f}%)")

    # Keep sparse, avoids toarray() blowing up memory at 130k-spot scale
    adata.layers['count'] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True].copy()
    sc.pp.scale(adata)

    if sp.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)

    return adata


# ===================== MAIN LOOP =====================
out_dir = Path(base_output) / RUN_NAME
out_dir.mkdir(parents=True, exist_ok=True)

n_clusters = get_n_clusters()
all_records = []

tqdm.write(f"\n  >> {section_id}: preloading + preprocessing once, reused by all seeds")
cached_base_adata = load_and_preprocess_section()

seed_bar = tqdm(random_seeds, desc=f"[{section_id}] Seeds")
for seed in seed_bar:
    seed_bar.set_postfix(seed=seed)
    tqdm.write(f"\n  >> {section_id} | seed={seed}")

    run_dir = out_dir / section_id / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    cpu_before = psutil.cpu_percent(interval=None)
    start_time = time.time()
    process = psutil.Process(os.getpid())

    try:
        MultiST.fix_seed(seed)

        adata = cached_base_adata.copy()
        tqdm.write(f"     Using cached preprocessed data: "
                   f"{adata.n_obs} spots, {adata.n_vars} genes, n_clusters={n_clusters}")

        adata.obsm['X_pca'] = PCA(n_components=200, random_state=seed).fit_transform(adata.X)

        # ---- Graph + MultiST ----
        graph_dict = graph_construction_visium_hd(adata, n=12, negative_ratio=1)
        MultiST_net = MultiSTModel(
            X=adata.obsm['X_pca'], graph_dict=graph_dict,
            use_gan=USE_GAN, gan_w=0.3, mmd_w=0.2, rec_w=10, gcn_w=0.1,
            mode='clustering', device=device,
        )
        MultiST_net.train_with_dec(preepoch=400, epochs=500, N=1)
        MultiST_feat, _, _, _ = MultiST_net.process()
        adata.obsm['MultiST'] = MultiST_feat

        # ---- Image branch ----
        if USE_IMAGE:
            optimized_model = EnhancedDualModalMultiST(
                trained_MultiST_model=MultiST_net.model,
                num_clusters=n_clusters,
                image_dim=512, hidden_dim=256, spatial_k=7,
                patch_size=64, image_size=224,
                num_workers=2,  
                sdm_temperature=0.1, use_contrastive=True,
                attention_mode='img2gene',
                gene_weight=0.7,
                apply_normalization=True, save_images=False,
                normalization_strategy='optimal_diverse',
            )

            spatial_coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
            library_id = list(adata.uns['spatial'].keys())[0]
            scale_factor = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
            tqdm.write(f"     Image scale factor: {scale_factor}")

            trainer, eval_results, _ = train_enhanced_dual_modal_MultiST(
                model=optimized_model,
                gene_features=MultiST_feat,
                image_path=tissue_image_path,
                spatial_coords=spatial_coords,
                scale_factor=scale_factor,
                epochs=50, batch_size=128, device=device,
                log_dir=str(run_dir),
                log_name=f"{section_id}_seed{seed}",
                adata=adata, save_images=False,
            )

            if 'fused_features' not in eval_results:
                raise KeyError(f"'fused_features' not found in eval_results, "
                                f"current keys: {list(eval_results.keys())}")

            fusion_features = np.asarray(eval_results['fused_features'], dtype=np.float32)
            tqdm.write(f"     Fused feature shape: {fusion_features.shape}, "
                       f"NaN: {np.isnan(fusion_features).sum()}, Inf: {np.isinf(fusion_features).sum()}")
            fusion_features = np.nan_to_num(fusion_features, nan=0.0, posinf=0.0, neginf=0.0)
            adata.obsm['features'] = fusion_features

            cluster_key = safe_cluster_fused_features(
                adata=adata, n_clusters=n_clusters, seed=seed,
                input_key='features', pca_key='MultiST_pca', cluster_key='mclust',
                n_pca_components=30,
            )

        else:
            MultiST.mclust_R(adata, n_clusters, use_rep='MultiST',
                              key_added='MultiST_mclust', random_seed=seed)
            if 'MultiST_mclust' not in adata.obs.columns:
                raise RuntimeError("Gene-only mclust did not return labels")
            cluster_key = 'MultiST_mclust'
            adata.uns['clustering_method'] = 'R_mclust'

        if USE_LABEL_PROP:
            adata = weighted_label_propagation_gaussian(
                adata, label_key=cluster_key, spatial_key='spatial',
                k=8, anchor_keep_ratio=0.01, sigma=None, n_iter=10, out_key='refined_mclust',
            )
            final_key = 'refined_mclust'
        else:
            final_key = cluster_key

        elapsed = time.time() - start_time
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_usage = (cpu_before + psutil.cpu_percent(interval=1)) / 2
        ram_mb = process.memory_info().rss / 1024 / 1024

        ari, ami, comp = compute_metrics(adata, final_key)
        tqdm.write(f"     ARI={ari:.4f}  AMI={ami:.4f}  Comp={comp:.4f}  "
                   f"Time={elapsed:.1f}s  CPU={cpu_usage:.1f}%  RAM={ram_mb:.1f}MB")

        fig_path = run_dir / f"clustering_{section_id}_seed{seed}.png"
        save_clustering_fig(adata, final_key, 'layer_guess', ari, str(fig_path), title_suffix=f"({RUN_NAME})")

        clustering_method = adata.uns.get('clustering_method', 'unknown')
        tqdm.write(f"     Final clustering method: {clustering_method}")

        h5ad_path = run_dir / f"{section_id}_seed{seed}.h5ad"
        adata.write_h5ad(str(h5ad_path))

        record = {
            'run_name': RUN_NAME,
            'section_id': section_id,
            'seed': seed,
            'n_clusters': n_clusters,
            'clustering_method': clustering_method,
            'ARI': ari,
            'AMI': ami,
            'Completeness': comp,
            'runtime_sec': elapsed,
            'cpu_percent': cpu_usage,
            'ram_mb': ram_mb,
            'mem_peak_mb': mem_peak / 1024 / 1024,
            'status': 'success',
        }

    except Exception as e:
        tracemalloc.stop()
        elapsed = time.time() - start_time
        tqdm.write(f"     ERROR: {e}")
        import traceback; traceback.print_exc()

        record = {
            'run_name': RUN_NAME,
            'section_id': section_id,
            'seed': seed,
            'n_clusters': n_clusters,
            'clustering_method': 'error',
            'ARI': np.nan,
            'AMI': np.nan,
            'Completeness': np.nan,
            'runtime_sec': elapsed,
            'cpu_percent': np.nan,
            'ram_mb': np.nan,
            'mem_peak_mb': np.nan,
            'status': f'error: {str(e)[:100]}',
        }

    all_records.append(record)
    pd.DataFrame(all_records).to_csv(out_dir / f"results_{RUN_NAME}.csv", index=False)

    for var_name in ["MultiST_net", "optimized_model", "trainer", "eval_results",
                      "fusion_features", "graph_dict"]:
        if var_name in locals():
            del locals()[var_name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

df_all = pd.DataFrame(all_records)
df_all.to_csv(Path(base_output) / "all_runs_global.csv", index=False)

summary = df_all[df_all['status'] == 'success'][
    ['ARI', 'AMI', 'Completeness', 'runtime_sec']
].agg(['mean', 'std']).round(4)
summary.to_csv(out_dir / f"summary_{RUN_NAME}.csv")

print(f"Results saved to: {out_dir}")
print("\n=== Results summary ===")
print(df_all[['run_name', 'section_id', 'seed', 'ARI', 'AMI', 'Completeness']])