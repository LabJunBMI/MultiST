import os
import time
import tracemalloc
import warnings
from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder

import torch
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import MultiST
from MultiST.MultiST_model import MultiST as MultiSTModel
from MultiST.image_process import EnhancedDualModalMultiST, train_enhanced_dual_modal_MultiST


# ===================== CONFIG =====================
data_path = "/home/wang3wa/Spatial/paper/data/BRCA1"
sample_name = "V1_Human_Breast_Cancer_Block_A_Section_1"
base_output = "/data/bai/test_PLOS"


data_names = [sample_name]
FIXED_N_CLUSTERS = 20

# ===================== KEY PARAMETERS =====================
SEED = 2024
USE_GAN = True
USE_IMAGE = True
USE_LABEL_PROP = True
RUN_NAME = "BRCA"

# R environment
os.environ["R_HOME"] = "/home/wang3wa/.conda/envs/wei/lib/R"
os.environ["R_PATH"] = "/home/wang3wa/.conda/envs/wei/bin/R"
os.environ["R_LIBS_USER"] = "/home/wang3wa/.conda/envs/wei/lib/R/library"
os.environ["PATH"] = "/home/wang3wa/.conda/envs/wei/bin:" + os.environ["PATH"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ===================== HELPERS =====================
def get_n_clusters(section_id):
    return FIXED_N_CLUSTERS


def sanitize_name(name: str) -> str:
    """Convert sample name into a filesystem-safe name."""
    return name.replace(" ", "_").replace("/", "_")


def fix_all_seeds(seed):
    """Fix all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    MultiST.fix_seed(seed)

    import rpy2.robjects as ro
    ro.r(f"set.seed({seed})")


def weighted_label_propagation_gaussian(
    adata,
    label_key="mclust",
    spatial_key="spatial",
    k=7,
    anchor_keep_ratio=0.01,
    sigma=None,
    n_iter=10,
    out_key="refined_label",
):
    coords = adata.obsm[spatial_key]
    labels = adata.obs[label_key].astype(str).values
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    n_classes = len(le.classes_)
    n_samples = len(labels)

    A = kneighbors_graph(coords, n_neighbors=k, mode="distance", include_self=False)
    dist_matrix = A.toarray()
    dist_matrix[dist_matrix == 0] = np.inf

    if sigma is None:
        sigma = np.median(dist_matrix[dist_matrix != np.inf])

    W = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
    W[dist_matrix == np.inf] = 0

    agree_scores = np.zeros(n_samples)
    for c in range(n_classes):
        mask_c = label_ids == c
        agree_scores[mask_c] = (W[mask_c] * (label_ids == c).astype(float)).sum(axis=1)

    anchor_mask = np.zeros(n_samples, dtype=bool)
    for c in range(n_classes):
        mask_c = label_ids == c
        scores_c = agree_scores[mask_c]
        cutoff = np.quantile(scores_c, 1 - anchor_keep_ratio)
        anchor_mask[mask_c] = scores_c >= cutoff

    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), label_ids] = 1

    for _ in range(n_iter):
        Y_new = W.dot(Y)
        Y_new[anchor_mask] = Y[anchor_mask]
        Y = Y_new / (Y_new.sum(axis=1, keepdims=True) + 1e-9)

    refined_ids = np.argmax(Y, axis=1)
    adata.obs[out_key] = pd.Series(
        le.inverse_transform(refined_ids), index=adata.obs.index
    ).astype("category")
    return adata


def compute_metrics(adata, pred_key, true_key="layer_guess"):
    sub = adata[~pd.isnull(adata.obs[true_key])]
    y_true = sub.obs[true_key].values
    y_pred = sub.obs[pred_key].values

    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    comp = metrics.completeness_score(y_true, y_pred)
    return ari, ami, comp


def save_clustering_fig(adata, pred_key, true_key, ari, save_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    sc.pl.spatial(adata, color=true_key, ax=axes[0], show=False)
    sc.pl.spatial(adata, color=pred_key, ax=axes[1], show=False)
    axes[0].set_title("Manual Annotation")
    axes[1].set_title(f"Clustering {title_suffix} (ARI={ari:.4f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===================== MAIN RUN =====================
all_records = []

run_dir_root = Path(base_output) / RUN_NAME
run_dir_root.mkdir(parents=True, exist_ok=True)

for section_id in data_names:
    n_clusters = get_n_clusters(section_id)
    safe_section_id = sanitize_name(section_id)

    section_dir = run_dir_root / safe_section_id
    section_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  >> {section_id} | seed={SEED}")

    run_dir = section_dir / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    cpu_before = psutil.cpu_percent(interval=None)
    start_time = time.time()
    process = psutil.Process(os.getpid())

    try:
        fix_all_seeds(SEED)

        # ---- Load data ----
        dir_input = os.path.join(data_path, section_id)

        adata = sc.read_visium(dir_input)
        adata.var_names_make_unique()

        df_meta = pd.read_csv(os.path.join(dir_input, "metadata.tsv"), sep="\t")
        adata.obs["layer_guess"] = df_meta["fine_annot_type"].values

        adata.layers["count"] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            layer="count",
            n_top_genes=2000,
        )
        adata = adata[:, adata.var["highly_variable"] == True]
        sc.pp.scale(adata)

        adata_X = PCA(n_components=200, random_state=SEED).fit_transform(adata.X)
        adata.obsm["X_pca"] = adata_X

        # ---- Graph + MultiST ----
        graph_dict = MultiST.graph_construction(adata, 12)
        MultiST_net = MultiSTModel(
            X=adata.obsm["X_pca"],
            graph_dict=graph_dict,
            use_gan=USE_GAN,
            gan_w=0.3,
            mmd_w=0.2,
            rec_w=10,
            gcn_w=0.1,
            mode="clustering",
            device=device,
        )
        MultiST_net.train_with_dec(preepoch=400, epochs=500, N=1)
        MultiST_feat, _, _, _ = MultiST_net.process()
        adata.obsm["MultiST"] = MultiST_feat

        # ---- Image branch ----
        if USE_IMAGE:
            image_path = os.path.join(
                dir_input, "spatial", "tissue_hires_image.png"
            )

            optimized_model = EnhancedDualModalMultiST(
                trained_MultiST_model=MultiST_net.model,
                num_clusters=n_clusters,
                image_dim=512,
                hidden_dim=256,
                spatial_k=7,
                patch_size=64,
                image_size=224,
                num_workers=8,
                sdm_temperature=0.1,
                use_contrastive=True,
                attention_mode="bidirectional",
                gene_weight=0.7,
                apply_normalization=True,
                save_images=False,
                normalization_strategy="optimal_diverse",
            )

            spatial_coords = adata.obsm["spatial"]
            library_id = list(adata.uns["spatial"].keys())[0]
            scale_factor = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]

            trainer, eval_results, _ = train_enhanced_dual_modal_MultiST(
                model=optimized_model,
                gene_features=MultiST_feat,
                image_path=image_path,
                spatial_coords=spatial_coords,
                scale_factor=scale_factor,
                epochs=50,
                batch_size=128,
                device=device,
                log_dir=str(run_dir),
                log_name=f"{safe_section_id}_seed{SEED}",
                adata=adata,
                save_images=False,
            )

            fusion_features = eval_results["fused_features"]
            adata.obsm["features"] = fusion_features
            adata.obsm["MultiST_pca"] = PCA(
                n_components=50, random_state=SEED
            ).fit_transform(adata.obsm["features"])

            import rpy2.robjects as ro
            ro.r(f"set.seed({SEED})")
            MultiST.mclust_R(
                adata,
                n_clusters,
                use_rep="MultiST_pca",
                key_added="mclust",
                random_seed=SEED,
            )
            cluster_key = "mclust"

        else:
            import rpy2.robjects as ro
            ro.r(f"set.seed({SEED})")
            MultiST.mclust_R(
                adata,
                n_clusters,
                use_rep="MultiST",
                key_added="MultiST_mclust",
                random_seed=SEED,
            )
            cluster_key = "MultiST_mclust"

        # ---- Label propagation ----
        if USE_LABEL_PROP:
            adata = weighted_label_propagation_gaussian(
                adata,
                label_key=cluster_key,
                spatial_key="spatial",
                k=8,
                anchor_keep_ratio=0.01,
                sigma=None,
                n_iter=10,
                out_key="refined_mclust",
            )
            final_key = "refined_mclust"
        else:
            final_key = cluster_key

        # ---- Resource snapshot ----
        elapsed = time.time() - start_time
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_after = psutil.cpu_percent(interval=1)
        cpu_usage = (cpu_before + cpu_after) / 2
        ram_mb = process.memory_info().rss / 1024 / 1024

        # ---- Metrics ----
        ari, ami, comp = compute_metrics(adata, final_key)
        print(
            f"     ARI={ari:.4f}  AMI={ami:.4f}  Comp={comp:.4f}  "
            f"Time={elapsed:.1f}s  CPU={cpu_usage:.1f}%  RAM={ram_mb:.1f}MB"
        )

        record = {
            "run_name": RUN_NAME,
            "section_id": section_id,
            "seed": SEED,
            "n_clusters": n_clusters,
            "final_key": final_key,
            "cluster_key": cluster_key,
            "ARI": ari,
            "AMI": ami,
            "Completeness": comp,
            "runtime_sec": elapsed,
            "cpu_percent": cpu_usage,
            "ram_mb": ram_mb,
            "mem_peak_mb": mem_peak / 1024 / 1024,
            "status": "success",
        }

        # ---- Save figure and h5ad for this run ----
        fig_path = section_dir / f"clustering_{safe_section_id}_seed{SEED}.png"
        save_clustering_fig(
            adata,
            final_key,
            "layer_guess",
            ari,
            str(fig_path),
            title_suffix=f"({RUN_NAME}, seed={SEED})",
        )

        h5ad_path = section_dir / f"{safe_section_id}_seed{SEED}.h5ad"
        adata.write_h5ad(str(h5ad_path))

    except Exception as e:
        try:
            tracemalloc.stop()
        except Exception:
            pass
        elapsed = time.time() - start_time
        print(f"     ERROR: {e}")
        import traceback
        traceback.print_exc()

        record = {
            "run_name": RUN_NAME,
            "section_id": section_id,
            "seed": SEED,
            "n_clusters": n_clusters,
            "final_key": None,
            "cluster_key": None,
            "ARI": np.nan,
            "AMI": np.nan,
            "Completeness": np.nan,
            "runtime_sec": elapsed,
            "cpu_percent": np.nan,
            "ram_mb": np.nan,
            "mem_peak_mb": np.nan,
            "status": f"error: {str(e)[:100]}",
        }

    all_records.append(record)

    # ---- Save run result CSV ----
    pd.DataFrame([record]).to_csv(
        section_dir / f"result_{safe_section_id}.csv",
        index=False,
    )

# ---- Save summary ----
df_all = pd.DataFrame(all_records)
df_all.to_csv(run_dir_root / f"results_{RUN_NAME}.csv", index=False)
df_all.to_csv(Path(base_output) / "all_runs_global.csv", index=False)

print("\n\n========== RUN COMPLETE ==========")
print(f"Results saved to: {run_dir_root}")
print("\n=== Results summary ===")
print(df_all[["run_name", "section_id", "seed", "ARI", "AMI", "Completeness"]])