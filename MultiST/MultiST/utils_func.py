#
import os
import torch
import numpy as np
import scanpy as sc

def adata_preprocess(adata_vis, min_cells=50, min_counts=10, pca_n_comps=200):
    adata_vis.layers['count'] = adata_vis.X.toarray()
    sc.pp.filter_genes(adata_vis, min_cells=min_cells)
    sc.pp.filter_genes(adata_vis, min_counts=min_counts)

#     adata_vis.obs['mean_exp'] = adata_vis.X.toarray().mean(axis=1)
#     adata_vis.var['mean_exp'] = adata_vis.X.toarray().mean(axis=0)
#
#     # Load scRNA-seq data
#     adata_ref = sc.read_h5ad('/home/xuhang/disco_500t/Projects/spTrans/data/reference_data/GSE144136_DLPFC/raw/processed_raw.h5ad')
#     adata_ref.obs['mean_exp'] = adata_ref.X.toarray().mean(axis=1)
#     adata_ref.var['mean_exp'] = adata_ref.X.toarray().mean(axis=0)
#     common_genes = np.intersect1d(adata_vis.var.index, adata_ref.var.index)
#     adata_vis = adata_vis[:, common_genes]
#     adata_ref = adata_ref[:, common_genes]
#     adata_vis.var['ref_mean_exp'] = adata_ref.var['mean_exp']
#     adata_vis.var['ratio'] = np.log10(adata_vis.var['mean_exp'] / adata_vis.var['ref_mean_exp']+1)
#     adata_vis.var['selected'] = adata_vis.var['ratio'] < 1.5
#     remain_genes = adata_vis.var[adata_vis.var['selected']==True].index.tolist()
#     adata_vis = adata_vis[:, remain_genes]
#
#
    sc.pp.normalize_total(adata_vis, target_sum=1e6)
    # sc.pp.log1p(adata_vis)
    sc.pp.highly_variable_genes(adata_vis, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata_vis = adata_vis[:, adata_vis.var['highly_variable'] == True]
    sc.pp.scale(adata_vis)

    from sklearn.decomposition import PCA
    adata_X = PCA(n_components=pca_n_comps, random_state=42).fit_transform(adata_vis.X)
    adata_vis.obsm['X_pca'] = adata_X
    return adata_vis


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn

    #seed = 666
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

def extract_spot_images(image_path, spot_coordinates, patch_size=224, scale_factor=1.0):
    """
    Extract centered patches for each spatial spot in a Visium image.
    """
    print("Extracting spot patches...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    coords = spot_coordinates * scale_factor
    padding = patch_size // 2
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    coords = coords + padding

    patches = []
    for y, x in tqdm(coords, desc="Extracting", unit="spot"):
        x, y = int(x), int(y)
        patch = image[y - padding:y + padding, x - padding:x + padding]
        patch = cv2.resize(patch, (patch_size, patch_size))
        patch = transforms.ToTensor()(patch)  # [3, H, W]
        patches.append(patch)

    return torch.stack(patches)

def extract_3x3_spot_patches(image_path, spot_coordinates, patch_size=224, scale_factor=1.0, spacing=224):
    """
    For each spot, extract a 3x3 grid of patches centered around the spot.

    Args:
        image_path (str): Path to tissue image.
        spot_coordinates (np.ndarray): [N, 2] array of spatial coordinates (y, x).
        patch_size (int): Size of each small patch (default 224).
        scale_factor (float): Scaling applied to coordinates.
        spacing (int): Pixel distance between adjacent patch centers (default = patch_size).

    Returns:
        torch.Tensor: Shape [N, 9, 3, patch_size, patch_size]
    """
    print("Extracting 3x3 patches around each spot...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    coords = spot_coordinates * scale_factor
    padding = patch_size + spacing  # ensure full 3x3 patch can be extracted
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    coords = coords + padding

    patches_all = []
    offsets = [(-spacing, -spacing), (-spacing, 0), (-spacing, spacing),
               (0, -spacing),   (0, 0),   (0, spacing),
               (spacing, -spacing), (spacing, 0), (spacing, spacing)]

    for y, x in tqdm(coords, desc="Extracting", unit="spot"):
        x, y = int(x), int(y)
        patches_3x3 = []
        for dy, dx in offsets:
            cx, cy = x + dx, y + dy
            patch = image[cy - patch_size//2:cy + patch_size//2,
                          cx - patch_size//2:cx + patch_size//2]
            patch = cv2.resize(patch, (patch_size, patch_size))
            patch = transforms.ToTensor()(patch)  # [3, H, W]
            patches_3x3.append(patch)
        patches_all.append(torch.stack(patches_3x3))  # [9, 3, H, W]

    return torch.stack(patches_all)


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

def build_hybrid_graph(adata, k=12, lambda_=0.5, use_pca=True):

    coords = adata.obsm['spatial']
    X_expr = adata.obsm['X_pca'] if use_pca else adata.X


    A_spatial = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
    A_spatial = A_spatial.toarray()


    sim_matrix = cosine_similarity(X_expr)
    np.fill_diagonal(sim_matrix, 0)
    A_expr = np.zeros_like(sim_matrix)
    for i in range(sim_matrix.shape[0]):
        topk_idx = np.argsort(sim_matrix[i])[-k:]
        A_expr[i, topk_idx] = sim_matrix[i, topk_idx]


    A_hybrid = lambda_ * A_spatial + (1 - lambda_) * A_expr


    A_hybrid = sp.coo_matrix(A_hybrid)
    rowsum = np.array(A_hybrid.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hybrid @ D_inv_sqrt
    A_norm = A_norm.tocoo()


    indices = torch.LongTensor(np.vstack((A_norm.row, A_norm.col)))
    values = torch.FloatTensor(A_norm.data)
    shape = A_norm.shape
    adj_norm = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()

    adj_label = torch.sparse_coo_tensor(
        torch.LongTensor(np.vstack((A_hybrid.row, A_hybrid.col))),
        torch.FloatTensor(A_hybrid.data),
        torch.Size(A_hybrid.shape)
        ).coalesce()
    adj_mask = adj_label  

    graph_dict = {
        'adj_norm': adj_norm,
        'adj_label': adj_label,
        'norm_value': 1.0,
        'mask': adj_mask 
    }
    # return graph_dict
    # graph_dict = {
    # 'adj_norm': adj_norm,
    # 'adj_label': adj_label,
    # 'norm_value': 1.0  
    # }
    return graph_dict

