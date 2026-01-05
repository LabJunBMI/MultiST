
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import math
import cv2
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import warnings
import logging
import time
from datetime import datetime
import json
import scanpy as sc
import pandas as pd

# Import color normalization module
from MultiST.color_normalization import apply_color_normalization_to_image, visualize_normalization_results

warnings.filterwarnings('ignore')

# =====  Common Type Conversion Utilities =====
def ensure_tensor(data, dtype=torch.float32):
    """Ensure data is torch.Tensor type"""
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    else:
        return torch.tensor(data, dtype=dtype)

def ensure_numpy(data):
    """Ensure data is numpy.ndarray type"""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

# =====  Training Logger =====
class TrainingLogger:
    """Training logger - unified management of all log outputs"""
    
    def __init__(self, log_dir="logs", log_name=None, console_output=True):
        """
        Initialize training logger
        
        Args:
            log_dir: Log directory
            log_name: Log file name (without extension)
            console_output: Whether to output to console simultaneously
        """
        self.log_dir = log_dir
        self.console_output = console_output
        
        os.makedirs(log_dir, exist_ok=True)
        
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"dual_modal_MultiST_{timestamp}"
        
        self.log_file = os.path.join(log_dir, f"{log_name}.log")
        self.metrics_file = os.path.join(log_dir, f"{log_name}_metrics.json")
        
        # Configure log format
        self.logger = logging.getLogger("DualModalMultiST")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Metrics recording
        self.metrics = {
            'training_history': [],
            'timing_info': {},
            'model_config': {},
            'data_info': {},
            'normalization_info': {}
        }
        
        self.start_time = time.time()
        
        self.info("=" * 60)
        self.info(" Enhanced Dual-Modal Spatial Transcriptomics Clustering Training Started")
        self.info(f" Log file: {self.log_file}")
        self.info(f" Metrics file: {self.metrics_file}")
        self.info("=" * 60)
    
    def info(self, message):
        """Record info level log"""
        self.logger.info(message)
    
    def warning(self, message):
        """Record warning level log"""
        self.logger.warning(message)
    
    def error(self, message):
        """Record error level log"""
        self.logger.error(message)
    
    def debug(self, message):
        """Record debug level log"""
        self.logger.debug(message)
    
    def log_model_config(self, config):
        """Record model configuration"""
        self.metrics['model_config'] = config
        self.info("🔧 Model Configuration:")
        for key, value in config.items():
            self.info(f"   {key}: {value}")
    
    def log_data_info(self, data_info):
        """Record data information"""
        self.metrics['data_info'] = data_info
        self.info(" Data Information:")
        for key, value in data_info.items():
            self.info(f"   {key}: {value}")
    
    def log_normalization_info(self, norm_info):
        """Record color normalization information"""
        self.metrics['normalization_info'] = norm_info
        self.info(" Color Normalization Information:")
        for key, value in norm_info.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 5:
                self.info(f"   {key}: {type(value).__name__}[{len(value)}]")
            else:
                self.info(f"   {key}: {value}")
    
    def log_timing(self, stage, duration):
        """Record time consumption"""
        self.metrics['timing_info'][stage] = duration
        self.info(f" {stage}: {duration:.2f} seconds")
    
    def log_epoch_results(self, epoch, results):
        """Record training epoch results"""
        # Add to history
        epoch_data = {
            'epoch': epoch,
            'timestamp': time.time(),
            'total_loss': results['total_loss'],
            'stage': results['stage'],
            'losses': results['losses']
        }
        self.metrics['training_history'].append(epoch_data)
        
        # Write to log
        stage_emoji = {
            'dual_modal_training': ''
        }
        emoji = stage_emoji.get(results['stage'], '')
        
        self.info(f"{emoji} Epoch {epoch:3d}: Loss={results['total_loss']:.4f} | Stage={results['stage']}")
        
        # Detailed loss information
        if 'losses' in results:
            for loss_name, loss_value in results['losses'].items():
                if isinstance(loss_value, (int, float)):
                    self.info(f"     {loss_name}: {loss_value:.6f}")
    
    def log_evaluation_results(self, results):
        """Record evaluation results"""
        self.info(" Evaluation Results:")
        self.info(f"    Predicted labels shape: {results['predicted_labels'].shape}")
        self.info(f"    Fused features shape: {results['fused_features'].shape}")
        self.info(f"    Unique cluster count: {len(np.unique(results['predicted_labels']))}")
        
        # Cluster distribution statistics
        unique_labels, counts = np.unique(results['predicted_labels'], return_counts=True)
        self.info("    Cluster Distribution:")
        for label, count in zip(unique_labels, counts):
            self.info(f"      Cluster {label}: {count} spots ({count/len(results['predicted_labels'])*100:.1f}%)")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        # Calculate total training time
        total_time = time.time() - self.start_time
        self.metrics['timing_info']['total_training_time'] = total_time
        
        # Save to file
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False, default=str)
        
        self.info(f" Training metrics saved to: {self.metrics_file}")
    
    def finalize(self):
        """Finalize log recording"""
        total_time = time.time() - self.start_time
        self.info("=" * 60)
        self.info(f" Training Complete! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.info(f" Detailed log: {self.log_file}")
        self.info("=" * 60)
        
        # Save metrics
        self.save_metrics()

# =====  Spatial Cache Manager =====
class SpatialCacheManager:
    """Spatial relationship cache manager - precompute and cache KNN neighbor relationships"""
    
    def __init__(self, k_neighbors=8, logger=None):
        self.k_neighbors = k_neighbors
        self.cache = {}
        self.knn_cache = {}
        self.logger = logger
        
    def log(self, message, level='info'):
        """Record log"""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def get_cache_key(self, spatial_coords, k_neighbors=None):
        """Generate cache key"""
        if k_neighbors is None:
            k_neighbors = self.k_neighbors
        coords_array = ensure_numpy(spatial_coords)
        coords_hash = hash(coords_array.tobytes())
        return f"knn_{coords_hash}_{k_neighbors}"
    
    def build_and_cache_knn(self, spatial_coords, k_neighbors=None):
        """Build and cache KNN relationships"""
        if k_neighbors is None:
            k_neighbors = self.k_neighbors
            
        # Ensure spatial_coords is numpy array
        coords_array = ensure_numpy(spatial_coords)
        
        cache_key = self.get_cache_key(coords_array, k_neighbors)
        
        if cache_key not in self.knn_cache:
            self.log(f"🔧 Building KNN graph: k={k_neighbors}")
            start_time = time.time()
            
            nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1).fit(coords_array)
            distances, indices = nbrs.kneighbors(coords_array)
            
            # Remove self (first neighbor)
            neighbor_indices = indices[:, 1:]
            neighbor_distances = distances[:, 1:]
            
            # Calculate Gaussian weights
            sigma = np.median(neighbor_distances) * 2
            weights = np.exp(-neighbor_distances**2 / (2 * sigma**2))
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            self.knn_cache[cache_key] = {
                'neighbor_indices': neighbor_indices,
                'neighbor_distances': neighbor_distances,
                'weights': weights,
                'sigma': sigma
            }
            
            duration = time.time() - start_time
            self.log(f" KNN graph cached: {neighbor_indices.shape}, time: {duration:.2f} seconds")
        
        return self.knn_cache[cache_key]
    
    def get_spatial_graph(self, spatial_coords, k_neighbors=None):
        """Get spatial graph (from cache or new build)"""
        # Ensure spatial_coords is numpy array (KNN computation on CPU)
        if isinstance(spatial_coords, torch.Tensor):
            spatial_coords_numpy = spatial_coords.cpu().numpy()
        else:
            spatial_coords_numpy = ensure_numpy(spatial_coords)
            
        knn_data = self.build_and_cache_knn(spatial_coords_numpy, k_neighbors)
        return knn_data['neighbor_indices'], knn_data['weights']
    
    def clear_cache(self):
        """Clear cache"""
        self.knn_cache.clear()
        self.cache.clear()
        self.log("🧹 Spatial cache cleared")

# =====  Enhanced Image Patches Precomputer with Color Normalization =====
class EnhancedImagePatchesPrecomputer:
    """Enhanced image patches precomputer with integrated color normalization"""
    
    def __init__(self, patch_size=64, image_size=224, num_workers=4, 
                 apply_normalization=True, normalization_strategy='optimal_diverse',
                 target_ratio=0.05, save_images=True, logger=None):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.apply_normalization = apply_normalization
        self.normalization_strategy = normalization_strategy
        self.target_ratio = target_ratio
        self.save_images = save_images
        self.logger = logger
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Cache
        self.patches_cache = {}
        self.image_cache = {}
        self.normalization_cache = {}
        
        self.log(f" Enhanced image precomputer initialized: patch_size={patch_size}, image_size={image_size}, workers={num_workers}")
        self.log(f" Color normalization: {apply_normalization}, strategy={normalization_strategy}")
    
    def log(self, message, level='info'):
        """Record log"""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def load_and_prepare_image(self, image_path, adata, scale_factor=1.0, output_dir=None):
        """Load and prepare image with optional color normalization"""
        cache_key = f"{image_path}_{scale_factor}_{self.apply_normalization}_{self.normalization_strategy}"
        
        if cache_key not in self.image_cache:
            self.log(f" Loading and preparing image: {image_path}")
            start_time = time.time()
            
            if self.apply_normalization:
                # Apply color normalization with image saving
                self.log(" Applying color normalization...")
                try:
                    # Import with save_images parameter
                    from color_normalization import apply_color_normalization_to_image
                    
                    original_image, normalized_image, norm_info = apply_color_normalization_to_image(
                        image_path=image_path,
                        adata=adata,
                        scale_factor=scale_factor,
                        patch_size=self.patch_size,
                        selection_strategy=self.normalization_strategy,
                        target_ratio=self.target_ratio,
                        save_images=self.save_images,
                        output_dir=output_dir
                    )
                    
                    # Use normalized image if available, otherwise use original
                    if normalized_image is not None:
                        image = normalized_image
                        self.log(" Using color normalized image")
                        
                        # Log saved image paths if available
                        if 'saved_images' in norm_info:
                            saved_paths = norm_info['saved_images']
                            self.log(" Processed images saved:")
                            for img_type, path in saved_paths.items():
                                if path:
                                    self.log(f"   {img_type}: {path}")
                        
                        # Log normalization info
                        if self.logger:
                            self.logger.log_normalization_info(norm_info)
                        
                        # Store normalization info in cache
                        self.normalization_cache[cache_key] = norm_info
                    else:
                        image = original_image
                        self.log(" Color normalization failed, using original image")
                        
                except Exception as e:
                    self.log(f"Color normalization error: {e}, using original image", level='error')
                    # Fallback to loading original image
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Cannot read image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Load original image without normalization
                self.log(" Loading original image without normalization")
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Cannot read image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w, _ = image.shape
            
            # Add padding
            padding = self.patch_size // 2
            image = cv2.copyMakeBorder(
                image, padding, padding, padding, padding, cv2.BORDER_REFLECT
            )
            
            self.image_cache[cache_key] = {
                'image': image,
                'padding': padding,
                'original_shape': (h, w),
                'normalization_applied': self.apply_normalization
            }
            
            duration = time.time() - start_time
            self.log(f" Image cached: {image.shape}, time: {duration:.2f} seconds")
        
        return self.image_cache[cache_key]
    
    def extract_single_patch(self, image_data, coord, scale_factor=1.0):
        """Extract single patch from image"""
        image = image_data['image']
        padding = image_data['padding']
        
        # Adjust coordinates
        x, y = coord[0] * scale_factor + padding, coord[1] * scale_factor + padding
        x, y = int(x), int(y)
        
        # Extract patch
        patch = image[y - padding:y + padding, x - padding:x + padding]
        
        # Convert to PIL and apply transforms
        patch_pil = Image.fromarray(patch)
        patch_tensor = self.transform(patch_pil)
        
        return patch_tensor
    
    def precompute_all_patches(self, image_path, spatial_coords, adata, scale_factor=1.0, output_dir=None):
        """Precompute all patches - core optimization function with color normalization"""
        
        # Generate cache key
        coords_array = ensure_numpy(spatial_coords)
        coords_hash = hash(coords_array.tobytes())
        cache_key = f"{image_path}_{scale_factor}_{coords_hash}_{self.apply_normalization}_{self.normalization_strategy}"
        
        if cache_key not in self.patches_cache:
            self.log(f" Starting patch precomputation: {len(spatial_coords)} spots")
            total_start_time = time.time()
            
            # Load and prepare image (with optional color normalization and saving)
            image_data = self.load_and_prepare_image(image_path, adata, scale_factor, output_dir)
            
            # Parallel patch extraction
            all_patches = []
            
            if self.num_workers > 1:
                self.log(f" Multi-threaded parallel patch extraction: {self.num_workers} workers")
                # Multi-threaded version
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    for i, coord in enumerate(coords_array):
                        future = executor.submit(
                            self.extract_single_patch, image_data, coord, scale_factor
                        )
                        futures.append(future)
                    
                    # Collect results
                    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Extracting patches")):
                        patch = future.result()
                        all_patches.append(patch)
                        
                        # Regular progress logging
                        if (i + 1) % 1000 == 0:
                            self.log(f" Patches extracted: {i+1}/{len(futures)}")
            else:
                self.log(" Single-threaded patch extraction")
                # Single-threaded version
                for i, coord in enumerate(tqdm(coords_array, desc="Extracting patches")):
                    patch = self.extract_single_patch(image_data, coord, scale_factor)
                    all_patches.append(patch)
                    
                    # Regular progress logging
                    if (i + 1) % 1000 == 0:
                        self.log(f" Patches extracted: {i+1}/{len(coords_array)}")
            
            # Stack all patches
            self.log(" Stacking patches tensor...")
            all_patches_tensor = torch.stack(all_patches)
            
            # Cache results
            self.patches_cache[cache_key] = all_patches_tensor
            
            total_duration = time.time() - total_start_time
            memory_usage = all_patches_tensor.numel() * 4 / 1024**3  # GB
            
            self.log(f" Patch precomputation complete!")
            self.log(f"    Shape: {all_patches_tensor.shape}")
            self.log(f"    Memory: {memory_usage:.2f} GB")
            self.log(f"    Time: {total_duration:.2f} seconds")
            self.log(f"    Average speed: {len(spatial_coords)/total_duration:.1f} patches/second")
            
            # Log normalization info if available
            if cache_key in self.normalization_cache:
                norm_info = self.normalization_cache[cache_key]
                self.log(f"    Normalization applied: {norm_info.get('normalization_applied', False)}")
        
        return self.patches_cache[cache_key]
    
    def get_normalization_info(self, image_path, scale_factor=1.0):
        """Get normalization information for the processed image"""
        cache_key = f"{image_path}_{scale_factor}_{self.apply_normalization}_{self.normalization_strategy}"
        return self.normalization_cache.get(cache_key, {})
    
    def clear_cache(self):
        """Clear all caches"""
        self.patches_cache.clear()
        self.image_cache.clear()
        self.normalization_cache.clear()
        self.log(" Image cache cleared")

# =====  Optimized Spatial-Aware Image Encoder =====
class OptimizedSpatialAwareImageEncoder(nn.Module):
    """Optimized spatial neighborhood-aware image feature extractor using precomputed patches"""
    
    def __init__(self, feature_dim=512, k_neighbors=8, spatial_weight=0.3, 
                 patch_size=64, image_size=224, num_workers=4, 
                 apply_normalization=True, normalization_strategy='optimal_diverse',
                 save_images=True, logger=None):
        super(OptimizedSpatialAwareImageEncoder, self).__init__()
        
        self.logger = logger
        
        # Base image encoder
        try:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').vision_model
            self.projector = nn.Linear(self.clip.config.hidden_size, feature_dim)
            for param in self.clip.parameters():
                param.requires_grad = False
            self.use_clip = True
            self.log(" Using CLIP vision model")
        except ImportError:
            self.clip = models.resnet50(pretrained=True)
            self.clip.fc = nn.Linear(self.clip.fc.in_features, feature_dim)
            self.use_clip = False
            self.log(" Using ResNet50 model")
        
        self.k_neighbors = k_neighbors
        self.spatial_weight = spatial_weight
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.apply_normalization = apply_normalization
        self.save_images = save_images
        
        # Feature post-processing
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
        #  Optimization components
        self.patches_precomputer = EnhancedImagePatchesPrecomputer(
            patch_size, image_size, num_workers, 
            apply_normalization, normalization_strategy, save_images=save_images, logger=logger
        )
        self.spatial_cache = SpatialCacheManager(k_neighbors, logger)
        
        self.log(f" Optimized spatial-aware image encoder initialized: k={k_neighbors}, spatial_weight={spatial_weight}")
        self.log(f" Color normalization enabled: {apply_normalization}")
        self.log(f" Save processed images: {save_images}")
    
    def log(self, message, level='info'):
        """Record log"""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def precompute_patches(self, image_path, spatial_coords, adata, scale_factor=1.0, output_dir=None):
        """Precompute all patches with color normalization"""
        return self.patches_precomputer.precompute_all_patches(
            image_path, spatial_coords, adata, scale_factor, output_dir
        )
    
    def get_normalization_info(self, image_path, scale_factor=1.0):
        """Get color normalization information"""
        return self.patches_precomputer.get_normalization_info(image_path, scale_factor)
    
    def spatial_smoothing(self, features, spatial_coords):
        """Spatial neighborhood-based feature smoothing using cached KNN"""
        if spatial_coords is None:
            return features
        
        # Ensure spatial_coords on CPU for KNN computation
        if isinstance(spatial_coords, torch.Tensor):
            spatial_coords_cpu = spatial_coords.cpu()
        else:
            spatial_coords_cpu = spatial_coords
        
        #  Use cached KNN results
        neighbor_indices, weights = self.spatial_cache.get_spatial_graph(spatial_coords_cpu)
        
        # Convert to tensor and ensure on same device as features
        device = features.device
        neighbor_indices = torch.LongTensor(neighbor_indices).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Collect neighbor features
        neighbor_features = features[neighbor_indices]  # [N, k, D]
        weights = weights.unsqueeze(-1)  # [N, k, 1]
        
        # Weighted average
        smoothed_features = (neighbor_features * weights).sum(dim=1)  # [N, D]
        
        # Fuse with original features
        final_features = (1 - self.spatial_weight) * features + self.spatial_weight * smoothed_features
        
        return final_features
    
    def forward(self, image_patches, spatial_coords=None, batch_size=64):
        """
        Forward propagation using precomputed patches
        """
        
        if not isinstance(image_patches, torch.Tensor):
            raise ValueError("image_patches must be precomputed torch.Tensor")
        

        all_features = []
        num_patches = len(image_patches)
        device = next(self.parameters()).device
        
        self.log(f" Starting image feature extraction: {num_patches} patches, batch_size={batch_size}")
        
        for i in range(0, num_patches, batch_size):
            end_idx = min(i + batch_size, num_patches)
            batch_patches = image_patches[i:end_idx].to(device)
            
            # Extract base image features
            if self.use_clip:
                outputs = self.clip(pixel_values=batch_patches).pooler_output
                batch_features = self.projector(outputs)
            else:
                batch_features = self.clip(batch_patches)
            
            batch_features = self.dropout(self.norm(batch_features))
            all_features.append(batch_features)

            if (i // batch_size + 1) % 10 == 0:
                self.log(f" Feature extraction progress: {end_idx}/{num_patches}")
        
        # Merge all features
        features = torch.cat(all_features, dim=0)
        
        # Apply spatial smoothing
        if spatial_coords is not None:
            self.log("Applying spatial smoothing...")
            features = self.spatial_smoothing(features, spatial_coords)
        
        self.log(f" Image feature extraction complete: {features.shape}")
        return features
    
    def clear_cache(self):
        """Clear all caches"""
        self.patches_precomputer.clear_cache()
        self.spatial_cache.clear_cache()

# =====  Enhanced Bidirectional Cross-Attention Fusion with Weighted Fusion =====
class WeightedFlexibleCrossAttention(nn.Module):
    
    def __init__(self, gene_dim=32, image_dim=512, hidden_dim=256, num_heads=8, 
                 gene_weight=0.7):
        super(WeightedFlexibleCrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gene_weight = gene_weight  
        self.img_weight = 1 - gene_weight  
        
        self.gene_proj = nn.Linear(gene_dim, hidden_dim)
        self.img_proj = nn.Linear(image_dim, hidden_dim)
        
        self.img_key_proj = nn.Linear(image_dim, hidden_dim)
        self.img_value_proj = nn.Linear(image_dim, hidden_dim)
        self.gene_key_proj = nn.Linear(gene_dim, hidden_dim)
        self.gene_value_proj = nn.Linear(gene_dim, hidden_dim)
        
        self.gene_attended_to_original = nn.Linear(hidden_dim, gene_dim)
        
        # Image → Gene attention layers
        self.cross_attention_img2gene = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(2)
        ])
        
        # Gene → Image attention layers (for bidirectional mode)
        self.cross_attention_gene2img = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(2)
        ])
        
        self.norm_img = nn.LayerNorm(hidden_dim)
        self.norm_gene = nn.LayerNorm(hidden_dim)
        
        print(f" Weighted fusion cross-attention initialized: hidden_dim={hidden_dim}, heads={num_heads}")
        print(f"    Gene weight g={gene_weight:.1f}, Image weight (1-g)={self.img_weight:.1f}")
        print(f"   Bidirectional fusion formula: F_fused = {gene_weight:.1f} * F_gene_attended + {self.img_weight:.1f} * F_img_attended")
    
    def cross_modal_fusion(self, img_features, gene_features, mode="bidirectional"):
        """
         Cross-modal fusion with weighted fusion support
        
        Args:
            img_features: Image features [N, img_dim]
            gene_features: Gene features [N, gene_dim]
            mode: "bidirectional" or "img2gene"
        
        Returns:
            fused_features: Fused features
            img_attended: Image attention features
            gene_attended: Gene attention features
        """
        batch_size = img_features.shape[0]
        
        img_feat = img_features.unsqueeze(1)  # [N, 1, img_dim]
        gene_feat = gene_features.unsqueeze(1)  # [N, 1, gene_dim]
        
        # Image → Gene attention (always executed)
        q_gene = self.gene_proj(gene_feat)  # Query: gene features
        k_img = self.img_key_proj(img_feat)  # Key: image features
        v_img = self.img_value_proj(img_feat)  # Value: image features
        
        # Apply multi-layer attention
        for layer in self.cross_attention_img2gene:
            attended_output, attention_weights = layer(q_gene, k_img, v_img)
            q_gene = q_gene + attended_output
            q_gene = self.norm_gene(q_gene)
        
        gene_attended = q_gene.squeeze(1)  # [N, hidden_dim] - gene features enhanced by image information
        
        if mode == "bidirectional":
            # Bidirectional mode: simultaneously execute Gene → Image attention
            q_img = self.img_proj(img_feat)  # Query: image features
            k_gene = self.gene_key_proj(gene_feat)  # Key: gene features
            v_gene = self.gene_value_proj(gene_feat)  # Value: gene features
            
            # Apply multi-layer attention
            for layer in self.cross_attention_gene2img:
                attended_output, attention_weights = layer(q_img, k_gene, v_gene)
                q_img = q_img + attended_output
                q_img = self.norm_img(q_img)
            
            img_attended = q_img.squeeze(1)  # [N, hidden_dim] - image features enhanced by gene information
            
            #  Bidirectional fusion: weighted sum instead of concatenation
            # F_fused = g * F_gene_attended + (1-g) * F_img_attended
            fused = self.gene_weight * gene_attended + self.img_weight * img_attended  # [N, hidden_dim]
            
        elif mode == "img2gene":
            #  Unidirectional mode: use unprocessed original 32-dim gene features with attention features weighted sum
            # Project gene_attended back to original dimensions to match gene feature dimensions
            gene_attended_original_dim = self.gene_attended_to_original(gene_attended)  # [N, gene_dim=32]
            
            #  Weighted sum with completely unprocessed original gene features
            # Fusion in 32-dim space (preserve original gene features)
            fused_original_dim = self.gene_weight * gene_features + self.img_weight * gene_attended_original_dim  # [N, 32]

            # Project back to 256-dim for classifier compatibility
            fused = self.gene_proj(fused_original_dim.unsqueeze(1)).squeeze(1)  # [N, 256]
            img_attended = self.img_proj(img_feat).squeeze(1)  # Simple projection of image features
            
        else:
            raise ValueError(f"Unsupported mode: {mode}. Please choose 'bidirectional' or 'img2gene'")
        
        return fused, img_attended, gene_attended
    
    def forward(self, gene_features, image_features, mode="bidirectional"):
        return self.cross_modal_fusion(image_features, gene_features, mode)

# =====  Loss Functions =====
class DualModalSDMLoss(nn.Module):
    """Dual-modal similarity distribution matching loss"""
    
    def __init__(self, temperature=0.1):
        super(DualModalSDMLoss, self).__init__()
        self.temperature = temperature
        
    def compute_similarity_matrix(self, features, metric='cosine'):
        if metric == 'cosine':
            features_norm = F.normalize(features, p=2, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t())
        return similarity_matrix
    
    def similarity_to_distribution(self, similarity_matrix):
        distribution = F.softmax(similarity_matrix / self.temperature, dim=1)
        return distribution
    
    def forward(self, img_attended, gene_attended):
        if torch.isnan(img_attended).any() or torch.isinf(img_attended).any():
            return torch.tensor(0.0, device=img_attended.device), {}
        
        if torch.isnan(gene_attended).any() or torch.isinf(gene_attended).any():
            return torch.tensor(0.0, device=gene_attended.device), {}
        
        img_sim = self.compute_similarity_matrix(img_attended, 'cosine')
        gene_sim = self.compute_similarity_matrix(gene_attended, 'cosine')
        
        img_dist = self.similarity_to_distribution(img_sim)
        gene_dist = self.similarity_to_distribution(gene_sim)
        
        img_dist_log = torch.log(img_dist + 1e-8)
        gene_dist_log = torch.log(gene_dist + 1e-8)
        
        img_dist_log = torch.clamp(img_dist_log, -50, 50)
        gene_dist_log = torch.clamp(gene_dist_log, -50, 50)
        
        kl_img_to_gene = F.kl_div(img_dist_log, gene_dist, reduction='batchmean')
        kl_gene_to_img = F.kl_div(gene_dist_log, img_dist, reduction='batchmean')
        
        sdm_loss = (kl_img_to_gene + kl_gene_to_img) / 2
        
        return sdm_loss, {
            'kl_img_to_gene': kl_img_to_gene,
            'kl_gene_to_img': kl_gene_to_img
        }

class DualModalContrastiveLoss(nn.Module):
    """Cross-modal contrastive learning loss"""
    
    def __init__(self, temperature=0.1):
        super(DualModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, img_attended, gene_attended, pseudo_labels=None):
        try:
            batch_size = img_attended.shape[0]
            
            img_norm = F.normalize(img_attended, p=2, dim=1)
            gene_norm = F.normalize(gene_attended, p=2, dim=1)
            
            cross_similarity = torch.matmul(img_norm, gene_norm.T) / self.temperature
            cross_similarity = torch.clamp(cross_similarity, -50, 50)
            
            labels = torch.arange(batch_size, device=img_attended.device)
            
            img_to_gene_loss = F.cross_entropy(cross_similarity, labels)
            gene_to_img_loss = F.cross_entropy(cross_similarity.T, labels)
            
            cross_modal_loss = (img_to_gene_loss + gene_to_img_loss) / 2
            
            return cross_modal_loss, {
                'img_to_gene_loss': img_to_gene_loss,
                'gene_to_img_loss': gene_to_img_loss
            }
            
        except Exception as e:
            return torch.tensor(0.1, device=img_attended.device), {}

# =====  Enhanced Dual-Modal MultiST Model with Color Normalization =====
class EnhancedDualModalMultiST(nn.Module):
    
    def __init__(self, trained_MultiST_model, num_clusters,
                 image_dim=512, hidden_dim=256, spatial_k=8, spatial_weight=0.3,
                 patch_size=64, image_size=224, num_workers=4,
                 apply_normalization=True, normalization_strategy='optimal_diverse',
                 save_images=True, sdm_temperature=0.1, use_contrastive=True, 
                 attention_mode="bidirectional", gene_weight=0.7, logger=None):
        super(EnhancedDualModalMultiST, self).__init__()
        
        self.logger = logger
        self.attention_mode = attention_mode  # "bidirectional" or "img2gene"
        self.gene_weight = gene_weight 
        

        self.MultiST_model = trained_MultiST_model
        self.MultiST_model.eval()
        for param in self.MultiST_model.parameters():
            param.requires_grad = False
        
        self.num_clusters = num_clusters
        self.use_contrastive = use_contrastive
        gene_dim = self.MultiST_model.latent_dim
        
        #  Enhanced spatial-aware image encoder with color normalization
        self.image_encoder = OptimizedSpatialAwareImageEncoder(
            feature_dim=image_dim, 
            k_neighbors=spatial_k,
            spatial_weight=spatial_weight,
            patch_size=patch_size,
            image_size=image_size,
            num_workers=num_workers,
            apply_normalization=apply_normalization,
            normalization_strategy=normalization_strategy,
            save_images=save_images,
            logger=logger
        )
        
        # Image branch feature projector
        self.img_projector = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gene branch feature projector
        self.gene_projector = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.cross_attention = WeightedFlexibleCrossAttention(
            gene_dim=gene_dim, image_dim=image_dim, hidden_dim=hidden_dim, 
            gene_weight=gene_weight
        )
        
        fusion_input_dim = hidden_dim  # After weighted fusion, all are hidden_dim
        
        # Final fusion classifier (optional, for clustering)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_clusters)
        )
        
        # Loss functions
        self.sdm_loss = DualModalSDMLoss(temperature=sdm_temperature)
        if use_contrastive:
            self.contrastive_loss = DualModalContrastiveLoss()
        
        # Record model configuration
        config = {
            'num_clusters': num_clusters,
            'image_dim': image_dim,
            'hidden_dim': hidden_dim,
            'spatial_k': spatial_k,
            'spatial_weight': spatial_weight,
            'patch_size': patch_size,
            'image_size': image_size,
            'num_workers': num_workers,
            'apply_normalization': apply_normalization,
            'normalization_strategy': normalization_strategy,
            'save_images': save_images,
            'sdm_temperature': sdm_temperature,
            'use_contrastive': use_contrastive,
            'attention_mode': attention_mode,  
            'gene_weight': gene_weight, 
            'gene_dim': gene_dim
        }
        if logger:
            logger.log_model_config(config)
        
        self.log(" Enhanced dual-modal MultiST initialization complete")
        self.log("    Integrated color normalization preprocessing")
        self.log("    Weighted fusion cross-attention")
        self.log("    Focus on dual-modal feature fusion")
        self.log(f"    Cross-attention mode: {attention_mode}")
        self.log(f"    Gene weight g={gene_weight:.1f}, Image weight (1-g)={1-gene_weight:.1f}")
        self.log(f"    Color normalization: {apply_normalization}")
        
        if attention_mode == "img2gene":
            self.log("    Fusion formula: F_fused = F_gene + F_gene2img")
        elif attention_mode == "bidirectional":
            self.log(f"    Fusion formula: F_fused = {gene_weight:.1f} * F_gene_attended + {1-gene_weight:.1f} * F_img_attended")
    
    def log(self, message, level='info'):
        """Record log"""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def precompute_patches(self, image_path, spatial_coords, adata, scale_factor=1.0, output_dir=None):
        """Precompute all patches with color normalization"""
        return self.image_encoder.precompute_patches(image_path, spatial_coords, adata, scale_factor, output_dir)
    
    def get_normalization_info(self, image_path, scale_factor=1.0):
        """Get color normalization information"""
        return self.image_encoder.get_normalization_info(image_path, scale_factor)
    
    def set_attention_mode(self, mode):
        """Runtime setting of attention mode"""
        if mode not in ["bidirectional", "img2gene"]:
            raise ValueError(f"Unsupported attention mode: {mode}")
        
        old_mode = self.attention_mode
        self.attention_mode = mode
        
        self.log(f" Attention mode changed: {old_mode} → {mode}")
        
        if mode == "img2gene":
            self.log("    New fusion formula: F_fused = F_gene + F_gene2img")
        elif mode == "bidirectional":
            self.log(f"    New fusion formula: F_fused = {self.gene_weight:.1f} * F_gene_attended + {1-self.gene_weight:.1f} * F_img_attended")
    
    def set_gene_weight(self, gene_weight):
        """ Runtime adjustment of gene weight"""
        if not 0 <= gene_weight <= 1:
            raise ValueError(f"Gene weight must be in [0,1], current value: {gene_weight}")
        
        old_weight = self.gene_weight
        self.gene_weight = gene_weight
        
        # Update cross-attention module weights
        self.cross_attention.gene_weight = gene_weight
        self.cross_attention.img_weight = 1 - gene_weight
        
        self.log(f" Gene weight changed: {old_weight:.2f} → {gene_weight:.2f}")
        self.log(f"    New fusion weights: gene={gene_weight:.2f}, image={1-gene_weight:.2f}")
    
    def forward(self, gene_features, image_patches, spatial_coords=None, stage="fusion", attention_mode=None):
        """
        Forward propagation
        Args:
            gene_features: Gene features [N, gene_dim]
            image_patches: Precomputed patches [N, 3, H, W]
            spatial_coords: Spatial coordinates [N, 2]
            stage: "image", "gene", "fusion" controls which branch output to return
            attention_mode: Runtime specified attention mode, overrides default setting
        """
        results = {}
        
        # Use specified attention mode, if not specified use default mode
        current_mode = attention_mode if attention_mode is not None else self.attention_mode
        
        # Image branch
        if stage in ["image", "fusion"]:
            img_features = self.image_encoder(image_patches, spatial_coords)
            img_projected = self.img_projector(img_features)
            
            results.update({
                'img_features': img_features,        #  New: original image features
                'img_projected': img_projected
            })
        
        # Gene branch
        if stage in ["gene", "fusion"]:
            gene_projected = self.gene_projector(gene_features)
            
            results.update({
                'gene_projected': gene_projected
            })
        
        # Fusion branch
        if stage == "fusion":
            #  Weighted fusion cross-attention
            fused_features, img_attended, gene_attended = self.cross_attention.cross_modal_fusion(
                img_features, gene_features, mode=current_mode
            )
            
            # Final clustering output (optional)
            fusion_logits = self.fusion_classifier(fused_features)
            fusion_probs = F.softmax(fusion_logits, dim=1)
            
            results.update({
                'fused_features': fused_features,
                'img_attended': img_attended,
                'gene_attended': gene_attended,
                'fusion_logits': fusion_logits,
                'fusion_probs': fusion_probs,
                'attention_mode_used': current_mode,  # Record used attention mode
                'gene_weight_used': self.gene_weight  #  Record used gene weight
            })
            
            # Record fusion feature dimension information
            if self.logger and hasattr(self, '_log_fusion_info'):
                if not self._log_fusion_info:
                    self.log(f" Fusion feature dimensions: {fused_features.shape}")
                    self.log(f" Used attention mode: {current_mode}")
                    self.log(f" Used gene weight: {self.gene_weight:.2f}")
                    self._log_fusion_info = True
            elif not hasattr(self, '_log_fusion_info'):
                self._log_fusion_info = False
        
        return results
    
    def compute_losses(self, results):
        """Compute losses - simplified version, only SDM and contrastive learning losses"""
        losses = {}
        
        # SDM loss
        if 'img_attended' in results and 'gene_attended' in results:
            sdm_total, sdm_details = self.sdm_loss(
                results['img_attended'], results['gene_attended']
            )
            losses['sdm_total'] = sdm_total
            losses.update(sdm_details)
            
            # Contrastive learning loss
            if self.use_contrastive:
                contrastive_total, contrastive_details = self.contrastive_loss(
                    results['img_attended'], results['gene_attended']
                )
                losses['contrastive_total'] = contrastive_total
                losses.update(contrastive_details)
        
        # Optional: add feature regularization loss
        if 'img_projected' in results and 'gene_projected' in results:
            # L2 regularization
            img_l2 = torch.norm(results['img_projected'], p=2, dim=1).mean()
            gene_l2 = torch.norm(results['gene_projected'], p=2, dim=1).mean()
            losses['regularization'] = 0.01 * (img_l2 + gene_l2)
        
        return losses
    
    def clear_cache(self):
        """Clear all caches"""
        self.image_encoder.clear_cache()
        self.log("🧹 Model cache cleared")

# ===== Enhanced Trainer =====
class EnhancedTrainer:
    """Enhanced trainer with color normalization integration"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-4, 
                 batch_size=32, total_epochs=200, logger=None):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.logger = logger
        
        # Optimizer setup
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs
        )
        
        self.log(" Enhanced trainer initialization complete")
        self.log(f"    Total epochs: {total_epochs}")
        self.log(f"    Batch size: {batch_size}")
        self.log(f"    Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        self.log("    Integrated color normalization preprocessing")
    
    def log(self, message, level='info'):
        """Record log"""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def train_epoch(self, gene_features, image_patches, spatial_coords, epoch, 
                   attention_mode=None, gene_weight=None):
        """
        Single epoch training with color normalization
        
        Args:
            attention_mode: Can dynamically specify attention mode during training
            gene_weight:  Can dynamically specify gene weight during training
        """
        self.model.train()
        
        # Convert to tensor
        gene_features = ensure_tensor(gene_features)
        image_patches = ensure_tensor(image_patches)
        if spatial_coords is not None:
            spatial_coords = ensure_tensor(spatial_coords)
        
        #  If gene weight specified, dynamically adjust
        if gene_weight is not None:
            old_weight = self.model.gene_weight
            self.model.set_gene_weight(gene_weight)
            self.log(f" Epoch {epoch}: Dynamically adjust gene weight {old_weight:.2f} → {gene_weight:.2f}")
        
        # If attention mode specified, record log
        if attention_mode is not None:
            self.log(f" Epoch {epoch}: Using attention mode {attention_mode}")
        
        # Manual batching
        n_samples = len(gene_features)
        indices = torch.randperm(n_samples)
        
        total_loss = 0
        batch_losses = {}
        num_batches = 0
        
        batch_count = 0
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            gene_batch = gene_features[batch_indices].to(self.device)
            patches_batch = image_patches[batch_indices].to(self.device)
            
            # Get corresponding spatial coordinates
            if spatial_coords is not None:
                batch_spatial_coords = spatial_coords[batch_indices].to(self.device)
            else:
                batch_spatial_coords = None
            
            # Forward propagation - support dynamic attention mode
            results = self.model(
                gene_batch, 
                patches_batch,
                batch_spatial_coords, 
                stage="fusion",
                attention_mode=attention_mode  
            )
            
            # Compute losses - simplified version, no pseudo-labels and DEC needed
            losses = self.model.compute_losses(results)
            
 
            loss_weights = {
                'sdm_total': 2.0, 
                'contrastive_total': 1.0,
                'regularization': 0.5
            }
            
            batch_loss = sum(
                loss_weights.get(k, 0) * v for k, v in losses.items() 
                if k in loss_weights and torch.is_tensor(v)
            )
            
     
            if batch_loss == 0:
                self.log(" Batch loss is 0, skipping backpropagation", level='warning')
                continue
            
            # Backpropagation
            if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += batch_loss.item()
                for k, v in losses.items():
                    if torch.is_tensor(v):
                        batch_losses[k] = batch_losses.get(k, 0) + v.item()
                        
                num_batches += 1
            
            batch_count += 1
            if batch_count % 20 == 0:
                progress = (end_idx / n_samples) * 100
                self.log(f" Epoch {epoch} training progress: {progress:.1f}% ({end_idx}/{n_samples})")
        
        self.scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_losses = {k: v / num_batches for k, v in batch_losses.items()} if num_batches > 0 else {}
        
        return {
            'total_loss': avg_loss,
            'losses': avg_losses,
            'stage': 'dual_modal_training',
            'attention_mode': attention_mode if attention_mode else self.model.attention_mode,
            'gene_weight': self.model.gene_weight  
        }
    
    def evaluate(self, gene_features, image_patches, spatial_coords, attention_mode=None):
        """
        Model evaluation - return fusion features, can be used for clustering
        
        Args:
            attention_mode: Attention mode used during evaluation
        """
        self.log(" Starting model evaluation...")
        
        # If attention mode specified, record log
        if attention_mode is not None:
            self.log(f" Evaluation using attention mode: {attention_mode}")
        
        self.model.eval()
        
        # Convert to tensor
        gene_features = ensure_tensor(gene_features)
        image_patches = ensure_tensor(image_patches)
        if spatial_coords is not None:
            spatial_coords = ensure_tensor(spatial_coords)
        
        with torch.no_grad():
            results = self.model(
                gene_features.to(self.device),
                image_patches.to(self.device),
                spatial_coords.to(self.device) if spatial_coords is not None else None,
                stage="fusion",
                attention_mode=attention_mode 
            )
            
            # Extract clustering results from fusion features (if available)
            fusion_probs = results.get('fusion_probs', None)
            if fusion_probs is not None:
                fusion_probs = fusion_probs.cpu().numpy()
                predicted_labels = np.argmax(fusion_probs, axis=1)
            else:
                # If no prediction probabilities, use KMeans on fusion features for clustering
                from sklearn.cluster import KMeans
                fused_features_np = results['fused_features'].cpu().numpy()
                kmeans = KMeans(n_clusters=self.model.num_clusters, random_state=42)
                predicted_labels = kmeans.fit_predict(fused_features_np)
                # Create soft labels
                distances = kmeans.transform(fused_features_np)
                fusion_probs = np.exp(-distances) / np.exp(-distances).sum(axis=1, keepdims=True)
            
        #  Enhanced evaluation results with original gene features
        eval_results = {
            'fusion_probs': fusion_probs,
            'predicted_labels': predicted_labels,
            'fused_features': results['fused_features'].cpu().numpy(),
            'img_attended': results['img_attended'].cpu().numpy(),
            'gene_attended': results['gene_attended'].cpu().numpy(),
            'img_features': results['img_features'].cpu().numpy(),  # original image features
            'original_gene_features': gene_features.cpu().numpy(),  # original gene features (32-dim)
            'attention_mode_used': results.get('attention_mode_used', 'unknown'),  # used attention mode
            'gene_weight_used': results.get('gene_weight_used', 'unknown')  #  used gene weight
        }
        

        if self.logger:
            self.logger.log_evaluation_results(eval_results)
            self.log(f" Used attention mode: {eval_results['attention_mode_used']}")
            self.log(f" Used gene weight: {eval_results['gene_weight_used']}")
            self.log(f" Fusion feature dimensions: {eval_results['fused_features'].shape}")
            self.log(f" Image feature dimensions: {eval_results['img_features'].shape}")
            self.log(f" Original gene feature dimensions: {eval_results['original_gene_features'].shape}")
        
        return eval_results

# =====  Enhanced Training Main Function with Color Normalization =====
def train_enhanced_dual_modal_MultiST(model, gene_features, image_path, spatial_coords, adata,
                                  scale_factor=1.0, epochs=200, batch_size=32, 
                                  device='cuda', log_dir="logs", log_name=None,
                                  attention_mode=None, 
                                  epoch_attention_schedule=None,
                                  gene_weight_schedule=None,
                                  save_adata=True, save_images=True, output_path=None):
    """
    Args:
        model: EnhancedDualModalMultiST model
        gene_features: Gene features [N, gene_dim]
        image_path: Tissue slice image path (str)
        spatial_coords: Spatial coordinates [N, 2]
        adata: AnnData object for color normalization
        scale_factor: Coordinate scaling factor
        epochs: Training epochs
        batch_size: Batch size
        device: Device
        log_dir: Log directory
        log_name: Log file name
        attention_mode: Global attention mode ("bidirectional" or "img2gene")
        epoch_attention_schedule: Epoch to attention mode mapping dict {epoch: mode}
        gene_weight_schedule:  Epoch to gene weight mapping dict {epoch: weight}
        save_adata: Whether to save results to AnnData format
        save_images: Whether to save processed images
        output_path: Output file path for AnnData
    """
    
    #  Step 0: Initialize logging system
    logger = TrainingLogger(log_dir=log_dir, log_name=log_name, console_output=True)
    
    # Pass logger to model
    model.logger = logger
    
    # Convert to tensor
    gene_features = ensure_tensor(gene_features)
    if spatial_coords is not None:
        spatial_coords = ensure_tensor(spatial_coords)
    
    # Record data information
    data_info = {
        'n_spots': len(gene_features),
        'gene_dim': gene_features.shape[1],
        'image_path': image_path,
        'spatial_coords_shape': spatial_coords.shape if spatial_coords is not None else None,
        'scale_factor': scale_factor,
        'epochs': epochs,
        'batch_size': batch_size,
        'device': device,
        'attention_mode': attention_mode,
        'epoch_attention_schedule': epoch_attention_schedule,
        'gene_weight_schedule': gene_weight_schedule,
        'save_adata': save_adata,
        'save_images': save_images
    }
    logger.log_data_info(data_info)
    
    logger.info(" Starting enhanced dual-modal spatial transcriptomics clustering training")
    logger.info("    Integrated color normalization preprocessing")
    logger.info("    Dynamic gene weight adjustment support")
    
    # Record attention mode settings
    if attention_mode:
        logger.info(f"    Global attention mode: {attention_mode}")
    if epoch_attention_schedule:
        logger.info("    Dynamic attention scheduling:")
        for epoch, mode in epoch_attention_schedule.items():
            logger.info(f"      Epoch {epoch}: {mode}")
    
    #  Record gene weight scheduling
    if gene_weight_schedule:
        logger.info("    Dynamic gene weight scheduling:")
        for epoch, weight in gene_weight_schedule.items():
            logger.info(f"      Epoch {epoch}: g={weight:.2f}")
    
    #  Step 1: Precompute all patches with color normalization
    logger.info("=" * 50)
    logger.info(" Step 1: Precompute all image patches with color normalization")
    logger.info("=" * 50)
    
    start_time = time.time()
    image_patches = model.precompute_patches(image_path, spatial_coords, adata, scale_factor, output_dir=log_dir)
    precompute_time = time.time() - start_time
    
    # Get and log normalization information
    norm_info = model.get_normalization_info(image_path, scale_factor)
    if norm_info:
        logger.log_normalization_info(norm_info)
    
    logger.log_timing("Precompute patches with normalization", precompute_time)
    logger.info(f"    Patches shape: {image_patches.shape}")
    logger.info(f"    GPU memory usage: {image_patches.numel() * 4 / 1024**3:.2f} GB")
    logger.info(f"    Average speed: {len(spatial_coords)/precompute_time:.1f} patches/second")
    
    #  Step 2: Precompute and cache KNN relationships
    logger.info("=" * 50)
    logger.info(" Step 2: Precompute KNN spatial relationships")
    logger.info("=" * 50)
    
    start_time = time.time()
    # Trigger KNN cache building
    model.image_encoder.spatial_cache.build_and_cache_knn(spatial_coords)
    knn_time = time.time() - start_time
    
    logger.log_timing("Cache KNN relationships", knn_time)
    
    #  Step 3: Create trainer and start training
    logger.info("=" * 50)
    logger.info(" Step 3: Start enhanced dual-modal training")
    logger.info("=" * 50)
    
    trainer = EnhancedTrainer(
        model, device=device, batch_size=batch_size, total_epochs=epochs, logger=logger
    )
    
    # Start training
    train_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        #  Determine attention mode for current epoch
        current_attention_mode = None
        if epoch_attention_schedule and epoch in epoch_attention_schedule:
            current_attention_mode = epoch_attention_schedule[epoch]
        elif attention_mode:
            current_attention_mode = attention_mode
        
        #  Determine gene weight for current epoch
        current_gene_weight = None
        if gene_weight_schedule and epoch in gene_weight_schedule:
            current_gene_weight = gene_weight_schedule[epoch]
        
        # Train one epoch
        results = trainer.train_epoch(
            gene_features, image_patches, spatial_coords, epoch, 
            attention_mode=current_attention_mode,
            gene_weight=current_gene_weight  
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Record epoch results to log
        logger.log_epoch_results(epoch, results)
        
        # Regular detailed logging
        if epoch % 10 == 0 or epoch < 10:
            logger.info(f"   ⏱ Epoch time: {epoch_time:.2f} seconds")
            
            # Add detailed loss information
            if 'losses' in results:
                for loss_name, loss_value in results['losses'].items():
                    if isinstance(loss_value, (int, float)):
                        logger.info(f"     {loss_name}: {loss_value:.6f}")
            
            #  Record current gene weight
            if 'gene_weight' in results:
                logger.info(f"     Gene weight: {results['gene_weight']:.3f}")
    
    train_time = time.time() - train_start_time
    logger.log_timing("Training time", train_time)
    
    total_time = precompute_time + knn_time + train_time
    
    logger.info("=" * 50)
    logger.info("⚡ Training time statistics")
    logger.info("=" * 50)
    logger.info(f"    Precompute patches: {precompute_time:.2f} seconds")
    logger.info(f"   Cache KNN relationships: {knn_time:.2f} seconds")
    logger.info(f"   Training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    logger.info(f"    Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"   Average per epoch: {train_time/epochs:.2f} seconds")
    
    # Step 4: Final evaluation
    logger.info("=" * 50)
    logger.info(" Step 4: Final evaluation")
    logger.info("=" * 50)
    
    # Use attention mode from last epoch for evaluation
    final_attention_mode = None
    if epoch_attention_schedule and (epochs-1) in epoch_attention_schedule:
        final_attention_mode = epoch_attention_schedule[epochs-1]
    elif attention_mode:
        final_attention_mode = attention_mode
    
    eval_results = trainer.evaluate(gene_features, image_patches, spatial_coords, 
                                   attention_mode=final_attention_mode)
    
    # # Step 5: Save results to AnnData format
    # if save_adata:
    #     logger.info("=" * 50)
    #     logger.info(" Step 5: Save results to AnnData format")
    #     logger.info("=" * 50)
        
    #     if output_path is None:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         output_path = os.path.join(log_dir, f"enhanced_dual_modal_results_{timestamp}.h5ad")
        
    #     try:
    #         # Create enhanced AnnData with all results
    #         result_adata = adata.copy()
            
    #         # Add prediction results to observations
    #         result_adata.obs['predicted_clusters'] = eval_results['predicted_labels']
    #         result_adata.obs['attention_mode_used'] = eval_results['attention_mode_used']
    #         result_adata.obs['gene_weight_used'] = eval_results['gene_weight_used']
            
    #         # Add cluster probabilities
    #         fusion_probs = eval_results['fusion_probs']
    #         for i in range(fusion_probs.shape[1]):
    #             result_adata.obs[f'cluster_{i}_prob'] = fusion_probs[:, i]
            
    #         # Add all features to obsm (observations matrix)
    #         result_adata.obsm['fused_features'] = eval_results['fused_features']
    #         result_adata.obsm['img_attended'] = eval_results['img_attended']
    #         result_adata.obsm['gene_attended'] = eval_results['gene_attended']
    #         result_adata.obsm['img_features'] = eval_results['img_features']
    #         result_adata.obsm['original_gene_features'] = eval_results['original_gene_features'] 
            
    #         # Add metadata to uns (unstructured annotations)
    #         result_adata.uns['enhanced_dual_modal_results'] = {
    #             'training_date': datetime.now().isoformat(),
    #             'model_config': logger.metrics.get('model_config', {}),
    #             'normalization_info': norm_info if norm_info else {},
    #             'training_summary': {
    #                 'total_epochs': epochs,
    #                 'final_attention_mode': eval_results['attention_mode_used'],
    #                 'final_gene_weight': eval_results['gene_weight_used'],
    #                 'training_time_minutes': train_time / 60,
    #                 'color_normalization_applied': norm_info.get('normalization_applied', False) if norm_info else False
    #             },
    #             'feature_dimensions': {
    #                 'fused_features': eval_results['fused_features'].shape,
    #                 'img_attended': eval_results['img_attended'].shape,
    #                 'gene_attended': eval_results['gene_attended'].shape,
    #                 'img_features': eval_results['img_features'].shape,
    #                 'original_gene_features': eval_results['original_gene_features'].shape
    #             }
    #         }
            
    #         # Save enhanced AnnData file
    #         result_adata.write(output_path)
            
    #         logger.info(f" Results saved to: {output_path}")
    #         logger.info("\n Saved data structure:")
    #         logger.info(f"    Original data shape: {result_adata.shape}")
    #         logger.info(f"   Clusters found: {len(np.unique(eval_results['predicted_labels']))}")
    #         logger.info(f"   Observations (obs): {list(result_adata.obs.columns)}")
    #         logger.info(f"   Feature matrices (obsm): {list(result_adata.obsm.keys())}")
    #         logger.info(f"    Metadata (uns): {list(result_adata.uns.keys())}")
    #         logger.info(f"    Color normalization applied: {norm_info.get('normalization_applied', False) if norm_info else False}")
            
    #     except Exception as e:
    #         logger.error(f" Failed to save AnnData results: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    # Finalize logging
    logger.finalize()
    
    return trainer, eval_results, output_path if save_adata else None

# =====  Usage Example Function =====


if __name__ == "__main__":
    print(" Enhanced Dual-Modal MultiST with Color Normalization Module Loaded")
    print("="*80)
    print("Key Features:")
    print("   Integrated color normalization preprocessing")
    print("   Automatic saving of processed images")
    print("   Weighted cross-attention fusion")
    print("   Optimized spatial-aware encoding")
    print("   Comprehensive AnnData result saving")
    print("   Dynamic gene weight and attention mode scheduling")
    print("")
    print("To run example:")
    print("  trainer, eval_results, output_path = example_usage()")
    print("")
    print("Key Functions:")
    print("  - EnhancedDualModalMultiST: Main model class")
    print("  - train_enhanced_dual_modal_MultiST: Complete training workflow")
    print("  - Color normalization automatically applied during preprocessing")


