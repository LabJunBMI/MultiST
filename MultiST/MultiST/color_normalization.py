#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color Normalization Module for Spatial Transcriptomics Images
============================================================

This module provides advanced color normalization techniques for H&E stained 
spatial transcriptomics images using intelligent target patch selection.


"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scanpy as sc
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class AdvancedTargetSelector:
    """
    Advanced Target Patch Selector for Color Normalization
    
    This class implements intelligent patch selection strategies to identify
    optimal target patches for color normalization in H&E stained images.
    """
    
    def __init__(self, selection_strategy='optimal_diverse', target_ratio=0.05):
        """
        Initialize the target selector
        
        Args:
            selection_strategy (str): Selection strategy
                - 'top_quality': Pure quality-based selection
                - 'optimal_diverse': Quality + diversity (recommended)
                - 'representative_clusters': Representative clustering
                - 'stain_quality': Based on H&E staining quality
            target_ratio (float): Target patch ratio (default 5%)
        """
        self.selection_strategy = selection_strategy
        self.target_ratio = target_ratio
    
    def calculate_comprehensive_score(self, patch):
        """
        Calculate comprehensive quality score for a patch
        
        Args:
            patch (np.ndarray): Input patch [H, W, 3]
            
        Returns:
            tuple: (total_score, detailed_scores)
        """
        scores = {}
        
        # 1. Tissue coverage ratio
        tissue_mask = np.sum(patch, axis=2) < 200 * 3
        scores['tissue_coverage'] = np.mean(tissue_mask)
        
        # 2. Contrast
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        scores['contrast'] = np.std(gray) / 255.0
        
        # 3. H&E staining quality
        scores['he_quality'] = self._assess_he_staining_quality(patch)
        
        # 4. Texture complexity
        scores['texture_complexity'] = self._calculate_texture_complexity(patch)
        
        # 5. Color diversity
        scores['color_diversity'] = self._calculate_color_diversity(patch)
        
        # Weighted comprehensive score
        weights = {
            'tissue_coverage': 0.3,
            'contrast': 0.2,
            'he_quality': 0.25,
            'texture_complexity': 0.15,
            'color_diversity': 0.1
        }
        
        total_score = sum(weights[k] * scores[k] for k in scores)
        
        return total_score, scores
    
    def _assess_he_staining_quality(self, patch):
        """
        Assess H&E staining quality
        
        Args:
            patch (np.ndarray): Input patch
            
        Returns:
            float: H&E staining quality score [0, 1]
        """
        try:
            # Convert to HSV space
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            
            # Detect Hematoxylin (blue-purple)
            blue_mask = ((h_channel >= 100) & (h_channel <= 140)) & (s_channel > 50)
            blue_ratio = np.mean(blue_mask)
            
            # Detect Eosin (pink)
            pink_mask = (((h_channel >= 150) & (h_channel <= 180)) | (h_channel <= 20)) & (s_channel > 50)
            pink_ratio = np.mean(pink_mask)
            
            # Ideal case: both blue (nuclei) and pink (cytoplasm)
            he_balance = min(blue_ratio + pink_ratio, 1.0)
            
            # Penalize over-staining or under-staining
            saturation_balance = 1.0 - abs(np.mean(s_channel/255.0) - 0.5) * 2
            
            return he_balance * 0.7 + saturation_balance * 0.3
        except:
            return 0.5  # Default medium score
    
    def _calculate_texture_complexity(self, patch):
        """
        Calculate texture complexity using Sobel edge detection
        
        Args:
            patch (np.ndarray): Input patch
            
        Returns:
            float: Texture complexity score [0, 1]
        """
        try:
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            
            # Use Sobel operator for edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize to [0,1]
            texture_score = np.mean(sobel_magnitude) / 255.0
            
            return min(texture_score, 1.0)
        except:
            return 0.5
    
    def _calculate_color_diversity(self, patch):
        """
        Calculate color diversity using RGB standard deviation
        
        Args:
            patch (np.ndarray): Input patch
            
        Returns:
            float: Color diversity score [0, 1]
        """
        try:
            # Reshape patch to pixel list
            pixels = patch.reshape(-1, 3)
            
            # Use RGB standard deviation as diversity metric
            diversity = np.mean(np.std(pixels, axis=0)) / 255.0
            
            return min(diversity, 1.0)
        except:
            return 0.5
    
    def extract_patch_features(self, patch):
        """
        Extract patch features for clustering
        
        Args:
            patch (np.ndarray): Input patch
            
        Returns:
            np.ndarray: Feature vector for clustering
        """
        try:
            # Color features
            rgb_mean = np.mean(patch.reshape(-1, 3), axis=0)
            rgb_std = np.std(patch.reshape(-1, 3), axis=0)
            
            # LAB space features
            lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            lab_mean = np.mean(lab.reshape(-1, 3), axis=0)
            
            # Texture features
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            texture_std = np.std(gray)
            
            return np.concatenate([rgb_mean, rgb_std, lab_mean, [texture_std]])
        except:
            return np.zeros(10)  # Default feature vector
    
    def select_target_patches(self, patches, strategy=None):
        """
        Select target patches for color normalization
        
        Args:
            patches (list): List of patches
            strategy (str, optional): Selection strategy override
            
        Returns:
            tuple: (target_patches, all_scores, target_indices)
        """
        if strategy is None:
            strategy = self.selection_strategy
        
        print(f" Using strategy '{strategy}' to select target patches...")
        print(f" Number of candidate patches: {len(patches)}")
        
        # Calculate quality scores for all patches
        print(" Calculating patch quality scores...")
        all_scores = []
        detailed_scores = []
        
        for i, patch in enumerate(patches):
            total_score, score_details = self.calculate_comprehensive_score(patch)
            all_scores.append(total_score)
            detailed_scores.append(score_details)
            
            if (i + 1) % 500 == 0:
                print(f"   Processed: {i+1}/{len(patches)}")
        
        all_scores = np.array(all_scores)
        
        # Select target patches based on strategy
        n_targets = max(5, int(len(patches) * self.target_ratio))
        
        if strategy == 'top_quality':
            # Pure quality-based selection
            top_indices = np.argsort(all_scores)[-n_targets:]
            target_indices = top_indices
            selection_info = f"Selected top {len(target_indices)} highest quality patches"
            
        elif strategy == 'optimal_diverse':
            # Quality + diversity selection
            candidate_ratio = 0.3
            n_candidates = max(n_targets, int(len(patches) * candidate_ratio))
            candidate_indices = np.argsort(all_scores)[-n_candidates:]
            
            # Select diverse patches from candidates
            target_indices = self._select_diverse_from_candidates(
                patches, candidate_indices, n_targets
            )
            selection_info = f"Selected diverse {len(target_indices)} patches from top {candidate_ratio*100:.0f}% candidates"
            
        elif strategy == 'stain_quality':
            # Focus on H&E staining quality
            he_scores = [details['he_quality'] for details in detailed_scores]
            he_scores = np.array(he_scores)
            
            # Select patches with best H&E staining quality
            target_indices = np.argsort(he_scores)[-n_targets:]
            selection_info = f"Selected {len(target_indices)} patches based on H&E staining quality"
        
        else:
            # Default to top_quality
            target_indices = np.argsort(all_scores)[-n_targets:]
            selection_info = f"Default quality selection: {len(target_indices)} patches"
        
        target_patches = [patches[i] for i in target_indices]
        
        # Output detailed information
        print(f" {selection_info}")
        if len(target_indices) > 0:
            target_scores = [all_scores[i] for i in target_indices]
            print(f" Average quality score: {np.mean(target_scores):.3f}")
            print(f" Score range: [{np.min(target_scores):.3f}, {np.max(target_scores):.3f}]")
        
        # Analyze target patch quality distribution
        self._analyze_target_quality(detailed_scores, target_indices)
        
        return target_patches, all_scores, target_indices
    
    def _select_diverse_from_candidates(self, patches, candidate_indices, n_targets):
        """
        Select diverse targets from candidate patches using clustering
        
        Args:
            patches (list): All patches
            candidate_indices (np.ndarray): Indices of candidate patches
            n_targets (int): Number of targets to select
            
        Returns:
            list: Selected target indices
        """
        try:
            # Extract features from candidate patches
            candidate_features = []
            for idx in candidate_indices:
                features = self.extract_patch_features(patches[idx])
                candidate_features.append(features)
            
            candidate_features = np.array(candidate_features)
            
            # Standardize features
            if len(candidate_features) > 1:
                scaler = StandardScaler()
                candidate_features_scaled = scaler.fit_transform(candidate_features)
                
                # Use KMeans clustering
                n_clusters = min(n_targets, len(candidate_indices))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(candidate_features_scaled)
                
                # Select patch closest to center from each cluster
                selected_indices = []
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_indices = candidate_indices[cluster_mask]
                    
                    if len(cluster_indices) > 0:
                        cluster_features = candidate_features_scaled[cluster_mask]
                        center = kmeans.cluster_centers_[cluster_id]
                        distances = np.linalg.norm(cluster_features - center, axis=1)
                        best_in_cluster = cluster_indices[np.argmin(distances)]
                        selected_indices.append(best_in_cluster)
                
                return selected_indices[:n_targets]
            else:
                return candidate_indices[:n_targets]
                
        except Exception as e:
            print(f" Diverse selection failed, falling back to quality selection: {e}")
            return candidate_indices[:n_targets]
    
    def _analyze_target_quality(self, detailed_scores, target_indices):
        """
        Analyze quality distribution of target patches
        
        Args:
            detailed_scores (list): Detailed score information
            target_indices (list): Indices of target patches
        """
        if len(target_indices) == 0:
            return
            
        print("\n Target patch quality analysis:")
        
        quality_metrics = ['tissue_coverage', 'contrast', 'he_quality', 'texture_complexity', 'color_diversity']
        
        for metric in quality_metrics:
            target_values = [detailed_scores[i][metric] for i in target_indices]
            avg_value = np.mean(target_values)
            std_value = np.std(target_values)
            print(f"   {metric}: {avg_value:.3f} (±{std_value:.3f})")


def debug_normalization_results(original_image, normalized_image, patches, valid_coords, patch_size):
    """
    Diagnose normalization results for debugging
    
    Args:
        original_image (np.ndarray): Original image
        normalized_image (np.ndarray): Normalized image  
        patches (list): List of patches
        valid_coords (list): Valid coordinates
        patch_size (int): Patch size
    """
    print(" Diagnosing normalization results:")
    
    if original_image is not None:
        print(f"Original image: {original_image.shape}, {original_image.dtype}, [{original_image.min()}, {original_image.max()}]")
    
    if normalized_image is not None:
        print(f"Normalized image: {normalized_image.shape}, {normalized_image.dtype}, [{normalized_image.min():.3f}, {normalized_image.max():.3f}]")
        print(f"Contains NaN: {np.isnan(normalized_image).any()}")
        print(f"Contains Inf: {np.isinf(normalized_image).any()}")
        
        # Check if all zeros or constant
        if np.all(normalized_image == 0):
            print("    Warning: Image is all zeros")
        elif np.all(normalized_image == normalized_image.flat[0]):
            print(f"    Warning: All pixels have same value ({normalized_image.flat[0]})")
    else:
        print(" Normalized image is None")


def fix_normalization_display_issues(normalized_image):
    """
    Fix normalization display issues for visualization
    
    Args:
        normalized_image (np.ndarray): Normalized image
        
    Returns:
        np.ndarray: Fixed image for display
    """
    if normalized_image is None:
        return None
    
    print("Fixing normalization display issues...")
    
    # Copy image to avoid modifying original data
    fixed = normalized_image.copy().astype(np.float32)
    
    # Handle invalid values
    fixed = np.nan_to_num(fixed, nan=0.0, posinf=255.0, neginf=0.0)
    
    print(f"Original range: [{fixed.min():.1f}, {fixed.max():.1f}]")
    
    # Adjust value range
    if fixed.max() > 1.0:
        # If values > 1, assume 0-255 range, convert to 0-1
        fixed = fixed / 255.0
        print("Converted to [0,1] range")
    
    # Ensure in [0,1] range
    fixed = np.clip(fixed, 0, 1)
    
    print(f"Fixed range: [{fixed.min():.3f}, {fixed.max():.3f}]")
    
    return fixed


def simple_color_normalization(image, target_patches):
    """
    Simplified color normalization using target patches
    
    Args:
        image (np.ndarray): Input image to normalize
        target_patches (list): List of target patches for normalization
        
    Returns:
        np.ndarray: Color normalized image
    """
    
    print(" Executing simplified color normalization...")
    
    if not target_patches:
        print(" No target patches available")
        return None
    
    try:
        # Calculate target patch statistics
        target_pixels = []
        for patch in target_patches:
            # Only take tissue region pixels
            tissue_mask = np.sum(patch, axis=2) < 200 * 3
            if np.any(tissue_mask):
                tissue_pixels_patch = patch[tissue_mask]
                target_pixels.append(tissue_pixels_patch)
        
        if not target_pixels:
            print(" No tissue pixels in target patches")
            return image
        
        combined_pixels = np.vstack(target_pixels)
        
        # Calculate target statistics
        target_mean = np.mean(combined_pixels, axis=0)
        target_std = np.std(combined_pixels, axis=0)
        
        print(f"Target statistics - Mean: {target_mean}, Std: {target_std}")
        
        # Apply statistical standardization to entire image
        image_float = image.astype(np.float32)
        
        # Only calculate statistics from tissue regions (exclude background)
        tissue_mask = np.sum(image, axis=2) < 200 * 3
        
        if np.sum(tissue_mask) > 1000:  # If sufficient tissue pixels
            tissue_pixels = image_float[tissue_mask]
            
            current_mean = np.mean(tissue_pixels, axis=0)
            current_std = np.std(tissue_pixels, axis=0)
            
            # Avoid division by zero
            current_std = np.maximum(current_std, 1e-6)
            target_std = np.maximum(target_std, 1e-6)
            
            print(f"Current statistics - Mean: {current_mean}, Std: {current_std}")
            
            # Normalize
            normalized = (image_float - current_mean) / current_std
            normalized = normalized * target_std + target_mean
            
            # Ensure valid range
            normalized = np.clip(normalized, 0, 255)
            
            print(f"Normalized range: [{normalized.min():.1f}, {normalized.max():.1f}]")
            
            return normalized.astype(np.uint8)
        else:
            print(" Insufficient tissue pixels, returning original image")
            return image
            
    except Exception as e:
        print(f" Normalization process error: {e}")
        import traceback
        traceback.print_exc()
        return None


def apply_color_normalization_to_image(image_path, adata, scale_factor=1.0, 
                                     patch_size=64, selection_strategy='optimal_diverse',
                                     target_ratio=0.05, save_images=True, output_dir=None):
    """
    Apply color normalization to spatial transcriptomics image
    
    Args:
        image_path (str): Path to the image file
        adata (anndata.AnnData): Spatial transcriptomics data
        scale_factor (float): Coordinate scaling factor
        patch_size (int): Size of patches for normalization
        selection_strategy (str): Target patch selection strategy
        target_ratio (float): Ratio of patches to use as targets
        save_images (bool): Whether to save processed images
        output_dir (str): Directory to save processed images
        
    Returns:
        tuple: (original_image, normalized_image, normalization_info)
    """
    
    print(" Starting color normalization workflow...")
    print("="*60)
    
    # Load original image
    print(f" Loading image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Use PIL to load image
        pil_img = Image.open(image_path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        original_image = np.array(pil_img)
        print(f" Image loaded successfully: {original_image.shape}, {original_image.dtype}")
        
    except Exception as e:
        print(f" Image loading failed: {e}")
        try:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            print(f" Fallback method loaded successfully: {original_image.shape}")
        except Exception as e2:
            raise Exception(f"All loading methods failed: {e2}")
    
    # Extract patches for normalization
    print(" Extracting patches for normalization...")
    
    # Get spatial coordinates
    spot_coords = adata.obsm['spatial'].astype(int)
    spot_coords = (spot_coords * scale_factor).astype(int)
    spot_coords = spot_coords[:, [1, 0]]  # [x,y] -> [y,x]
    
    h, w, _ = original_image.shape
    patches = []
    valid_coords = []
    
    def get_patch(img, center, size=64):
        y, x = center
        half = size // 2
        return img[y - half:y + half, x - half:x + half]
    
    skipped = 0
    for coord in spot_coords:
        y, x = coord
        half = patch_size // 2
        
        if (y - half < 0 or y + half > h or x - half < 0 or x + half > w):
            skipped += 1
            continue
        
        patch = get_patch(original_image, (y, x), patch_size)
        patches.append(patch)
        valid_coords.append((y, x))
    
    print(f" Extraction complete: {len(patches)} patches, skipped {skipped}")
    
    # Select target patches using advanced selector
    print(" Selecting target patches for normalization...")
    selector = AdvancedTargetSelector(
        selection_strategy=selection_strategy,
        target_ratio=target_ratio
    )
    
    target_patches, all_scores, target_indices = selector.select_target_patches(patches)
    
    # Perform color normalization
    print(" Performing color normalization...")
    try:
        normalized_image = simple_color_normalization(original_image, target_patches)
        if normalized_image is not None:
            print(f" Color normalization completed: {normalized_image.shape}")
        else:
            print(" Color normalization failed")
            normalized_image = original_image  # Fallback to original
            
    except Exception as e:
        print(f" Color normalization error: {e}")
        normalized_image = original_image  # Fallback to original
    
    # Save processed images if requested
    if save_images:
        print(" Saving processed images...")
        
        # Create output directory if not specified
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
            if not output_dir:
                output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save original image copy
            original_save_path = os.path.join(output_dir, f"{base_name}_original_{timestamp}.png")
            Image.fromarray(original_image.astype(np.uint8)).save(original_save_path)
            print(f" Original image saved: {original_save_path}")
            
            # Save normalized image
            normalized_save_path = os.path.join(output_dir, f"{base_name}_normalized_{timestamp}.png")
            Image.fromarray(normalized_image.astype(np.uint8)).save(normalized_save_path)
            print(f" Normalized image saved: {normalized_save_path}")
            
            # Save comparison image
            comparison_save_path = os.path.join(output_dir, f"{base_name}_comparison_{timestamp}.png")
            save_comparison_image(original_image, normalized_image, comparison_save_path, normalization_info)
            print(f" Comparison image saved: {comparison_save_path}")
            
            # Save target patches visualization
            if target_patches:
                patches_save_path = os.path.join(output_dir, f"{base_name}_target_patches_{timestamp}.png")
                save_target_patches_visualization(target_patches, all_scores, target_indices, patches_save_path)
                print(f" Target patches visualization saved: {patches_save_path}")
            
            # Update normalization info with saved paths
            normalization_info.update({
                'saved_images': {
                    'original_image_path': original_save_path,
                    'normalized_image_path': normalized_save_path,
                    'comparison_image_path': comparison_save_path,
                    'target_patches_path': patches_save_path if target_patches else None
                }
            })
            
        except Exception as e:
            print(f" Failed to save some images: {e}")
    
    # Compile normalization information
    normalization_info = {
        'target_patches_count': len(target_patches),
        'total_patches_count': len(patches),
        'target_ratio_actual': len(target_patches) / len(patches),
        'selection_strategy': selection_strategy,
        'target_scores': [all_scores[i] for i in target_indices] if target_indices else [],
        'overall_score_mean': np.mean(all_scores) if len(all_scores) > 0 else 0,
        'target_score_mean': np.mean([all_scores[i] for i in target_indices]) if target_indices else 0,
        'normalization_applied': normalized_image is not None and not np.array_equal(original_image, normalized_image)
    }
    
    print(" Color normalization workflow completed")
    print(f" Normalization info: {normalization_info['normalization_applied']}")
    
    return original_image, normalized_image, normalization_info


def visualize_normalization_results(original_image, normalized_image, normalization_info):
    """
    Visualize color normalization results
    
    Args:
        original_image (np.ndarray): Original image
        normalized_image (np.ndarray): Normalized image
        normalization_info (dict): Normalization information
    """
    if normalized_image is None:
        print(" Cannot visualize: normalized image is None")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Normalized image
    display_normalized = fix_normalization_display_issues(normalized_image)
    if display_normalized is not None:
        axes[1].imshow(display_normalized)
        axes[1].set_title("Normalized Image")
        axes[1].axis('off')
    
    plt.suptitle(f"Color Normalization Results\nStrategy: {normalization_info.get('selection_strategy', 'unknown')}")
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n Normalization Statistics:")
    print(f"   Target patches: {normalization_info.get('target_patches_count', 0)}")
    print(f"   Total patches: {normalization_info.get('total_patches_count', 0)}")
    print(f"   Target ratio: {normalization_info.get('target_ratio_actual', 0):.3f}")
    print(f"   Normalization applied: {normalization_info.get('normalization_applied', False)}")


def save_comparison_image(original_image, normalized_image, save_path, normalization_info):
    """
    Save side-by-side comparison of original and normalized images
    
    Args:
        original_image (np.ndarray): Original image
        normalized_image (np.ndarray): Normalized image
        save_path (str): Path to save comparison image
        normalization_info (dict): Normalization information
    """
    try:
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f"Original Image\nShape: {original_image.shape}", fontsize=14)
        axes[0].axis('off')
        
        # Normalized image  
        axes[1].imshow(normalized_image)
        axes[1].set_title(f"Normalized Image\nShape: {normalized_image.shape}", fontsize=14)
        axes[1].axis('off')
        
        # Add overall title with normalization info
        strategy = normalization_info.get('selection_strategy', 'unknown')
        target_count = normalization_info.get('target_patches_count', 0)
        applied = normalization_info.get('normalization_applied', False)
        
        plt.suptitle(f"Color Normalization Comparison\n"
                    f"Strategy: {strategy} | Target patches: {target_count} | Applied: {applied}", 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f" Failed to save comparison image: {e}")


def save_target_patches_visualization(target_patches, all_scores, target_indices, save_path):
    """
    Save visualization of selected target patches
    
    Args:
        target_patches (list): List of target patches
        all_scores (np.ndarray): Quality scores for all patches
        target_indices (list): Indices of target patches
        save_path (str): Path to save visualization
    """
    try:
        # Create patches grid
        n_patches = min(16, len(target_patches))  # Show up to 16 patches
        n_cols = 4
        n_rows = (n_patches + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_patches):
            row = i // n_cols
            col = i % n_cols
            
            # Show patch
            axes[row, col].imshow(target_patches[i])
            
            # Add score information
            if i < len(target_indices):
                target_idx = target_indices[i]
                score = all_scores[target_idx] if target_idx < len(all_scores) else 0
                axes[row, col].set_title(f"Patch {i+1}\nScore: {score:.3f}", fontsize=10)
            else:
                axes[row, col].set_title(f"Patch {i+1}", fontsize=10)
            
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_patches, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"Selected Target Patches for Color Normalization\n"
                    f"Total: {len(target_patches)} patches", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f" Failed to save target patches visualization: {e}")


def save_normalization_summary(normalization_info, save_path):
    """
    Save detailed normalization summary as text file
    
    Args:
        normalization_info (dict): Normalization information
        save_path (str): Path to save summary
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Color Normalization Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Selection Strategy: {normalization_info.get('selection_strategy', 'unknown')}\n")
            f.write(f"  Target Ratio: {normalization_info.get('target_ratio_actual', 0):.3f}\n")
            f.write(f"  Normalization Applied: {normalization_info.get('normalization_applied', False)}\n\n")
            
            f.write("Statistics:\n")
            f.write(f"  Total Patches: {normalization_info.get('total_patches_count', 0)}\n")
            f.write(f"  Target Patches: {normalization_info.get('target_patches_count', 0)}\n")
            f.write(f"  Overall Score Mean: {normalization_info.get('overall_score_mean', 0):.4f}\n")
            f.write(f"  Target Score Mean: {normalization_info.get('target_score_mean', 0):.4f}\n\n")
            
            if 'saved_images' in normalization_info:
                f.write("Saved Files:\n")
                for key, path in normalization_info['saved_images'].items():
                    if path:
                        f.write(f"  {key}: {path}\n")
        
        print(f" Normalization summary saved: {save_path}")
        
    except Exception as e:
        print(f" Failed to save normalization summary: {e}")


if __name__ == "__main__":
    print(" Color Normalization Module Loaded")
    print("="*50)
    print("Available functions:")
    print("  - AdvancedTargetSelector: Intelligent patch selection")
    print("  - simple_color_normalization: Color normalization algorithm")
    print("  - apply_color_normalization_to_image: Complete workflow")
    print("  - visualize_normalization_results: Result visualization")