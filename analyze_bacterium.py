import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology
from skimage.color import label2rgb
from analyze_ecoli_imagej import mean_value_at_mask, background_adjustments
from mask_enum import MaskType
from skimage.segmentation import find_boundaries

def detect_each_bacteria(image_path, show_plot=False):
    """
    Analyzes individual E. coli bacteria in a mask image.

    Parameters:
    - image_path: str, path to the binary (black/white) image mask
    - show_plot: bool, whether to display a plot of the labeled bacteria

    Returns:
    - bacteria_indices: List of lists, each containing flattened pixel indices of a single bacterium
    - labeled_image: Labeled image where each bacterium has a unique integer label
    - avg_intensities: List of average grayscale intensities for each bacterium
    """
    # Step 1: Load original grayscale image
    image = io.imread(image_path, as_gray=True)

    # Preserve the original for intensity measurements
    original_image = image.copy()

    # Step 2: Threshold to binary (for detecting bacteria)
    binary = image > 0.5

    # Step 3: Clean noise
    cleaned = morphology.remove_small_objects(binary, min_size=10)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=10)

    # Step 4: Label connected components
    labeled_image = measure.label(cleaned, connectivity=2)
    regions = measure.regionprops(labeled_image)

    # Step 5: Extract flattened indices and average intensity per bacterium
    shape = image.shape
    bacteria_indices = []
    avg_intensities = []

    for region in regions:
        coords = region.coords
        flat_indices = [np.ravel_multi_index((r, c), shape) for r, c in coords]
        bacteria_indices.append(flat_indices)

        # Compute average intensity using the original image
        intensity_values = [original_image[r, c] for r, c in coords]
        avg_intensities.append(np.mean(intensity_values))

    # Step 6: Visualize labeled regions
    if show_plot:
        colored_labels = label2rgb(labeled_image, bg_label=0)
        plt.figure(figsize=(8, 8))
        plt.imshow(colored_labels)
        plt.title("Colored Bacteria Regions")
        plt.axis("off")
        plt.show()

    return bacteria_indices, labeled_image, avg_intensities


# def compute_bacteria_intensities(image_path, bacteria_indices, mask_path, bg_gradient):
#     """
#     Computes average grayscale intensity for each bacterium in the given image.
#     Includes debug plots to visualize before and after background adjustment.
#
#     Parameters:
#     - image_path: Path to the image file
#     - bacteria_indices: List of lists with flattened pixel indices for each bacterium
#     - mask_path: Path to the mask file
#     - bg_gradient: Background gradient parameter
#
#     Returns:
#     - List of average intensities for each bacterium
#     """
#     image = io.imread(image_path, as_gray=True)
#     shape = image.shape
#
#     # Create debug visualization
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#     # Plot 1: Original image before background adjustment
#     axes[0].imshow(image, cmap='gray')
#     axes[0].set_title('Original Image (Before Background Adjustment)')
#     axes[0].axis('off')
#
#     # Apply background adjustment (moved outside the loop for efficiency)
#     image_adjusted = background_adjustments(image, bg_gradient)
#
#     # Plot 2: Image after background adjustment with negative pixel handling
#     # Create a custom colormap that shows negative values in red
#     display_image = image_adjusted.copy()
#
#     # Create RGB image for custom coloring
#     rgb_image = np.zeros((*shape, 3))
#
#     # Normal pixels (non-negative) - grayscale
#     positive_mask = image_adjusted >= 0
#     normalized_positive = np.clip(image_adjusted, 0, 1)  # Ensure values are in [0,1] range
#     rgb_image[positive_mask, 0] = normalized_positive[positive_mask]  # R
#     rgb_image[positive_mask, 1] = normalized_positive[positive_mask]  # G
#     rgb_image[positive_mask, 2] = normalized_positive[positive_mask]  # B
#
#     # Negative pixels - red color with intensity based on absolute value
#     negative_mask = image_adjusted < 0
#     if np.any(negative_mask):
#         # Normalize negative values for display (make them positive for intensity)
#         abs_negative = np.abs(image_adjusted[negative_mask])
#         max_abs_negative = np.max(abs_negative) if len(abs_negative) > 0 else 1
#         normalized_negative = abs_negative / max_abs_negative
#
#         rgb_image[negative_mask, 0] = normalized_negative  # R - red channel
#         rgb_image[negative_mask, 1] = 0  # G - no green
#         rgb_image[negative_mask, 2] = 0  # B - no blue
#
#         print(f"Warning: Found {np.sum(negative_mask)} negative pixels!")
#         print(f"Negative pixel value range: {np.min(image_adjusted)} to {np.max(image_adjusted[negative_mask])}")
#
#     axes[1].imshow(rgb_image)
#     axes[1].set_title('After Background Adjustment\n(Negative pixels in RED)')
#     axes[1].axis('off')
#
#     # Plot 3: Histogram of pixel values after adjustment
#     axes[2].hist(image_adjusted.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
#     axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
#     axes[2].set_xlabel('Pixel Intensity')
#     axes[2].set_ylabel('Frequency')
#     axes[2].set_title('Histogram of Adjusted Pixel Values')
#     axes[2].legend()
#     axes[2].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
#     # Print some debug statistics
#     print(f"\nDebug Statistics:")
#     print(f"Original image range: [{np.min(image):.4f}, {np.max(image):.4f}]")
#     print(f"Adjusted image range: [{np.min(image_adjusted):.4f}, {np.max(image_adjusted):.4f}]")
#     print(f"Number of negative pixels: {np.sum(image_adjusted < 0)}")
#     print(f"Percentage of negative pixels: {np.sum(image_adjusted < 0) / image_adjusted.size * 100:.2f}%")
#
#     # Convert flattened indices back to 2D and compute mean intensity
#     avg_intensities = []
#     for i, flat_index_list in enumerate(bacteria_indices):
#         coords = np.unravel_index(flat_index_list, shape)
#         intensity_values = image_adjusted[coords]
#         avg_intensity = np.mean(intensity_values)
#         avg_intensities.append(avg_intensity)
#
#         # Print per-bacterium debug info
#         print(f"Bacterium {i + 1}: {len(intensity_values)} pixels, "
#               f"intensity range [{np.min(intensity_values):.4f}, {np.max(intensity_values):.4f}], "
#               f"avg = {avg_intensity:.4f}")
#
#     return avg_intensities

def compute_bacteria_intensities(image_path, bacteria_indices, mask_path, bg_gradient):
    """
    Computes average grayscale intensity for each bacterium in the given image.

    Parameters:
    - image: 2D NumPy array (grayscale image)
    - bacteria_indices: List of lists with flattened pixel indices for each bacterium

    Returns:
    - List of average intensities for each bacterium
    """
    image = io.imread(image_path, as_gray=True)
    shape = image.shape
    image= background_adjustments(image, bg_gradient)


    # Convert flattened indices back to 2D and compute mean intensity
    avg_intensities = []
    for flat_index_list in bacteria_indices:
        coords = np.unravel_index(flat_index_list, shape)
        intensity_values = image[coords]
        # background_mean_val = mean_value_at_mask(image_path, mask_path, MaskType.WHITE, bg_gradient)
        avg_intens = np.mean(intensity_values)
        avg_intensities.append(avg_intens)

    return avg_intensities

# def compute_bacteria_intensities(image_path, bacteria_indices, mask_path, bg_gradient):
#     """
#     Computes background-subtracted average intensities of bacteria in an image
#     and displays:
#     1. Adjusted image
#     2. Image with red outlines on bacteria
#     3. Image with red outlines on non-bacteria
#
#     Parameters:
#     - image_path: str – path to TIFF image (16-bit or float)
#     - bacteria_indices: list of lists of flattened pixel indices per bacterium
#     - mask_path: str – for background reference
#     - bg_gradient: object used in background_adjustments()
#
#     Returns:
#     - List of average intensities per bacterium (background-subtracted)
#     """
#
#     # Load image (TIFF may be 16-bit)
#     image = io.imread(image_path).astype(np.float32)
#
#     # Step 1: Background adjustment (includes dark count subtraction etc.)
#     image_adj = background_adjustments(image, bg_gradient)
#
#     # Step 2: Save adjusted image for analysis (don't touch this)
#     image_adj_for_analysis = image_adj.copy()
#
#     # Step 3: Prepare display version (normalize just for visualization)
#     display_image = image_adj.copy()
#     min_val, max_val = np.min(display_image), np.max(display_image)
#     if max_val > min_val:
#         display_image = (display_image - min_val) / (max_val - min_val)
#     else:
#         display_image = np.zeros_like(display_image)
#
#     # Step 4: Build bacteria mask and compute average intensities
#     shape = image.shape
#     bacteria_mask = np.zeros(shape, dtype=bool)
#     avg_intensities = []
#
#     for flat_index_list in bacteria_indices:
#         coords = np.unravel_index(flat_index_list, shape)
#         bacteria_mask[coords] = True
#         intensity_values = image_adj_for_analysis[coords]
#         background_mean_val = mean_value_at_mask(image_path, mask_path, MaskType.WHITE, bg_gradient)
#         avg_intensities.append(np.mean(intensity_values) - background_mean_val)
#
#     # Step 5: Find boundaries
#     bacteria_boundary = find_boundaries(bacteria_mask, mode='inner')
#     non_bacteria_boundary = find_boundaries(~bacteria_mask, mode='inner')
#
#     # Step 6: Convert grayscale image to RGB for overlays
#     display_rgb = np.stack([display_image]*3, axis=-1)  # shape: (H, W, 3)
#     display_rgb = np.clip(display_rgb, 0, 1)  # make sure values are in range
#
#     # Step 7: Create red boundary overlays
#     bacteria_overlay = display_rgb.copy()
#     bacteria_overlay[bacteria_boundary] = [1, 0, 0]
#
#     non_bacteria_overlay = display_rgb.copy()
#     non_bacteria_overlay[non_bacteria_boundary] = [1, 0, 0]
#
#     # Step 8: Plot
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#
#     axs[0].imshow(display_image, cmap='gray')
#     axs[0].set_title("Adjusted Image")
#     axs[0].axis('off')
#
#     # axs[1].imshow(bacteria_overlay)
#     # axs[1].set_title("Bacteria Boundaries in Red")
#     # axs[1].axis('off')
#
#     axs[1].imshow(non_bacteria_overlay)
#     axs[1].set_title(f"Non-Bacteria Boundaries in Red {image_path}")
#     axs[1].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#     return avg_intensities
