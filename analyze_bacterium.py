import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology
from skimage.color import label2rgb
from analyze_ecoli_imagej import mean_value_at_mask, background_adjustments
from mask_enum import MaskType

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

    # Convert flattened indices back to 2D and compute mean intensity
    avg_intensities = []
    for flat_index_list in bacteria_indices:
        coords = np.unravel_index(flat_index_list, shape)
        image= background_adjustments(image, bg_gradient)
        intensity_values = image[coords]
        background_mean_val = mean_value_at_mask(image_path, mask_path, MaskType.WHITE, bg_gradient)
        avg_intensities.append(np.mean(intensity_values) - background_mean_val)

    return avg_intensities
