import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology
from skimage.color import label2rgb


def detect_each_bacteria(image_path, show_plot=True):
    """
    Analyzes individual E. coli bacteria in a mask image.

    Parameters:
    - image_path: str, path to the binary (black/white) image mask
    - show_plot: bool, whether to display a plot of the labeled bacteria

    Returns:
    - bacteria_indices: List of lists, where each sublist contains
      the flattened pixel indices of a single bacterium
    - labeled_image: The labeled image where each bacterium has a unique integer label
    """
    # Step 1: Load grayscale image
    image = io.imread(image_path, as_gray=True)

    # Step 2: Threshold to binary
    binary = image > 0.5

    # Step 3: Clean noise
    cleaned = morphology.remove_small_objects(binary, min_size=10)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=10)

    # Step 4: Label connected components
    labeled_image = measure.label(cleaned, connectivity=2)
    regions = measure.regionprops(labeled_image)

    # Step 5: Extract flattened indices per bacterium
    shape = image.shape
    bacteria_indices = [
        [np.ravel_multi_index((r, c), shape) for r, c in region.coords]
        for region in regions
    ]

    # Step 6: Visualize colored labels
    if show_plot:
        colored_labels = label2rgb(labeled_image, bg_label=0)
        plt.figure(figsize=(8, 8))
        plt.imshow(colored_labels)
        plt.title("Colored Bacteria Regions")
        plt.axis("off")
        plt.show()

    return bacteria_indices, labeled_image

