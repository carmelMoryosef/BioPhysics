import re
import os
import shutil
import imagej
import numpy as np
from imagej._java import jimport
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile
import pandas as pd
import cv2
from extract_files import extract_and_rename_images 
from mask_enum import MaskType

# Initialize ImageJ once (reuse in both functions)
ij = imagej.init('sc.fiji:fiji', headless=False)
IJ = jimport('ij.IJ')
Prefs = jimport('ij.Prefs')
WM = jimport('ij.WindowManager')

# BASE_FOLDER = r"G:/My Drive/bio_physics/pictures"
BASE_FOLDER = "./data/basic_experiment/"
PICTURE_FOLDER = "/pictures/"
MASKS_FOLDER = "/masks/"
PHASE_SUFFIX = "Phase_100.tif"
MASK_PREFIX = "mask"
GFP_FILE_INCLUDES = "GFP"


def analyze_bacteria(image_path, with_pause=False, min_size=5, max_size=1e9):
    """
    Loads an image, thresholds, runs particle analysis, and returns bacteria count.
    Parameters:
        image_path: str - Path to the image file.
        min_size: float - Minimum particle size to include.
        max_size: float - Maximum particle size to include.
    Returns:
        count: int - Number of detected bacteria.
    """
    name = image_path.split("\\")[-1].split(".")[0]
    dir_res = "\\".join(str(os.path.dirname(image_path)).split("\\")[:-1])
    
    mask_path = fr"{os.path.dirname(image_path)}\masks\mask_{name}.tif"
    res_path = fr"{dir_res}\results\res_{name}.csv"

    if GFP_FILE_INCLUDES in name:
        shutil.copy2(image_path, mask_path)
        return

    image = ij.io().open(image_path)
    ij.ui().show("Image", image)

    imp = IJ.getImage()
    IJ.run(imp, "Color Balance...", "")
    IJ.run(imp, "Enhance Contrast", "saturated=0.35")
    IJ.run(imp, "Apply LUT", "")
    IJ.run("Close")
    
    imp.setAutoThreshold("Default dark no-reset")
    IJ.setRawThreshold(imp, 0, 65535)
    IJ.setRawThreshold(imp, 0, 65535)
    IJ.setRawThreshold(imp, 0, 200)
    IJ.setRawThreshold(imp, 0, 200)

    Prefs.blackBackground = True
    IJ.run(imp, "Convert to Mask", "")
    IJ.run(imp, "Analyze Particles...", "size=30-110 show=Masks exclude clear summarize overlay")
    IJ.saveAs("Results", res_path)
    mask_imp = IJ.getImage()
    imp.changes = False
    imp.close()
    IJ.saveAs(mask_imp, "Tiff", mask_path)

    if with_pause:
        input("Press Enter to close...")
    IJ.run("Close All")

    return mask_path


def show_marked_image(image_path, min_size=5, max_size=1e9):
    """
    Opens the image in an interactive Fiji window with outlines showing detected bacteria.
    Parameters:
        image_path: str - Path to the image file.
        min_size: float - Minimum particle size to include.
        max_size: float - Maximum particle size to include.
    """
    image = ij.io().open(image_path)
    image_gray = ij.op().run("convert.uint8", image)
    ij.ui().show("Gray Image", image_gray)

    # Apply histogram equalization or normalization
    equalized = ij.op().run("normalize", image)

    # Show result
    ij.ui().show("Auto Balanced (Simulated)", equalized)

    # Keep script running so the window stays open
    print("Threshold and analysis done. Leave this script running to inspect the window.")
    input("Press Enter to close...")


def show_image(image_path):
    image = ij.io().open(image_path)
    ij.ui().show("Image", image)
    input("Press Enter to close...")
    #david teaching me stuff


def analyze_all_pictures(image_folder):
    src_root = Path(image_folder)

    for image_path in src_root.glob("*.tif"):
        image_path_name = image_path.name
        print(f"[!] analyzing {image_path_name}...")
        # print(str(image_path.relative_to(".")))
        # rel_path = str(image_path.relative_to("."))
        # mask = analyze_bacteria(f"./{rel_path}")
        mask = analyze_bacteria(str(image_path.absolute()))
        # show_image(mask)
        #but its too late for my brain
    print(f"[V] Done!")


def load_tiff_grayscale(filepath):
    """
    Loads a TIFF file as a grayscale NumPy array.

    Args:
        filepath (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: A grayscale NumPy array representing the image, or None if an error occurs.
    """
    try:
        image = tifffile.imread(filepath)

        # If the image is not already grayscale, convert it
        if len(image.shape) > 2 and image.shape[-1] > 1:
            # Convert to grayscale (simple average of channels)
            image = np.mean(image, axis=-1)

        return image.astype(np.float32)  # Ensure it's float for further processing
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None

# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def mean_value_at_mask(image_path: str, mask_path: str, mask_type: MaskType, plot_each_image:bool=False) -> float:
    # Load and convert to grayscale
    image = Image.open(image_path)#.convert("L")
    mask = Image.open(mask_path).convert("L")

    image_array = np.array(image)
    mask_array = np.array(mask)

    # Sanity check: Ensure dimensions match
    if image_array.shape != mask_array.shape:
        raise ValueError("Image and mask must have the same dimensions.")

    # Find coordinates where mask is black
    y_coords, x_coords = np.where(mask_array == mask_type.value)
    pixel_values = image_array[y_coords, x_coords]
    # print(len(pixel_values))
    # print(image_path)
    # print(np.min(image_array), np.max(image_array))
    
    if plot_each_image:
        # Plot the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image_array, cmap='gray')
        plt.colorbar(label='Value')
        # plt.scatter(x_coords, y_coords, c='red', s=1, label='Black Mask Pixels')  # red overlay
        plt.title(f'Image for {os.path.basename(image_path)}')
        plt.axis('off')
        plt.show()

    return np.mean(pixel_values) #/ np.mean(image_array)


def extract_numeric_prefix(filename: str) -> int:
    """Extract the number before the first underscore."""
    match = re.match(rf"{MASK_PREFIX}_(\d+)_", filename)

    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"[X] Filename does not start with a number: {filename}")

def get_exposure(filename:str):
    return int(filename.split("_")[-1].split(".")[0])

def process_gfp_images(folder_path: str):
    """
    Finds all 'GFP' images in the folder, computes mean values for masked areas,
    and plots the results.
    
    Assumes:
      - Each GFP image has a corresponding mask with the same base name + mask_suffix.
      - The prefix before the first underscore is a number used for x-axis.
    """
    results = []

    for filename in os.listdir(folder_path):
        if GFP_FILE_INCLUDES in filename and filename.endswith(".tif"):
            base_name = os.path.split(filename)[0]
            # print("filename", filename, base_name)
            image_path = folder_path + filename  # os.path.join(folder_path, filename)
            mask_name = base_name + filename.split(GFP_FILE_INCLUDES)[0] + PHASE_SUFFIX
            mask_path = os.path.join(folder_path, mask_name)

            exposure = get_exposure(filename)
            # print(image_path, mask_path)
            # if os.path.exists(mask_path):
            try:
                x_value = extract_numeric_prefix(filename)
                # print(x_value)
                mean_val = mean_value_at_mask(image_path, mask_path, MaskType.BLACK)
                background_mean_val = mean_value_at_mask(image_path, mask_path, MaskType.WHITE)
                # print(mean_val, background_mean_val)
                if mean_val < background_mean_val:
                    print(f"[?] The background is lighter then the bacteria - file {filename}")
                results.append((x_value, mean_val - background_mean_val, exposure))# i added background simple value
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            # else:
            #     print(f"Mask not found for {filename}")

    # Sort by x-value (numeric prefix)
    results.sort(key=lambda x: x[0])
    # print(results)
    x_vals, y_vals, exposure_vals = zip(*results)

    cmap = plt.cm.rainbow
    norm = mpl.colors.Normalize(vmin=min(exposure_vals), vmax=max(exposure_vals))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals, y_vals, color=cmap(norm(exposure_vals)))
    # plt.scatter(x_vals, BG_vals, marker='*')#mean background
    plt.xlabel("Consentration")
    plt.ylabel("Mean value in non-black mask region")
    plt.title("Mean GFP Intensity over Time/Image Index")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # extract_and_rename_images(BASE_FOLDER, f"{BASE_FOLDER}/{PICTURE_FOLDER}")
    # analyze_bacteria(r"G:\My Drive\bio_physics\pictures\52_5_A_2_Phase_100.tif")
    # analyze_all_pictures(f"{BASE_FOLDER}/{PICTURE_FOLDER}")

    process_gfp_images(f"{BASE_FOLDER}/{PICTURE_FOLDER}/{MASKS_FOLDER}")
    # mean_val = mean_value_at_black_mask(r"G:\My Drive\bio_physics\pictures\210_B_3_GFP_3000.tif", r"G:\My Drive\bio_physics\pictures\masks\mask_52_5_A_2_Phase_100.tif")
    # mean_val = mean_value_at_black_mask(r"G:\My Drive\bio_physics\pictures\masks\mask_140_B_3_GFP_5000.tif", r"G:\My Drive\bio_physics\pictures\masks\mask_52_5_A_2_Phase_100.tif")

    # show_image(f"data/basic_experiment/pictures/masks/mask_210_B_1_Phase_100.tif")
    # show_image(r"G:\My Drive\bio_physics\pictures\masks\\mask_35_A_3_GFP_5000.tif")