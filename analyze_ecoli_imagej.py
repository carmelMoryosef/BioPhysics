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
import re
import plotly.graph_objs as go
from scipy.stats import skew, kurtosis, skewnorm
from matplotlib.pyplot import legend
from requests import delete
from skimage.io import imread
from scipy.stats import skew, kurtosis
# import cv2
from extract_files import extract_and_rename_images
from mask_enum import MaskType
from os import path
import analyze_bacterium as bct
import json
from collections import defaultdict
from scipy.optimize import curve_fit
from plot_process_avg import plot_multiple_res_dicts_grouped
from fit_functions import model, hillEq
from consts import *
from scipy.stats import skewnorm


exclude_subfolders = ["BackGround", "trash_measurments", "CarmelShachar (1)", "CarmelShachar", "20250518"]


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
    dst_root_src = fr"{os.path.dirname(image_path)}\{MASKS_FOLDER}"
    dst_root = Path(dst_root_src)
    dst_root.mkdir(parents=True, exist_ok=True)

    mask_path = fr"{dst_root_src}\mask_{name}.tif"
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
    IJ.setRawThreshold(imp, 0, Threshold)
    IJ.setRawThreshold(imp, 0, Threshold)

    Prefs.blackBackground = True
    IJ.run(imp, "Convert to Mask", "")
    IJ.run(imp, "Analyze Particles...", "size=30-170 circularity=0.5-1 show=Masks exclude clear summarize overlay")
    # IJ.saveAs("Results", res_path)
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
    # david teaching me stuff


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
        # but its too late for my brain
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
def background_picture_gradient(bg_folder_path):
    """
    Computes the mean background image from all subfolders in BASE_FOLDER/BackGround/BG_folder_path.
    Assumes each folder contains another subfolder where the background image is located.
    also subtract darkcount value

    Args:
        BG_folder_pathg (str): Subdirectory inside BackGround containing folders with background image subfolders.

    Returns:
        np.ndarray: Averaged background image.
    """
    full_bg_dir = os.path.join(BASE_FOLDER, bg_folder_path)
    if not os.path.exists(full_bg_dir):
        raise FileNotFoundError(f"Background path not found: {full_bg_dir}")

    image_list = []

    for outer_folder_name in os.listdir(full_bg_dir):
        outer_folder_path = os.path.join(full_bg_dir, outer_folder_name)
        if not os.path.isdir(outer_folder_path):
            continue

        # Look for the inner subfolder
        inner_subfolders = [d for d in os.listdir(outer_folder_path) if
                            os.path.isdir(os.path.join(outer_folder_path, d))]
        if not inner_subfolders:
            print(f"[!] No subfolder inside {outer_folder_path}")
            continue

        inner_path = os.path.join(outer_folder_path, inner_subfolders[0])  # Take the first subfolder

        # Look for .tif image inside the inner folder
        tif_files = [f for f in os.listdir(inner_path) if f.lower().endswith(".tif")]
        if not tif_files:
            print(f"[!] No TIFF files found in {inner_path}")
            continue

        image_path = os.path.join(inner_path, tif_files[0])  # Assume one relevant image per sub-subfolder
        image_array = load_tiff_grayscale(image_path)

        if image_array is not None:
            image_list.append(image_array)
        else:
            print(f"[X] Failed to load {image_path}")

    if not image_list:
        raise ValueError("No valid background images found.")

    stacked_images = np.stack(image_list, axis=0)
    mean_image = np.mean(stacked_images, axis=0) - DARK_COUNT

    print(f"[V] Averaged {len(image_list)} background images.")
    return mean_image


def background_adjustments(image, bg_picture):
    # before extracting pixel correlated to mask, background reductions should be made in this order:
    # dark count subtract
    # dividing by mean BG picture
    image = image - DARK_COUNT
    if np.any(image < 0):
        print("negative after Dark count")
    image_withnoBG = image / (bg_picture / 10)
    if np.any(image_withnoBG < 0):
        print("negative after division")
    # plt.imshow(image_withnoBG, cmap="gray")
    return image_withnoBG


def mean_value_at_mask(image_path: str, mask_path: str, mask_type: MaskType, bg_picture: str,
                       plot_each_image: bool = False) -> float:
    # Load and convert to grayscale
    image = Image.open(image_path)  # .convert("L")
    mask = Image.open(mask_path).convert("L")

    image_array = np.array(image)
    image_array = background_adjustments(image_array, bg_picture)
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

    return [np.mean(pixel_values),np.std(pixel_values/np.mean(pixel_values))]  # / np.mean(image_array)


def extract_numeric_prefix(filename: str) -> int:
    """Extract the number before the first underscore."""
    match = re.match(rf"{MASK_PREFIX}_(\d+)_([\d_]+)_([A-Za-z])_", filename)

    if match:
        return float(".".join(match.group(2).split("_")))
    else:
        raise ValueError(f"[X] Filename does not start with a number: {filename}")


def get_exposure(filename: str):
    return int(filename.split("_")[-1].split(".")[0])


def process_gfp_images(folder_path: str,UpperBound):
    """
    Finds all 'GFP' images in the folder, computes mean values for masked areas,
    and plots the results.

    Assumes:
      - Each GFP image has a corresponding 'phase' image starting with the same
        prefix and containing 'phase' (case-insensitive) somewhere after.
      - The prefix may include underscores, and GFP filenames may have extra parts.
    """
    results = []
    bg_gradient = background_picture_gradient(BACKGROUND)

    all_files = os.listdir(folder_path)

    for filename in all_files:
        if GFP_FILE_INCLUDES in filename and filename.endswith(".tif"):
            try:
                image_path = os.path.join(folder_path, filename)

                # Try to extract prefix before "_GFP" or other suffix
                prefix_match = re.match(r"(.+?)_(\d+)_GFP", filename)
                dig = prefix_match.group(2)
                prefix = prefix_match.group(1) if prefix_match else filename.split(GFP_FILE_INCLUDES)[0]
                if len(dig) == 1:
                    prefix = prefix + f"_{dig}"
            except Exception as e:
                print(f"Error in finding prefix {filename}: {e}")

            # Search for a matching phase file
            phase_file = None
            for candidate in all_files:
                if (candidate.startswith(prefix) and
                        # re.search(r'phase', candidate, re.IGNORECASE) and
                        not re.search(GFP_FILE_INCLUDES, candidate) and
                        candidate.endswith(".tif")):
                    phase_file = candidate
                    break

            if not phase_file:
                print(f"[!] No matching phase file found for {filename} with prefix {prefix}")
                continue

            mask_path = os.path.join(folder_path, phase_file)
            exposure = get_exposure(filename)

            try:
                x_value = extract_numeric_prefix(filename)
                # print(x_value)
                mean_val,std_val = mean_value_at_mask(image_path, mask_path, MaskType.BLACK, bg_gradient)
                background_mean_val,BG_std_val = mean_value_at_mask(image_path, mask_path, MaskType.WHITE, bg_gradient)
                # print(mean_val, background_mean_val)
                if mean_val < background_mean_val:
                    print(
                        f"[?] The background is lighter then the bacteria - file {filename}, diff: {background_mean_val - mean_val}")
                results.append((x_value, (((mean_val)/ background_mean_val)) * N, (BG_std_val * N )/2,
                                exposure))  # i added background simple value
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not results:
        print("[!] No valid results to plot.")
        return
        # Sort by x-value (numeric prefix)
    results.sort(key=lambda x: x[0])
    # print(results)
    x_vals, y_vals, BG_std_val, exposure_vals = zip(*results)

    # Convert to numpy arrays for indexing
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    exposure_vals = np.array(exposure_vals)

    # Color map settings
    cmap = plt.cm.rainbow
    norm = mpl.colors.Normalize(vmin=min(exposure_vals), vmax=max(exposure_vals))

    # Unique exposures
    unique_exposures = np.unique(exposure_vals)

    # Plot all fits
    plt.figure(figsize=(8, 5))

    res_dict = dict()

    for exp in unique_exposures:
        # Select data for this exposure
        mask = exposure_vals == exp
        x_exp = x_vals[mask]
        y_exp = y_vals[mask]
        # Group y values by x
        x_to_ys = defaultdict(list)
        for x_val, y_val in zip(x_exp, y_exp):
            x_to_ys[x_val].append(y_val)

        # Compute average y for each unique x
        x_avg = []
        y_avg = []
        for x_val in sorted(x_to_ys.keys()):
            ys = x_to_ys[x_val]
            x_avg.append(x_val)
            y_avg.append(np.mean(ys))

        # Convert to numpy arrays for fitting
        x_exp = np.array(x_avg)
        y_exp = np.array(y_avg)
        y_exp_60 = y_exp#[(x_exp <= 50) | (x_exp >= 70)]
        x_exp_60 = x_exp#[(x_exp <= 50) | (x_exp >= 70)]

        res_dict[exp] = dict(x=x_exp_60, y=y_exp_60,error=BG_std_val)
        # try to do somthing for hill coefficient
        # R = (y_exp - y_exp[x_exp.argmin()]) / (y_exp[y_exp.argmax()] - y_exp[x_exp.argmin()])  # y_exp[x_exp.argmin()]
        # Fit
        initial_guess = [7, 100, 0.5, 2, 0, 1]  # D, R, C, n, Y0
        lower_bounds = [0, 0, 0, 1, -1000, -1000]  # R >= 0 enforced here
        upper_bounds = [np.inf] * 3 + [UpperBound, np.inf, np.min(x_exp)]
        # initial_guess_hill = [1, 6]  # Kd,n
        # lower_bounds_hill = [0, 0.9]  # R >= 0 enforced here
        # upper_bounds_hill = [np.inf, 9]
        # try:
        #     params, cov = curve_fit(
        #         model, x_exp_60, y_exp_60,
        #         p0=initial_guess,
        #         bounds=(lower_bounds, upper_bounds),
        #         maxfev=100000)
        #     # params_hill, cov_hill = curve_fit(
        #     #     hillEq, x_exp, R,
        #     #     p0=initial_guess_hill,
        #     #     bounds=(lower_bounds_hill, upper_bounds_hill),
        #     #     maxfev=100000)
        #
        #     # Unpack parameters and errors
        #     D_fit, R_fit, C_fit, n_fit, Y0_fit, X0_fit = params
        #     # Kd_hillfit, n_hillfit = params_hill
        #     # Format label with ± uncertainties
        #     # label = (f'Exp {exp} | Kd={Kd_hillfit:.1f}, '
        #     #          f'n={n_hillfit:.2f}')
        #     label = (f'Exp {exp} | D={D_fit:.1f}, '
        #              f'R={R_fit:.1f}'
        #              f'C={C_fit:.4f}'
        #              f'n={n_fit:.1f}'
        #              f'Y₀={Y0_fit:.2f}')

            # uncertainties = np.sqrt(np.diag(covariance))
        # except RuntimeError:
        #     print(f"Fit failed for exposure {exp} param= {params}")
        #     continue

        # Plot
    #     color = cmap(norm(exp))
    #     plt.scatter(x_exp, y_exp, color=color)
    #     # plt.scatter(x_exp, R, color=color)
    #
    #     x_fit = np.linspace(min(x_exp), max(x_exp), 500)
    #     y_fit = model(x_fit, *params)
    #     # x_hill = np.linspace(min(x_exp), max(x_exp), 500)
    #     # R_hill = hillEq(x_hill, *params_hill)
    #     # plt.axvline(Kd_hillfit, color=color)
    #     plt.plot(x_fit, y_fit, color=color, label=label)
    #     # plt.plot(x_hill, R_hill, color=color, label=label)
    # # plt.plot(x_fit, y_fit, color='b', label='Fit')
    # # plt.axhline(0.5,color="black")
    # plt.legend()
    #
    # # plt.scatter(x_vals, background_mean_val, marker="*", color=cmap(norm(exposure_vals)))
    # # add y line at avrage of background_mean_val
    # # plt.plot([min(x_vals), max(x_vals)],[np.mean(background_mean_val),np.mean(background_mean_val)], color=cmap(norm(exposure_vals)))
    # # plt.ylim((0,1100))
    # # plt.scatter(x_vals, BG_vals, marker='*')#mean background
    # plt.xlabel("Inducer")
    # plt.ylabel("Mean value in non-black mask region")
    # plt.ylabel("R")
    # plt.title("Mean GFP Intensity over Time/Image Index")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"./figures/fit_N{N}_threshold{Threshold}_{Ind}_BGnorm_no10.png")
    # # plt.show()
    # # plt.savefig(f"./figures/fit_{N}_threshold{Threshold}.png")
    return res_dict


def underscore_to_point(s: str) -> str:
    """
    Converts underscores in a string to periods.
    """
    return float(s.replace('_', '.'))


def point_to_underscore(s: str) -> str:
    """
    Converts periods in a string back to underscores.
    """
    return str(s).replace('.', '_')


# TODO remove code duplications
def process_gfp_TMG_images(folder_path: str):
    """
    Finds all 'GFP' images in the folder, computes mean values for masked areas,
    and plots the results.

    Assumes:
      - Each GFP image has a corresponding 'phase' image starting with the same
        prefix and containing 'phase' (case-insensitive) somewhere after.
      - The prefix may include underscores, and GFP filenames may have extra parts.
    """
    # bg_gradient = background_picture_gradient(BACKGROUND)

    all_files = os.listdir(folder_path)
    all_ave_bact = {}
    # pattern = r'mask_\d+_(\d+_\d+_[A-Z])_(\d+)_GFP'
    # pattern = r'mask_(?:_(\d+))?_(?:TMG|TMD)_(\d+)_GFP_(?:_(\d+))?(3000|5000|800|100)'
    # pattern = r'(.+?)_(\d+_\d+)_[A-Z]_(?:TMG|TMD)_(?:\d+)_GFP_(?:_(\d+))?(3000|5000|800|100)'
    # pattern = r'(.+?)_(\d+_\d+)_[A-Z](?:_(?:\d+))?_(?:TMG|TMD|GFP)(?:_\d+)?_(?:GFP|TMG)_(?:_(\d+))?(?:.+)?(3000|5000|800|100)'
    pattern = r'(.+?)_(\d+_\d+)_[A-Z](?:_(?:\d+))?_(?:TMG|TMD|GFP)(?:_\d+)?(?:\_GFP|TMG)?_(?:_(\d+))?(?:.+)?(3000|5000|800|100)'
    bg_gradient = background_picture_gradient(BACKGROUND)

    for filename in all_files:
        if GFP_FILE_INCLUDES in filename and filename.endswith(".tif"):
            try:
                image_path = os.path.join(folder_path, filename)
                print("im still working")

                match = re.search(pattern, filename)
                inducer = extract_numeric_prefix(filename)
                exposure = match.group(4)

                # Try to extract prefix before "_GFP" or other suffix
                # (.+?)_(\d+)_(?:TMG|TMD)_(\d+)_GFP
                prefix_match = re.match(r"(.+?)_(\d+)_GFP", filename)
                # prefix_match = re.match(r"(.+?)(?:_(\d+))?_(?:TMG|TMD)_(\d+)_GFP", filename)
                dig = prefix_match.group(2)
                prefix = prefix_match.group(1) if prefix_match else filename.split(GFP_FILE_INCLUDES)[0]
                if len(dig) == 1:
                    prefix = prefix + f"_{dig}"
            except Exception as e:
                print(f"Error in finding prefix {filename}: {e}")
                continue

            # Search for a matching phase file
            phase_file = None
            for candidate in all_files:
                if (candidate.startswith(prefix) and
                        # re.search(r'phase', candidate, re.IGNORECASE) and
                        not re.search(GFP_FILE_INCLUDES, candidate) and
                        candidate.endswith(".tif")):
                    phase_file = candidate
                    break

            if not phase_file:
                print(f"[!] No matching phase file found for {filename} with prefix {prefix}")
                continue

            mask_path = os.path.join(folder_path, phase_file)
            # exposure = get_exposure(filename)

            try:
                # x_value = extract_numeric_prefix(filename)
                # print(x_value)
                indices, labeled, aveMaskBacterium = bct.detect_each_bacteria(mask_path)
                avebacterium = bct.compute_bacteria_intensities(image_path, indices, mask_path, bg_gradient)
                # mean_val = mean_value_at_mask(image_path, mask_path, MaskType.BLACK, bg_gradient)
                # background_mean_val = mean_value_at_mask(image_path, mask_path, MaskType.WHITE, bg_gradient)
                # print(mean_val, background_mean_val)
                # if mean_val < background_mean_val:
                #     print(f"[?] The background is lighter then the bacteria - file {filename}")
                if len(indices) > 1:
                    if (inducer, exposure) in all_ave_bact:
                        # TODO carmel, like this?
                        all_ave_bact[(inducer, exposure)].extend(avebacterium)
                    else:
                        all_ave_bact[(inducer, exposure)] = avebacterium
            except Exception as e:
                print(f"Error processing TMG {filename}: {e}")
    data_summary = []
    for (inducer, exposure), values in all_ave_bact.items():
        if inducer != 35 and inducer!=33: continue
        nbins = 70#max(30, len(values) // 7)
        print(nbins)
        log_values = np.log(values)
        mean_val = np.mean(log_values)
        std_val = np.std(log_values)
        skew_val = skew(log_values)
        kurt_val = kurtosis(log_values)


        n = len(log_values)

        # Standard error of skewness
        SE_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        print(f"Skewness = {skew_val:.3f} ± {SE_skew:.3f}")

        data_summary.append({
            "inducer_concentration": inducer,
            "Exposure (ms)": exposure,
            "Mean": mean_val,
            "Std": std_val,
            "Skew": skew_val,
            "skew_error": SE_skew,
            "Kurtosis": kurt_val
        })

        # Fit skew-normal
        log_values = log_values / np.linalg.norm(log_values)
        a, loc, scale = skewnorm.fit(log_values)
        x = np.linspace(min(log_values), max(log_values), 200)
        pdf = skewnorm.pdf(x, a, loc, scale)
        bin_width = (max(log_values) - min(log_values)) / nbins
        scaled_pdf = pdf * len(log_values) * bin_width

        # Create interactive Plotly figure
        hist = go.Histogram(
            x=log_values,
            nbinsx=nbins,
            name=fr'$Log \ Histogram$',
            opacity=0.7,
            marker=dict(color='lightblue'),
        )

        fit_line = go.Scatter(
            x=x,
            y=scaled_pdf,
            mode='lines',
            name=fr'$Fit \ ( \mu={loc:.2f},\ \sigma = {scale:.1f} \ skewness={skew_val:.1f})$',
            line=dict(color='red', dash='dash')
        )

        layout = go.Layout(
            title=fr'$Log-Distribution \ for \ {Ind} \ {inducer} [\mu M], \ Exposure \ {exposure}$',
            xaxis=dict(title=fr'$log(Fluorescence \ Intensity)$',title_font=dict(size=34),tickfont = dict(size=17)),
            yaxis=dict(title=fr'$Number \ of \ Bacteria$',title_font=dict(size=34),tickfont = dict(size=17)),
            template='plotly_white',
            legend=dict(x=0.99, y=0.99,font=dict(size=16), xanchor='right', yanchor='top')
        )

        fig = go.Figure(data=[hist, fit_line], layout=layout)
        fig.show()
        # fig.write_html(f"./figures/0629/hist_{Ind}_{point_to_underscore(inducer)}_{exposure}_BGdevide.html")
        # plt.show()
    df = pd.DataFrame(data_summary)

    # Save to CSV
    # df.to_csv(fr"C:\Users\carme\Documents\BioPhysics_python\figures\0629\image_statistics_summary_threshold{Threshold}_{Ind}_10bins_no_trash_try.csv", index=False)
    return (all_ave_bact)


if __name__ == "__main__":
    # Initialize ImageJ once (reuse in both functions)
    ij = imagej.init('sc.fiji:fiji', headless=False)
    IJ = jimport('ij.IJ')
    Prefs = jimport('ij.Prefs')
    WM = jimport('ij.WindowManager')

    # TODO shachar fix the folders so it can loop over all the dates folders
    # bg_mean = background_picture_gradient(BACKGROUND)
    # plt.imshow(bg_mean, cmap="gray")
    # plt.title("Mean Background Image")
    # plt.colorbar()
    # plt.show()

    # extract_and_rename_images(f"{BASE_FOLDER}", f"{BASE_FOLDER}/{PICTURE_FOLDER}", exclude_subfolders)
    # analyze_bacteria(r"G:\My Drive\bio_physics\pictures\52_5_A_2_Phase_100.tif")[!] analyzing 20250520_70_A_2_GFP_3000.tif...
    # analyze_all_pictures(f"{BASE_FOLDER}/{PICTURE_FOLDER}")

    # process_gfp_images(os.path.join(BASE_FOLDER, PICTURE_FOLDER, MASKS_FOLDER),2.5)

    # res_iptg = process_gfp_images(os.path.join(BASE_FOLDER_IPTG, PICTURE_FOLDER, MASKS_FOLDER),2.5)
    # res_tmg = process_gfp_images(os.path.join(BASE_FOLDER_TMG, PICTURE_FOLDER, MASKS_FOLDER),7.5)
    #
    # # Or with custom labels:
    # labels = ["IPTG", "TMG"]
    # del res_iptg[800]
    # del res_tmg[800]
    # res_dict_list = [res_iptg, res_tmg]
    # plot_multiple_res_dicts_grouped(res_dict_list, labels=labels, save_path="./figures/grouped_comparison.html")
    # #
    all_ave_bacterium=process_gfp_TMG_images(os.path.join(BASE_FOLDER, PICTURE_FOLDER, MASKS_FOLDER))
    # indices, labeled, aveMaskBacterium = bct.detect_each_bacteria(f"{BASE_FOLDER}/{PICTURE_FOLDER}/{MASKS_FOLDER}/mask_20250608_31_3_A_1_TMG_1_Phase_100.tif")
    # print(f"Found {len(indices)} bacteria.")
    # avebacterium=bct.compute_bacteria_intensities(f"{BASE_FOLDER}/{PICTURE_FOLDER}/{MASKS_FOLDER}/mask_20250608_31_3_A_2_TMG_1_GFP_5000.tif",indices)
    # plt.figure()
    # plt.scatter(range(len(avebacterium)),avebacterium)
    # plt.show()
    # plt.hist(avebacterium, bins=50)
    # plt.show()

    # print("First bacterium indices:", indices[0])
    # mean_val = mean_value_at_black_mask(r"G:\My Drive\bio_physics\pictures\210_B_3_GFP_3000.tif", r"G:\My Drive\bio_physics\pictures\masks\mask_52_5_A_2_Phase_100.tif")
    # mean_val = mean_value_at_black_mask(r"G:\My Drive\bio_physics\pictures\masks\mask_140_B_3_GFP_5000.tif", r"G:\My Drive\bio_physics\pictures\masks\mask_52_5_A_2_Phase_100.tif")

    # show_image(f"data/basic_experiment/pictures/TMG/masks/mask_20250610_9_3_B_5000_GFP_TMG_Default_5000.tif")
    # show_image(f"data/basic_experiment/pictures/20250608/9_3_B_TMD_1/Default/img_channel000_position000_time000000000_z000.tif")
    # show_image(f"data/basic_experiment/pictures/TMG/masks/mask_20250610_20_9_A_3000_GFP_TMG_Default_3000.tif")
    # show_image(f"data/basic_experiment/pictures/TMG/masks/mask_20250610_20_9_A_100_Phase_TMG_1_Default_100.tif")
    # process_gfp_images(os.path.join(BASE_FOLDER, PICTURE_FOLDER, TMG_FOLDER, MASKS_FOLDER))
    # show_image(r"G:\My Drive\bio_physics\pictures\masks\\mask_35_A_3_GFP_5000.tif")

    # # --- IMPORTANT ---
    # # Change this path to the root folder containing your experiment subdirectories.
    # # For example: r"G:\My Drive\bio_physics\CarmelShachar\20250608"
    # # Using "." will process the current working directory and its subfolders.
    # root_directory_to_process = rf"{BASE_FOLDER}\20250608"
    #
    # if not os.path.isdir(root_directory_to_process):
    #     print(f"Error: The specified directory does not exist: {root_directory_to_process}")
    # else:
    #     process_directory_for_ome(root_directory_to_process)
    #     print("All processing complete.")