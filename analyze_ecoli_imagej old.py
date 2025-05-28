import imagej
import numpy as np
from imagej._java import jimport
from pathlib import Path

# Initialize ImageJ once (reuse in both functions)
ij = imagej.init('sc.fiji:fiji',mode='interactive')
IJ = jimport('ij.IJ')
Prefs = jimport('ij.Prefs')


def analyze_bacteria(image_path, min_size=5, max_size=1e9):
    """
    Loads an image, thresholds, runs particle analysis, and returns bacteria count.
    Parameters:
        image_path: str - Path to the image file.
        min_size: float - Minimum particle size to include.
        max_size: float - Maximum particle size to include.
    Returns:
        count: int - Number of detected bacteria.
    """
    # image = ij.io().open(image_path)
    
    # # Convert to 8-bit grayscale (needed for Analyze Particles)
    # image_gray = ij.op().run("convert.uint8", image)

    # # Auto threshold (Otsu)
    # thresholded = ij.op().run("threshold.otsu", image_gray)
    
    # # Set threshold on image (required before Analyze Particles)
    # ij.py.run_macro(f"""
    #     setThreshold({thresholded.getMin()}, {thresholded.getMax()});
    #     run("Convert to Mask");
    #     run("Analyze Particles...", "size={min_size}-{max_size} display clear summarize");
    # """)

    # # Get number of rows in Results table = count of detected particles
    # results_table = ij.WindowManager.getWindow("Results").getTextPanel().getLines()
    # count = len(results_table) - 1  # First line is header

    # # Close Results window to keep things clean
    # ij.WindowManager.getWindow("Results").close()
    name = image_path.split("\\")[-1].split(".")[0]
    mask_path = fr"G:\My Drive\bio_physics\pictures\masks\mask_{name}.tif"
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
    IJ.run(imp, "Analyze Particles...", "size=30-110 show=Masks display exclude clear summarize overlay add composite")

    IJ.saveAs(imp, "Tiff", mask_path)
    IJ.saveAs("Results", fr"G:\My Drive\bio_physics\results_basic_experiment\res_{name}.csv")
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
    # thresholded = ij.op().run("threshold.otsu", image_gray)
    # image_gray = ij.op().run("convert.uint8", image)
    ij.ui().show("Gray Image", image_gray)

    # Apply threshold and convert to mask
    # setThreshold({thresholded.getMin()}, {thresholded.getMax()});
    # run("Convert to Mask");
    # run("RGB Color");
    # run("Color Balance...");



    # Apply histogram equalization or normalization
    equalized = ij.op().run("normalize", image)

    # Show result
    ij.ui().show("Auto Balanced (Simulated)", equalized)


    # setAutoThreshold("Otsu");
    # run("Analyze Particles...", "size={min_size}-{max_size} show=Outlines display clear summarize");
    # ij.py.run_macro(f"""
    #     open("{image_path}");
    #     run("RGB Color");
    #     run("Color Balance...");
    #     call("ij.plugin.frame.ColorBalance.applyAuto");
    # """)

    # The above macro will open the image and show outlines in Fiji UI
    
    # Keep script running so the window stays open
    print("Threshold and analysis done. Leave this script running to inspect the window.")
    input("Press Enter to close...")
    

def show_image(image_path):
    image = ij.io().open(image_path)
    ij.ui().show("Image", image)
    input("Press Enter to close...")
    
def analyze_all_pictures(image_folder):
    src_root = Path(image_folder)

    for image_path in src_root.glob("*.tif"):
        image_path_name = image_path.name
        print(f"[!] analyzing {image_path_name}...")
        # print(str(image_path.relative_to(".")))
        # rel_path = str(image_path.relative_to("."))
        mask = analyze_bacteria(str(image_path.absolute()))
        # show_image(mask)
    print(f"[!] Done!")


# images = [
#     "./data/basic_experiment/pictures/210_B_1_Phase_100.tif",
#     "./data/basic_experiment/pictures/210_B_2_Phase_100.tif",
#     "./data/basic_experiment/pictures/210_B_3_Phase_100.tif"
# ]




if __name__ == "__main__":
    image_folder = r"G:\My Drive\bio_physics\pictures"
    analyze_all_pictures(image_folder)
    
    # show_image(f"data/basic_experiment/pictures/masks/mask_210_B_1_Phase_100.tif")


