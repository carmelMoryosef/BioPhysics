import os
import json
import tifffile
from ome_types import from_tiff
from ome_types.model import OME, Image, Plane


# --- Part 1: Helper functions for ome_to_metadata ---

def _populate_summary_metadata(ome_obj: OME) -> dict:
    """Creates the main 'Summary' block for the metadata file."""
    if not ome_obj.images:
        return {}

    first_image = ome_obj.images[0]

    dim_order_str = first_image.pixels.dimension_order.value
    axis_map = {'C': 'channel', 'T': 'time', 'Z': 'z'}
    filtered_axis_order = [axis_map.get(char.upper()) for char in dim_order_str if char.upper() in axis_map]

    if len(ome_obj.images) > 1:
        filtered_axis_order.append('position')

    summary = {
        "MicroManagerVersion": "2.0.0",
        "MetadataVersion": "12.0.0",
        "ChannelGroup": "",
        "ChNames": [ch.name or "Default" for ch in first_image.pixels.channels],
        "AxisOrder": filtered_axis_order,
        "TimeFirst": first_image.pixels.dimension_order.value.upper().startswith('T'),
        "SlicesFirst": first_image.pixels.dimension_order.value.upper().startswith('Z'),
        "StartTime": first_image.acquisition_date.isoformat() if first_image.acquisition_date else "Unknown",
        "Width": first_image.pixels.size_x,
        "Height": first_image.pixels.size_y,
        "UserData": {},
        "IJType": 1,
        "PixelType": first_image.pixels.type.value.upper(),
    }
    return summary


def _generate_plane_filename(base_name: str, plane: Plane, position_index: int) -> str:
    """Generates the standardized filename for a single image plane."""
    return (
        f"{base_name}_channel{plane.the_c:03d}_"
        f"position{position_index:03d}_"
        f"time{plane.the_t:09d}_"
        f"z{plane.the_z:03d}.tif"
    )


def _create_plane_metadata(
        ome_obj: OME, image: Image, plane: Plane, position_index: int, dir_name: str, plane_filename: str
) -> tuple[dict, dict]:
    """Creates the 'Coords' and 'Metadata' blocks for a single plane."""
    pixels = image.pixels

    coords_data = {
        "Frame": plane.the_t, "FrameIndex": plane.the_t,
        "PositionIndex": position_index, "Slice": plane.the_z,
        "SliceIndex": plane.the_z, "ChannelIndex": plane.the_c,
    }

    plane_metadata = coords_data.copy()
    plane_metadata.update({
        "Width": pixels.size_x,
        "Height": pixels.size_y,
        "PixelType": pixels.type.value.upper(),
        "UUID": str(ome_obj.uuid) if ome_obj.uuid else None,
        "ROI": f"0-0-{pixels.size_x}-{pixels.size_y}",
        "ReceivedTime": image.acquisition_date.strftime('%Y-%m-%d %H:%M:%S.%f')[
                        :-3] + " +0000" if image.acquisition_date else None,
        "PixelSizeUm": pixels.physical_size_x if pixels.physical_size_x is not None else 0.0,
        "PixelSizeAffine": f"{pixels.physical_size_x or 0.0};0.0;0.0;{pixels.physical_size_y or 0.0};0.0;0.0",
        "PositionName": image.name or f"Position-{position_index}",
        "XPositionUm": plane.position_x if plane.position_x is not None else 0.0,
        "YPositionUm": plane.position_y if plane.position_y is not None else 0.0,
        "ZPositionUm": plane.position_z if plane.position_z is not None else 0.0,
        "BitDepth": 16 if "16" in pixels.type.value else 8,
        "FileName": f"{dir_name}/{plane_filename}",
    })

    if plane.exposure_time is not None:
        plane_metadata["Camera-1-Exposure"] = f"{plane.exposure_time * 1000:.2f}"
        plane_metadata["Camera-1-Timing-ExposureTimeNs"] = f"{plane.exposure_time * 1e9:.4f}"

    plane_metadata["ScopeDataKeys"] = list(plane_metadata.keys())
    plane_metadata["UserData"] = {key: {"type": "STRING", "scalar": str(value)} for key, value in
                                  plane_metadata.items()}

    return coords_data, plane_metadata


# --- Part 2: The main extraction function for a single OME-TIFF ---

def ome_to_metadata(ome_path: str, output_dir: str):
    """
    Extracts metadata and individual plane images from a multi-image OME-TIFF file.
    """
    dir_name = "Default"
    default_dir_path = os.path.join(output_dir, dir_name)
    os.makedirs(default_dir_path, exist_ok=True)
    metadata_path = os.path.join(default_dir_path, "metadata.txt")

    try:
        ome_obj = from_tiff(ome_path)
        all_pixel_data = tifffile.imread(ome_path)
    except Exception as e:
        print(f"Error reading OME-TIFF file '{ome_path}': {e}")
        return

    metadata_root = {"Summary": _populate_summary_metadata(ome_obj)}
    if not metadata_root["Summary"]:
        with open(metadata_path, 'w') as f:
            json.dump({}, f)
        return

    base_name = os.path.splitext(os.path.splitext(os.path.basename(ome_path))[0])[0]
    plane_index_counter = 0

    for image_index, image in enumerate(ome_obj.images):
        if not image.pixels.planes: continue

        for plane in image.pixels.planes:
            plane_filename = _generate_plane_filename(base_name, plane, image_index)

            if plane_index_counter < len(all_pixel_data):
                output_tiff_path = os.path.join(default_dir_path, plane_filename)
                tifffile.imwrite(output_tiff_path, all_pixel_data[plane_index_counter])
            else:
                print(
                    f"Warning: More planes in XML than in data for {ome_path}. Skipping plane index {plane_index_counter}.")

            coords_data, plane_metadata = _create_plane_metadata(
                ome_obj, image, plane, image_index, dir_name, plane_filename
            )
            metadata_root[f"Coords-{dir_name}/{plane_filename}"] = coords_data
            metadata_root[f"Metadata-{dir_name}/{plane_filename}"] = plane_metadata

            plane_index_counter += 1

    with open(metadata_path, 'w') as f:
        json.dump(metadata_root, f, indent=2)


# --- Part 3: The new function to process a directory ---

def process_directory_for_ome(root_folder: str):
    """
    Recursively finds all '*.ome.tif' files within a root folder and processes them.

    For each '.ome.tif' file found, it calls the `ome_to_metadata` function,
    setting the output directory to be the same folder where the '.ome.tif'
    file is located.

    :param root_folder: The top-level directory to start searching from.
    """
    print(f"Starting search in: {os.path.abspath(root_folder)}\n")

    # os.walk is a generator that traverses a directory tree
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Check if the file matches the target extension
            if filename.endswith(".ome.tif"):
                # Construct the full path to the found file
                ome_file_path = os.path.join(dirpath, filename)

                # The output directory is the folder containing the file
                output_directory = dirpath

                print(f"--- Processing: {ome_file_path} ---")
                print(f"Output will be saved in: {output_directory}")

                # Call the main extraction function
                ome_to_metadata(ome_file_path, output_directory)

                print("--- Done. ---\n")


# --- Part 4: Example of how to run the new function ---

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this path to the root folder containing your experiment subdirectories.
    # For example: r"G:\My Drive\bio_physics\CarmelShachar\20250608"
    # Using "." will process the current working directory and its subfolders.
    root_directory_to_process = r"G:\My Drive\bio_physics\CarmelShachar\20250608"

    if not os.path.isdir(root_directory_to_process):
        print(f"Error: The specified directory does not exist: {root_directory_to_process}")
    else:
        process_directory_for_ome(root_directory_to_process)
        print("All processing complete.")