import re
import os
import json
import shutil

from pathlib import Path

def extract_and_rename_images(src_root, dst_root, exclude_subfolders=None):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    exclude_subfolders = set(exclude_subfolders or [])

    # Iterate over all metadata files recursively
    for metadata_path in src_root.rglob("metadata.txt"):
        subfolder = metadata_path.parent
        if any(part in exclude_subfolders for part in subfolder.relative_to(src_root).parts):
            continue

        folder_name = subfolder.parent.name
        date_folder_name = subfolder.parent.parent.name

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Get channel names and clean them
        channel_list = metadata.get("Summary", {}).get("ChNames", [])
        channel_list = [re.sub(r'\d+', '', item) for item in channel_list]

        # Find all .tif files under the same folder
        for img_file in subfolder.rglob("*.tif"):
            try:
                file_data = metadata.get(f"Metadata-Default/{img_file.name}", {})
                exposure = int(file_data.get("Exposure-ms") or float(file_data["Camera-1-Exposure"]))
                # exposure=exposure/1000
                channel_index = file_data["ChannelIndex"]
                new_name = f"{date_folder_name}_{folder_name}_{channel_list[channel_index]}_{exposure}.tif"
                dst_path = dst_root / new_name
                shutil.copy(img_file, dst_path)
                print(f"[+] Saved {dst_path}")
            except Exception as e:
                print(f"[!] Failed to process {img_file}: {e}")
