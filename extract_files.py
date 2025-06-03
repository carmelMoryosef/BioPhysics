import re
import os
import json
import shutil

from pathlib import Path

def extract_and_rename_images(src_root, dst_root, exclude_subfolders=None):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for subfolder in src_root.iterdir():
        if subfolder.is_dir() and subfolder.name not in exclude_subfolders:
            folder_name = subfolder.name

            # Locate JSON .txt file
            json_txt_files = list(subfolder.glob("**/*/metadata.txt"))
            if not json_txt_files:
                print(f"[!] No JSON .txt file found in {subfolder}")
                continue
            
            with open(json_txt_files[0], 'r') as f:
                metadata = json.load(f)

            # Access metadata
            channel_list = metadata.get("Summary")["ChNames"]
            channel_list = [re.sub(r'\d+', '', item) for item in channel_list]

            # Process .tif images
            for img_file in subfolder.glob("**/*.tif"):
                print(img_file)
                file_data = metadata.get(f"Metadata-Default/{img_file.name}")
                if "Exposure-ms" in file_data:
                    exposure = int(file_data["Exposure-ms"])
                else:
                    exposure = int(float(file_data["Camera-1-Exposure"]))
                channel_index = file_data["ChannelIndex"]
                new_name = f"{folder_name}_{channel_list[channel_index]}_{exposure}.tif"
                dst_path = dst_root / new_name
                shutil.copy(img_file, dst_path)
                print(f"[+] Saved {dst_path}")

# if __name__ == "__main__":
#     extract_and_rename_images(r"G:\My Drive\bio_physics\20250520", r"G:\My Drive\bio_physics\pictures")
