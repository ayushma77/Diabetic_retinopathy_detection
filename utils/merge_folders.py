# merge_folders.py
import os
import shutil


def merge_subfolders(parent_folder, subfolder, destination_folder):
    source_folder = os.path.join(parent_folder, subfolder)
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, destination_path)

if __name__ == "__main__":
    folders_to_merge = [
        "No_DR",
        "Mild",
        "Moderate",
        "Proliferate_DR",
        "Severe"
    ]

    for folder in folders_to_merge:
        parent_folder = "data/gaussian_filtered_images/gaussian_filtered_images"
        destination_folder = f"data/{folder}"

        merge_subfolders(parent_folder, folder, destination_folder)
