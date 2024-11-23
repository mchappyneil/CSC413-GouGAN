from pathlib import Path
import shutil
import os
# Getting the data from kaggle
# # Download latest version
#import kagglehub
# path = kagglehub.dataset_download("awsaf49/artifact-dataset")

input_dir = "./test/test"
output_dir = Path("./cleaned_REAL_data")
output_dir.mkdir(parents=True, exist_ok=True)
path_train = Path("./imagenet/train")
path_val = Path("./imagenet/val")

def copy_selected_images(source_path: Path, destination_folder: Path, interval: int = 3) -> None:
    # Create destination folder if it doesn't exist
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    for subfolder in source_path.iterdir():
        if subfolder.is_dir():
            # Get all image files in subfolder
            images = [f for f in subfolder.glob('*.[jp][pn][g]*')]
            # Sort to ensure consistent ordering
            images.sort()
            # Get every nth image
            selected_images = images[::interval]
            
            print(f"Selected {len(selected_images)} images from {subfolder.name}")
            # Copy the selected images
            for img in selected_images:
                shutil.copy(img, destination_folder / img.name)

copy_selected_images(path_train, output_dir)
copy_selected_images(path_val, output_dir)
