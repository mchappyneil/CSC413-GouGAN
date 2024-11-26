import os
import shutil
from sklearn.model_selection import train_test_split


def split_datset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    
    images = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    
    train_val_ims, test_ims = train_test_split(images, test_size=(1-train_ratio), random_state=42)
    train_ims, val_ims = train_test_split(train_val_ims, test_size=val_ratio, random_state=42)
    
    print(f"Total images: {len(images)}")
    print(f"Training images: {len(train_ims)},\nValidation images: {len(val_ims)},\nTest images: {len(test_ims)}")
    
    train_dir = os.path.join(output_dir, "training")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    def move_images(image_list, src_dir, dest_dir):
        for image in image_list:
            shutil.move(os.path.join(src_dir, image), os.path.join(dest_dir, image))
            
    move_images(train_ims, input_dir, train_dir)
    move_images(val_ims, input_dir, val_dir)
    move_images(test_ims, input_dir, test_dir)
    
    print(f"Data split completed. Files saved in {output_dir}")
    
input_directory = './impressionist-resized'
output_directory = './impressionist-split-dataset'

split_datset(input_directory, output_directory, train_ratio=0.8, val_ratio=0.2)