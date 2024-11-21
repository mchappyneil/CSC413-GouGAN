import cv2
import os

WIDTH = HEIGHT = 200
count = 0

def resize_image(input_path, output_path):
    global count
    
    image = cv2.imread(input_path)
    
    
    if image is None:
        print(f"Error reading image file at {input_path}.")
        return
    resized_image = cv2.resize(image, (WIDTH, HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, resized_image)
    count += 1

input_dir = "./impressionist-original"
output_dir = "./impressionist-resized"

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_resized{os.path.splitext(filename)[1]}")
    
    resize_image(input_path, output_path)

print(f"Resizing complete, {count} images resized.")