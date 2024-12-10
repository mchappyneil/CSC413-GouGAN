#!/bin/bash

# Directories
INPUT_DIR="./test"
OUTPUT_DIR="./results"
CHECKPOINT="./best_models/best_models.pth"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all images in the input directory
for img in "$INPUT_DIR"/*; do
    # Get the base name of the file (e.g., "image.jpg")
    base_name=$(basename "$img")

    # Set the output file path
    output_path="$OUTPUT_DIR/$base_name"

    # Run the Python script to convert the image
    python3 train_generators.py \
        --mode convert \
        --input "$img" \
        --output "$output_path" \
        --checkpoint "$CHECKPOINT"

    echo "Converted $img to $output_path"
done

echo "All images have been processed and saved to $OUTPUT_DIR."
