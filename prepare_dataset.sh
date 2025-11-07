#!/bin/bash

# Prepare Korean math datasets for VLMEvalKit
# This script downloads necessary TSV files from HuggingFace to playground directory

echo "================================================"
echo "Preparing Korean Math Datasets for VLMEvalKit"
echo "================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to prepare_kor_dataset directory
cd "$SCRIPT_DIR/prepare_kor_dataset"

echo "Running dataset preparation script..."
python prepare_dataset.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Dataset preparation completed successfully!"
    echo "Files saved to: $SCRIPT_DIR/playground"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "Error: Dataset preparation failed!"
    echo "================================================"
    exit 1
fi
