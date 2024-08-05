#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR="${SCRIPT_DIR}/.."

# Delete files with .so extension
echo "Deleting .so files..."
find "$PARENT_DIR" -type f -name '*.so' -exec rm -f {} \;

# Delete directories named 'weights'
echo "Deleting 'weights' directories..."
find "$PARENT_DIR" -type d -name 'weights' -exec rm -rf {} \;

# Delete directories named 'build'
echo "Deleting 'build' directories..."
find "$PARENT_DIR" -type d -name 'build' -exec rm -rf {} \;

# Delete directories named 'build'
echo "Deleting 'results' directories..."
find "$PARENT_DIR" -type d -name 'results' -exec rm -rf {} \;

echo "Cleanup complete."