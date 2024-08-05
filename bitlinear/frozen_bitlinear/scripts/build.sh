#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define the parent directory paths
Kernel_DIR="${SCRIPT_DIR}/../cuda/kernels"
PACK_WEIGHTS_DIR="${SCRIPT_DIR}/../cuda/pack_weights"

# Iterate through each subdirectory under $PARENT_DIR/cuda
for dir in "${Kernel_DIR}/"*/; do
    if [ -f "${dir}setup.py" ]; then
        echo "Building in directory: $dir"
        (cd "$dir" && python setup.py build_ext --inplace)
    else
        # If `setup.py` is not in the first level, check subdirectories
        for subdir in "${dir}"*/; do
            if [ -f "${subdir}setup.py" ]; then
                echo "Building in subdirectory: $subdir"
                (cd "$subdir" && python setup.py build_ext --inplace)
            fi
        done
    fi
done

# Build in the pack_weights directory
if [ -f "${PACK_WEIGHTS_DIR}/setup.py" ]; then
    echo "Building in pack_weights directory: $PACK_WEIGHTS_DIR"
    (cd "$PACK_WEIGHTS_DIR" && python setup.py build_ext --inplace)
else
    echo "No setup.py found in pack_weights directory: ${PACK_WEIGHTS_DIR}"
fi

echo "Build complete."