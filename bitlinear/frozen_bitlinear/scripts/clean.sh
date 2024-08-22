#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-b] [-w] [-r]"
    echo "  -b    Deletes kernel builds"
    echo "  -w    Delete 'weights' directories"
    echo "  -r    Delete 'results' directories"
    exit 1
}

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR="${SCRIPT_DIR}/.."

# Parse command-line arguments
while getopts ":bwr" opt; do
    case ${opt} in
        b )
            DELETE_BUILD=true
            ;;
        w )
            DELETE_WEIGHTS=true
            ;;
        r )
            DELETE_RESULTS=true
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument" >&2
            usage
            ;;
    esac
done

# Shift parsed options away from the positional parameters
shift $((OPTIND -1))

if [ "$#" -eq 0 ]; then
    DELETE_BUILD=true
fi

# Delete .so files if -s is specified
if [ "$DELETE_BUILD" = true ]; then
    echo "Deleting .so files..."
    find "$PARENT_DIR" -type f -name '*.so' -exec rm -f {} \;
fi

# Delete 'build' directories if -b is specified
if [ "$DELETE_BUILD" = true ]; then
    echo "Deleting 'build' directories..."
    find "$PARENT_DIR" -type d -name 'build' -exec rm -rf {} \;
fi

# Delete 'weights' directories if -w is specified
if [ "$DELETE_WEIGHTS" = true ]; then
    echo "Deleting 'weights' directories..."
    find "$PARENT_DIR" -type d -name 'weights' -exec rm -rf {} \;
fi

# Delete 'results' directories if -r is specified
if [ "$DELETE_RESULTS" = true ]; then
    echo "Deleting 'results' directories..."
    find "$PARENT_DIR" -type d -name 'results' -exec rm -rf {} \;
fi

echo "Cleanup complete."
