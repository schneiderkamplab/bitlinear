#!/bin/bash

timestamp=$(date "+%Y%m%d_%T")
save_dir="results/$timestamp"

# Default values
device=0
kernel="TorchLinear"

# Function to display usage information
usage() {
    echo "Usage: $0 [-d <integer>] [-k <string>]"
    echo "  -d <integer>  Optional device argument (0, 1, 2, 3). Default is 0."
    echo "  -k <string>   Optional kernel argument. Default is 'TorchLinear'."
    exit 1
}

# Parse options
while getopts ":d:k:" opt; do
    case ${opt} in
        d )
            device=$OPTARG
            # Check if the integer argument is valid (0, 1, 2, 3)
            if ! [[ "$device" =~ ^[0-3]$ ]]; then
                echo "Error: Device argument must be one of 0, 1, 2, or 3"
                usage
            fi
            ;;
        k )
            kernel=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument" >&2
            usage
            ;;
    esac
done

shift $((OPTIND -1))

mkdir -p ${save_dir}
cp $0 $save_dir # saves the current training script

cd "$(dirname "$0")/.."

run_cmd="CUDA_VISIBLE_DEVICES=$device python -m tests \
    --save_dir $save_dir \
    --kernel $kernel \
    -a \
    2>&1 | tee ${save_dir}/test-${timestamp}.log"

echo "$run_cmd"
eval "$run_cmd"
