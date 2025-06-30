#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables
OUTPUT_DIR="dist"
TARGET_FILE="kernel.o"
FILE_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --file=*)
      FILE_NAME="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

if [[ -z "$FILE_NAME" ]]; then
  echo "Error: --file=FILENAME is required."
  exit 1
fi

# 1. Ensure the output directory exists
echo "Ensuring output directory '$OUTPUT_DIR' exists..."
mkdir -p $OUTPUT_DIR

# 2. Build the Docker image and export the compiled file in one step.
# The --output flag tells Docker to copy the specified file from the final image
# to the local filesystem.
# We target the final stage of the Dockerfile which only contains the kernel.o
echo "Building Docker image and extracting compiled kernel for $FILE_NAME..."
docker build --build-arg KERNEL_FILENAME=$FILE_NAME --output type=local,dest=$OUTPUT_DIR .

echo "Success! Compiled kernel is at $OUTPUT_DIR/$TARGET_FILE"
