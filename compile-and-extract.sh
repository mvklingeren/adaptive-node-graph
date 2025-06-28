#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables
OUTPUT_DIR="dist"
TARGET_FILE="kernel.o"

# 1. Ensure the output directory exists
echo "Ensuring output directory '$OUTPUT_DIR' exists..."
mkdir -p $OUTPUT_DIR

# 2. Build the Docker image and export the compiled file in one step.
# The --output flag tells Docker to copy the specified file from the final image
# to the local filesystem.
# We target the final stage of the Dockerfile which only contains the kernel.o
echo "Building Docker image and extracting compiled kernel..."
docker build --output type=local,dest=$OUTPUT_DIR .

echo "Success! Compiled kernel is at $OUTPUT_DIR/$TARGET_FILE"
