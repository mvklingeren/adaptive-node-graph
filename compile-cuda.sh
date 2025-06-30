#!/bin/bash

# Default values
BLOCK_SIZE=""
FILE_NAME="test-neural-graph"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --bs=*|--block-size=*)
      BLOCK_SIZE="${1#*=}"
      shift
      ;;
    --file=*)
      FILE_NAME="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--bs=SIZE|--block-size=SIZE] [--file=FILENAME]"
      echo "  SIZE must be > 0, <= 1024, and multiple of 32"
      echo "  FILENAME is the name of the typescript file in src/cuda-work/ (without .ts extension)"
      exit 1
      ;;
  esac
done

# Validate block size if provided
if [[ -n "$BLOCK_SIZE" ]]; then
  if ! [[ "$BLOCK_SIZE" =~ ^[0-9]+$ ]] || [ "$BLOCK_SIZE" -le 0 ] || [ "$BLOCK_SIZE" -gt 1024 ]; then
    echo "Error: Block size must be a number between 1 and 1024"
    exit 1
  fi
  
  if [ $((BLOCK_SIZE % 32)) -ne 0 ]; then
    echo "Error: Block size must be a multiple of 32 (warp size)"
    exit 1
  fi
  
  echo "Using block size: $BLOCK_SIZE"
fi

echo "Using file: $FILE_NAME"

# Clean and build
npm run clean

# Build the test with esbuild
npx esbuild src/cuda-work/$FILE_NAME.ts --bundle --platform=node --target=node18 --format=esm --outfile=dist/$FILE_NAME.js --external:pino

# Run the test with block size argument if provided
if [[ -n "$BLOCK_SIZE" ]]; then
  node dist/$FILE_NAME.js --bs=$BLOCK_SIZE
else
  node dist/$FILE_NAME.js
fi

# Compile the generated CUDA code
./compile-and-extract.sh
