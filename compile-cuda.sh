#!/bin/bash

# Default block size
BLOCK_SIZE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --bs=*|--block-size=*)
      BLOCK_SIZE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--bs=SIZE|--block-size=SIZE]"
      echo "  SIZE must be > 0, <= 1024, and multiple of 32"
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

# Clean and build
npm run clean

# Build the test with esbuild
npx esbuild src/cuda-work/test-neural-graph.ts --bundle --platform=node --target=node18 --format=esm --outfile=dist/test-neural-graph.js --external:pino

# Run the test with block size argument if provided
if [[ -n "$BLOCK_SIZE" ]]; then
  node dist/test-neural-graph.js --bs=$BLOCK_SIZE
else
  node dist/test-neural-graph.js
fi

# Compile the generated CUDA code
./compile-and-extract.sh
