# Workspace Size Configuration

## Problem Fixed
The original error was caused by insufficient workspace memory allocation in the CUDA kernel's internal allocator, not GPU VRAM limitations.

## Usage

### Default (1GB workspace):
```bash
make clean && make
./run_model
```

### Custom workspace size:
```bash
# For 2GB workspace
make clean && make WORKSPACE_SIZE_MB=2048
./run_model

# For 512MB workspace (might be too small)
make clean && make WORKSPACE_SIZE_MB=512
./run_model

# For 4GB workspace (plenty of headroom)
make clean && make WORKSPACE_SIZE_MB=4096
./run_model
```

## Changes Made

1. **Makefile**: Added `WORKSPACE_SIZE_MB` parameter with 1024MB default
2. **main.cpp**: Uses compile-time `WORKSPACE_SIZE_MB` macro to set workspace size
3. **GPU Architecture**: Updated to `sm_89` for RTX 4090 (was `sm_75` for Tesla T4)

## Memory Requirements

- **Original**: ~216MB (too small)
- **Recommended**: 1024MB (1GB) - should work fine
- **Safe**: 2048MB (2GB) - plenty of headroom
- **RTX 4090 VRAM**: 24GB total available

## Verification

The program will now print:
```
Using workspace size: 1024MB (1073741824 bytes)
```

If successful, you should see actual logit values instead of zeros in the output.
