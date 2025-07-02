# Dynamic Thread Block Configuration

## Overview

The CUDA kernel compilation system now supports dynamic thread block configuration through command-line parameters. This allows you to optimize GPU utilization based on your specific hardware capabilities and problem sizes.

## Usage

### Default Compilation (256 threads)
```bash
npm run compile:cuda
```

### Custom Block Size
```bash
npm run compile:cuda -- --bs=512
npm run compile:cuda -- --block-size=1024
```

### Direct Script Usage
```bash
./compile-cuda.sh --bs=512
./compile-cuda.sh --block-size=1024
```

## Block Size Requirements

- **Range**: 1 to 1024 threads
- **Alignment**: Must be a multiple of 32 (warp size)
- **Valid examples**: 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024

## GPU Architecture Recommendations

### Modern GPUs (RTX 30/40 series, A100, H100)
- **Recommended**: 512-1024 threads for compute-intensive kernels
- **Memory-bound operations**: 256-512 threads
- **Small problems**: 128-256 threads

### Older GPUs (GTX 10 series, RTX 20 series)
- **Recommended**: 256-512 threads
- **Conservative**: 256 threads (current default)

### Very old GPUs (GTX 9 series and older)
- **Recommended**: 128-256 threads
- **Safe default**: 256 threads

## Kernel-Specific Behavior

The system intelligently applies block sizes based on kernel type:

### Element-wise Operations (ReLU, Add, Scale)
- Uses the specified block size directly
- Grid size calculated as `ceil(total_elements / block_size)`

### Dense Layers
- Uses the specified block size for feature dimension
- Grid configured for optimal memory access patterns

### Attention Operations
- Some operations override block size for optimal performance
- Embedding and attention heads use problem-size specific configurations

### Reduction Operations (Softmax, LayerNorm)
- May use smaller block sizes based on reduction dimension
- Shared memory usage considered for optimal configuration

## Performance Tips

1. **Start with 512**: Good balance for most modern GPUs
2. **Monitor occupancy**: Use `nvprof` or Nsight to check GPU utilization
3. **Problem size matters**: Larger problems benefit from larger block sizes
4. **Memory bandwidth**: Memory-bound kernels may not benefit from larger blocks
5. **Experiment**: Different problems may have different optimal configurations

## Examples

```bash
# High-performance modern GPU
npm run compile:cuda -- --bs=1024

# Balanced configuration
npm run compile:cuda -- --bs=512

# Conservative for older hardware
npm run compile:cuda -- --bs=256

# Memory-constrained scenarios
npm run compile:cuda -- --bs=128
```

## Implementation Details

- Block size is validated at compile time
- Invalid sizes show helpful error messages
- The compiler applies the block size to appropriate kernel types
- Some specialized kernels may override the setting for optimal performance
- All generated kernels respect the 32-thread warp alignment requirement

## Troubleshooting

### "Block size must be a multiple of 32"
- Use values like 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, etc.
- Avoid values like 100, 200, 300 which are not warp-aligned

### "Block size must be between 1 and 1024"
- Modern GPUs support up to 1024 threads per block
- Use smaller values for older hardware

### Performance Issues
- Try different block sizes and measure performance
- Consider your specific GPU architecture and problem size
- Monitor GPU occupancy and memory bandwidth utilization
