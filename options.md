# Dynamic Thread Block Configuration Options

## Problem Statement
Current CUDA kernel generation uses fixed 256-thread blocks regardless of GPU capabilities or problem size. Modern GPUs support up to 1024 threads per block, and optimal configuration varies by kernel type and data size.

## Lightweight Solution Options

### Option 1: Compile-Time GPU Capability Constants
**Approach**: Generate GPU-specific constants at compile time and embed them in the kernel code.

**Implementation**:
```typescript
// In CudaGraphCompiler
private getGpuCapabilities(): GpuCapabilities {
  return {
    maxThreadsPerBlock: 1024,
    warpSize: 32,
    maxBlocksPerSM: 16,
    sharedMemoryPerBlock: 49152
  };
}
```

**Generated CUDA Code**:
```cuda
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define MAX_SHARED_MEM 49152

// Use in kernel configuration
const int optimal_block_size = min(problem_size, MAX_THREADS_PER_BLOCK);
```

**Pros**: Simple, no runtime overhead, works with current architecture
**Cons**: Not adaptive to different GPUs, requires recompilation for different hardware

### Option 2: Runtime GPU Query with Simple Heuristics
**Approach**: Query GPU capabilities once at runtime and use simple rules for block size selection.

**Implementation**:
```typescript
interface GpuCapabilities {
  maxThreadsPerBlock: number;
  warpSize: number;
  multiProcessorCount: number;
  sharedMemoryPerBlock: number;
}

// Extend CudaRuntime interface
async getGpuCapabilities(): Promise<GpuCapabilities>;
```

**Block Size Selection Rules**:
- Element-wise ops: `min(nextPowerOf2(totalElements/gridSize), maxThreadsPerBlock)`
- Reductions: `min(problemSize, maxThreadsPerBlock)` rounded to warp size
- Matrix ops: Based on tile size and shared memory constraints

**Pros**: Adaptive to hardware, simple implementation, minimal overhead
**Cons**: Requires extending runtime interface

### Option 3: Problem-Size Based Heuristics (No GPU Query)
**Approach**: Use mathematical heuristics based on tensor shapes and operation types.

**Implementation**:
```typescript
private calculateOptimalBlockSize(
  kernelType: string,
  primaryTensorShape: number[],
  maxThreads: number = 1024
): number {
  const totalElements = primaryTensorShape.reduce((a, b) => a * b, 1);
  
  switch (kernelType) {
    case 'elementwise':
      return Math.min(512, nextPowerOf2(Math.sqrt(totalElements)));
    case 'reduction':
      return Math.min(1024, nextPowerOf2(primaryTensorShape[primaryTensorShape.length - 1]));
    case 'matmul':
      return Math.min(256, nextPowerOf2(Math.sqrt(primaryTensorShape[primaryTensorShape.length - 1])));
    default:
      return Math.min(512, maxThreads);
  }
}
```

**Pros**: No runtime dependencies, works immediately, mathematically sound
**Cons**: Not hardware-adaptive, may not be optimal for all cases

### Option 4: Template-Based Configuration
**Approach**: Generate multiple kernel variants with different block sizes and select at runtime.

**Implementation**:
```cuda
// Generate multiple kernel variants
template<int BLOCK_SIZE>
__global__ void dense_forward_template(/* args */) { /* implementation */ }

// Instantiate common sizes
__global__ void dense_forward_256(/* args */) { dense_forward_template<256>(/* args */); }
__global__ void dense_forward_512(/* args */) { dense_forward_template<512>(/* args */); }
__global__ void dense_forward_1024(/* args */) { dense_forward_template<1024>(/* args */); }

// Runtime selection
extern "C" void executeGraph(/* args */) {
  if (problem_size <= 256) dense_forward_256<<<grid, 256>>>(/* args */);
  else if (problem_size <= 512) dense_forward_512<<<grid, 512>>>(/* args */);
  else dense_forward_1024<<<grid, 1024>>>(/* args */);
}
```

**Pros**: Optimal performance, compile-time optimization, flexible
**Cons**: Increases code size, complex generation logic

### Option 5: Occupancy-Based Calculation
**Approach**: Calculate theoretical occupancy and choose block size for maximum GPU utilization.

**Implementation**:
```typescript
private calculateOccupancyOptimalBlockSize(
  sharedMemoryPerBlock: number,
  registersPerThread: number,
  maxThreadsPerBlock: number
): number {
  // Simple occupancy calculation
  const maxBlocksFromSharedMem = Math.floor(49152 / sharedMemoryPerBlock);
  const maxBlocksFromRegisters = Math.floor(65536 / (registersPerThread * maxThreadsPerBlock));
  
  const limitingFactor = Math.min(maxBlocksFromSharedMem, maxBlocksFromRegisters);
  
  // Find block size that maximizes occupancy
  for (let blockSize = 1024; blockSize >= 32; blockSize -= 32) {
    const blocksPerSM = Math.floor(maxThreadsPerBlock / blockSize);
    if (blocksPerSM >= limitingFactor) return blockSize;
  }
  
  return 256; // fallback
}
```

**Pros**: Theoretically optimal, considers resource constraints
**Cons**: Requires register/shared memory usage estimation

### Option 6: Kernel-Specific Optimization Tables
**Approach**: Pre-computed lookup tables for optimal configurations per kernel type and problem size ranges.

**Implementation**:
```typescript
const KERNEL_CONFIGS = {
  'embedding_forward': [
    { minElements: 0, maxElements: 1024, blockSize: 256 },
    { minElements: 1024, maxElements: 8192, blockSize: 512 },
    { minElements: 8192, maxElements: Infinity, blockSize: 1024 }
  ],
  'dense_forward_2d': [
    { minElements: 0, maxElements: 512, blockSize: 128 },
    { minElements: 512, maxElements: 4096, blockSize: 256 },
    { minElements: 4096, maxElements: Infinity, blockSize: 512 }
  ],
  // ... more kernels
};
```

**Pros**: Fast lookup, empirically tuned, kernel-specific
**Cons**: Requires manual tuning, may not cover all cases

## Recommended Hybrid Approach

**Combine Options 2 + 3 + 6**: 
1. Query GPU capabilities once at startup (Option 2)
2. Use problem-size heuristics as base calculation (Option 3)  
3. Apply kernel-specific optimization tables for fine-tuning (Option 6)

**Implementation Strategy**:
```typescript
class DynamicBlockSizeCalculator {
  private gpuCaps: GpuCapabilities;
  
  async initialize(runtime: CudaRuntime) {
    this.gpuCaps = await runtime.getGpuCapabilities();
  }
  
  calculateOptimalBlockSize(
    kernelType: string,
    tensorShape: number[],
    sharedMemoryUsage: number = 0
  ): number {
    // 1. Get base size from heuristics
    const baseSize = this.getHeuristicBlockSize(kernelType, tensorShape);
    
    // 2. Apply hardware constraints
    const hwConstrainedSize = Math.min(baseSize, this.gpuCaps.maxThreadsPerBlock);
    
    // 3. Apply kernel-specific optimizations
    const optimizedSize = this.applyKernelSpecificOptimizations(
      kernelType, hwConstrainedSize, tensorShape
    );
    
    // 4. Ensure warp alignment
    return this.alignToWarpSize(optimizedSize);
  }
}
```

## Implementation Complexity Ranking

1. **Option 3** (Problem-Size Heuristics) - Simplest, immediate benefit
2. **Option 6** (Lookup Tables) - Simple data structures, good results  
3. **Option 2** (Runtime GPU Query) - Requires runtime interface extension
4. **Option 1** (Compile-Time Constants) - Simple but inflexible
5. **Option 5** (Occupancy Calculation) - Complex resource modeling
6. **Option 4** (Template Variants) - Most complex code generation

## Recommendation

Start with **Option 3** (Problem-Size Heuristics) for immediate improvement, then enhance with **Option 6** (Lookup Tables) for kernel-specific tuning. This provides significant benefit with minimal complexity and maintains the lightweight architecture.
