// Strategies for attention-related kernels
import { BaseKernelStrategy } from '../registry.js';
import { KernelLaunchConfig, CudaHardwareConstraints } from '../types.js';
import { GridBlockCalculator } from '../utils.js';

/**
 * Strategy for split heads forward kernel
 */
export class SplitHeadsStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length < 4) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, numHeads, seqLen, headDim] = shape;
    const blockSize = Math.min(headDim, constraints.maxThreadsPerBlock);
    const alignedBlockSize = GridBlockCalculator.alignToWarpSize(blockSize, constraints.warpSize);
    const finalBlockSize = Math.min(alignedBlockSize, constraints.maxThreadsPerBlock);
    
    return {
      gridDim: `dim3(${numHeads}, ${seqLen}, ${batchSize})`,
      blockDim: `dim3(${finalBlockSize}, 1, 1)`,
      sharedMemSize: 0,
    };
  }
}

/**
 * Strategy for concatenate heads forward kernel
 */
export class ConcatHeadsStrategy extends BaseKernelStrategy {
  private readonly DEFAULT_NUM_HEADS = 6; // Default for the specific model
  
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length < 3) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, seqLen, embedDim] = shape;
    
    return {
      gridDim: `dim3(${seqLen}, ${batchSize})`,
      blockDim: `dim3(${embedDim}, 1, 1)`,
      sharedMemSize: 0,
    };
  }
}

/**
 * Strategy for tiled matrix multiplication kernels
 */
export class MatMulTiledStrategy extends BaseKernelStrategy {
  private readonly TILE_SIZE = 32;
  
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length < 4) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, numHeads, outputRows, outputCols] = shape;
    const tilesM = Math.ceil(outputRows / this.TILE_SIZE);
    const tilesN = Math.ceil(outputCols / this.TILE_SIZE);
    const totalTiles = tilesM * tilesN;
    
    return {
      gridDim: `dim3(${totalTiles}, ${numHeads}, ${batchSize})`,
      blockDim: `dim3(${this.TILE_SIZE}, ${this.TILE_SIZE}, 1)`,
      sharedMemSize: 0, // Shared memory is allocated in the kernel itself
    };
  }
}
