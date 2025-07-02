// Strategies for activation and normalization kernels
import { BaseKernelStrategy } from '../registry.js';
import { KernelLaunchConfig, CudaHardwareConstraints } from '../types.js';
import { GridBlockCalculator } from '../utils.js';

/**
 * Strategy for softmax forward kernel
 */
export class SoftmaxStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length === 2) {
      const [batchSize, features] = shape;
      const blockSize = GridBlockCalculator.optimalBlockSize(features, constraints);
      
      return {
        gridDim: `dim3(${batchSize}, 1, 1)`,
        blockDim: `dim3(${blockSize}, 1, 1)`,
        sharedMemSize: blockSize * 4, // float size
      };
    } else if (shape.length === 4) {
      const [batchSize, numHeads, seqLen, lastDim] = shape;
      const blockSize = GridBlockCalculator.optimalBlockSize(seqLen, constraints);
      
      return {
        gridDim: `dim3(${seqLen}, ${numHeads}, ${batchSize})`,
        blockDim: `dim3(${blockSize}, 1, 1)`,
        sharedMemSize: blockSize * 4,
      };
    }
    
    // Fallback for other dimensions
    return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
  }
}

/**
 * Strategy for fused scale softmax forward kernel
 */
export class FusedScaleSoftmaxStrategy extends BaseKernelStrategy {
  private readonly MIN_BLOCK_SIZE = 128;
  
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length === 4) {
      const [batchSize, numHeads, seqLen, lastDim] = shape;
      const targetBlockSize = Math.max(this.MIN_BLOCK_SIZE, 
        GridBlockCalculator.alignToWarpSize(seqLen, constraints.warpSize));
      const blockSize = Math.min(targetBlockSize, constraints.maxThreadsPerBlock);
      
      return {
        gridDim: `dim3(${seqLen}, ${numHeads}, ${batchSize})`,
        blockDim: `dim3(${blockSize}, 1, 1)`,
        sharedMemSize: blockSize * 4,
      };
    } else if (shape.length === 2) {
      const [batchSize, features] = shape;
      const blockSize = GridBlockCalculator.optimalBlockSize(features, constraints);
      
      return {
        gridDim: `dim3(${batchSize}, 1, 1)`,
        blockDim: `dim3(${blockSize}, 1, 1)`,
        sharedMemSize: blockSize * 4,
      };
    }
    
    return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
  }
}

/**
 * Strategy for layer normalization forward kernel
 */
export class LayerNormStrategy extends BaseKernelStrategy {
  private readonly MIN_BLOCK_SIZE = 128;
  
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length < 3) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, seqLen, featureCount] = shape;
    const targetBlockSize = Math.max(this.MIN_BLOCK_SIZE,
      GridBlockCalculator.alignToWarpSize(featureCount, constraints.warpSize));
    const blockSize = Math.min(targetBlockSize, constraints.maxThreadsPerBlock);
    
    return {
      gridDim: `dim3(${seqLen}, ${batchSize})`,
      blockDim: `dim3(${blockSize}, 1, 1)`,
      sharedMemSize: blockSize * 4,
    };
  }
}

/**
 * Strategy for simple elementwise operations (scale, add, relu)
 */
export class ElementwiseStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    const totalElements = this.getTotalElements(shape);
    const blockSize = defaultBlockSize;
    const gridSize = Math.ceil(totalElements / blockSize);
    
    return {
      gridDim: `dim3(${gridSize}, 1, 1)`,
      blockDim: `dim3(${blockSize}, 1, 1)`,
      sharedMemSize: 0,
    };
  }
}
