// Strategies for embedding-related kernels
import { BaseKernelStrategy } from '../registry.js';
import { KernelLaunchConfig, CudaHardwareConstraints } from '../types.js';
import { GridBlockCalculator } from '../utils.js';

/**
 * Strategy for embedding forward kernel
 */
export class EmbeddingKernelStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length < 3) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, seqLen, embedDim] = shape;
    const blockSize = GridBlockCalculator.optimalBlockSize(embedDim, constraints);
    const finalBlockSize = Math.min(blockSize, constraints.maxThreadsPerBlock);
    
    return {
      gridDim: `dim3(${seqLen}, ${batchSize})`,
      blockDim: `dim3(${finalBlockSize}, 1, 1)`,
      sharedMemSize: 0,
    };
  }
}

/**
 * Strategy for positional encoding forward kernel
 */
export class PositionalEncodingStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length < 3) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, seqLen, embedDim] = shape;
    const totalElements = batchSize * seqLen * embedDim;
    const blockSize = defaultBlockSize;
    const gridSize = Math.ceil(totalElements / blockSize);
    
    return {
      gridDim: `dim3(${gridSize}, 1, 1)`,
      blockDim: `dim3(${blockSize}, 1, 1)`,
      sharedMemSize: 0,
    };
  }
}
