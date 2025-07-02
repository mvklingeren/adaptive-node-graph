// Strategies for dense layer kernels
import { BaseKernelStrategy } from '../registry.js';
import { KernelLaunchConfig, CudaHardwareConstraints } from '../types.js';
import { GridBlockCalculator } from '../utils.js';

/**
 * Strategy for 2D dense forward kernels
 */
export class DenseForward2DStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length !== 2) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, outputFeatures] = shape;
    const blockSize = defaultBlockSize;
    const gridX = Math.ceil(outputFeatures / blockSize);
    
    return {
      gridDim: `dim3(${Math.max(gridX, 1)}, ${batchSize})`,
      blockDim: `dim3(${blockSize}, 1, 1)`,
      sharedMemSize: 0,
    };
  }
}

/**
 * Strategy for 3D dense forward kernels
 */
export class DenseForward3DStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length !== 3) {
      return this.create1DConfig(this.getTotalElements(shape), defaultBlockSize, constraints);
    }
    
    const [batchSize, seqLen, outputFeatures] = shape;
    const blockSize = defaultBlockSize;
    const gridX = Math.ceil(outputFeatures / blockSize);
    
    return {
      gridDim: `dim3(${gridX}, ${seqLen}, ${batchSize})`,
      blockDim: `dim3(${blockSize}, 1, 1)`,
      sharedMemSize: 0,
    };
  }
}
