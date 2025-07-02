// Kernel configuration registry for managing kernel launch strategies
import { KernelConfigStrategy, KernelLaunchConfig, CudaHardwareConstraints, DEFAULT_CUDA_CONSTRAINTS } from './types.js';
import { GridBlockCalculator } from './utils.js';

export class KernelConfigRegistry {
  private strategies = new Map<string, KernelConfigStrategy>();
  private defaultStrategy: KernelConfigStrategy;
  
  constructor() {
    this.defaultStrategy = new DefaultKernelStrategy();
  }
  
  /**
   * Registers a strategy for a specific kernel function name
   */
  register(kernelName: string, strategy: KernelConfigStrategy): void {
    this.strategies.set(kernelName, strategy);
  }
  
  /**
   * Gets the launch configuration for a kernel
   */
  getConfig(
    kernelName: string, 
    shape: number[], 
    constraints: CudaHardwareConstraints = DEFAULT_CUDA_CONSTRAINTS,
    defaultBlockSize: number = 256
  ): KernelLaunchConfig {
    const strategy = this.strategies.get(kernelName) || this.defaultStrategy;
    return strategy.calculateConfig(shape, constraints, defaultBlockSize);
  }
  
  /**
   * Gets the default configuration for unknown kernel types
   */
  getDefaultConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints = DEFAULT_CUDA_CONSTRAINTS,
    defaultBlockSize: number = 256
  ): KernelLaunchConfig {
    return this.defaultStrategy.calculateConfig(shape, constraints, defaultBlockSize);
  }
  
  /**
   * Checks if a strategy is registered for a kernel
   */
  hasStrategy(kernelName: string): boolean {
    return this.strategies.has(kernelName);
  }
}

/**
 * Base class for kernel strategies with common functionality
 */
export abstract class BaseKernelStrategy implements KernelConfigStrategy {
  abstract calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints, 
    defaultBlockSize: number
  ): KernelLaunchConfig;
  
  /**
   * Helper to get total number of elements from shape
   */
  protected getTotalElements(shape: number[]): number {
    return shape.reduce((a, b) => a * b, 1);
  }
  
  /**
   * Helper to create a simple 1D grid configuration
   */
  protected create1DConfig(
    totalElements: number, 
    blockSize: number,
    constraints: CudaHardwareConstraints
  ): KernelLaunchConfig {
    const gridDim = GridBlockCalculator.calculate1DGrid(
      totalElements, 
      blockSize, 
      constraints.maxGridDimX
    );
    const blockDim = GridBlockCalculator.dim3(blockSize);
    
    return {
      gridDim,
      blockDim,
      sharedMemSize: 0,
    };
  }
}

/**
 * Default strategy for unknown kernel types
 */
class DefaultKernelStrategy extends BaseKernelStrategy {
  calculateConfig(
    shape: number[], 
    constraints: CudaHardwareConstraints,
    defaultBlockSize: number
  ): KernelLaunchConfig {
    if (shape.length === 0) {
      return {
        gridDim: "dim3(1, 1, 1)",
        blockDim: `dim3(${defaultBlockSize}, 1, 1)`,
        sharedMemSize: 0,
      };
    }
    
    // For multi-dimensional tensors, try to use 2D grid if possible
    if (shape.length >= 2) {
      const batchSize = shape[0];
      const features = shape[shape.length - 1];
      const blockSize = Math.min(features, defaultBlockSize);
      const alignedBlockSize = GridBlockCalculator.alignToWarpSize(blockSize, constraints.warpSize);
      const finalBlockSize = Math.min(alignedBlockSize, constraints.maxThreadsPerBlock);
      const gridSize = Math.ceil(features / finalBlockSize);
      
      return {
        gridDim: `dim3(${gridSize}, ${batchSize})`,
        blockDim: `dim3(${finalBlockSize}, 1, 1)`,
        sharedMemSize: 0,
      };
    }
    
    // Fallback to 1D grid
    const totalElements = this.getTotalElements(shape);
    return this.create1DConfig(totalElements, defaultBlockSize, constraints);
  }
}
