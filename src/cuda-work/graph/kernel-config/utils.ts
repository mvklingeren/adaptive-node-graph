// Utility functions for grid and block dimension calculations
import { CudaHardwareConstraints } from './types.js';

export class GridBlockCalculator {
  /**
   * Aligns a size to the nearest multiple of warp size
   */
  static alignToWarpSize(size: number, warpSize: number = 32): number {
    return Math.ceil(size / warpSize) * warpSize;
  }
  
  /**
   * Calculates 1D grid dimensions for a given number of elements
   */
  static calculate1DGrid(totalElements: number, blockSize: number, maxGrid: number = 65535): string {
    const gridSize = Math.min(Math.ceil(totalElements / blockSize), maxGrid);
    return `dim3(${gridSize}, 1, 1)`;
  }
  
  /**
   * Calculates 2D grid dimensions
   */
  static calculate2DGrid(rows: number, cols: number, blockSize: number): string {
    const gridX = Math.ceil(cols / blockSize);
    return `dim3(${gridX}, ${rows})`;
  }
  
  /**
   * Calculates 3D grid dimensions
   */
  static calculate3DGrid(x: number, y: number, z: number, blockSizeX: number): string {
    const gridX = Math.ceil(x / blockSizeX);
    return `dim3(${gridX}, ${y}, ${z})`;
  }
  
  /**
   * Determines optimal block size based on work per block and hardware constraints
   */
  static optimalBlockSize(
    workPerBlock: number, 
    constraints: CudaHardwareConstraints,
    minWarps: number = 3
  ): number {
    const minSize = Math.max(constraints.warpSize * minWarps, 96);
    const maxSize = constraints.maxThreadsPerBlock;
    const aligned = this.alignToWarpSize(Math.min(workPerBlock, maxSize), constraints.warpSize);
    return Math.max(minSize, Math.min(aligned, maxSize));
  }
  
  /**
   * Clamps a value between min and max
   */
  static clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(value, max));
  }
  
  /**
   * Creates a dim3 string
   */
  static dim3(x: number, y: number = 1, z: number = 1): string {
    return `dim3(${x}, ${y}, ${z})`;
  }
}
