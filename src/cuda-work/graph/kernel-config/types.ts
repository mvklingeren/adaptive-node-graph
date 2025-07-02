// Types and interfaces for kernel configuration system
export interface KernelLaunchConfig {
  gridDim: string;
  blockDim: string;
  sharedMemSize: number;
}

export interface CudaHardwareConstraints {
  maxThreadsPerBlock: number;
  warpSize: number;
  maxGridDimX: number;
  maxGridDimY: number;
  maxGridDimZ: number;
  maxSharedMemoryPerBlock: number;
}

export const DEFAULT_CUDA_CONSTRAINTS: CudaHardwareConstraints = {
  maxThreadsPerBlock: 1024,
  warpSize: 32,
  maxGridDimX: 65535,
  maxGridDimY: 65535,
  maxGridDimZ: 65535,
  maxSharedMemoryPerBlock: 49152, // 48KB typical
};

export interface KernelConfigStrategy {
  calculateConfig(shape: number[], constraints: CudaHardwareConstraints, defaultBlockSize: number): KernelLaunchConfig;
}
