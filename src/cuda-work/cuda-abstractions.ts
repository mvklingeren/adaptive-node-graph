// ============================================================================
// CUDA Abstractions - The Bridge to Native Code
// ============================================================================
// This file defines the TypeScript interfaces for interacting with a native
// CUDA addon. It creates a clear contract for what the C++ layer must
// implement. For now, we will use a mock/stub implementation of these
// interfaces to allow for development and testing of the graph logic
// without needing a compiled native module.
// ============================================================================

/**
 * Represents a handle to a block of memory allocated on the GPU.
 * In a real implementation, this would likely be a wrapper around a pointer.
 */
export interface CudaTensor {
  /** A unique identifier for the GPU memory block. */
  readonly id: string;
  /** The size of the allocated memory in bytes. */
  readonly byteLength: number;
  /** The dimensions of the tensor. */
  readonly shape: readonly number[];
  /** The data type of the tensor elements. */
  readonly dtype: "float32" | "int32"; // Add more types as needed
  /** Frees the memory on the GPU. */
  free(): Promise<void>;
}

/**
 * Represents a compiled CUDA kernel that is ready to be launched on the GPU.
 */
export interface CudaKernel {
  /** A unique identifier for the compiled kernel. */
  readonly id: string;
  /**
   * Launches the kernel on the GPU with the given arguments.
   * @param gridDim - The dimensions of the grid of thread blocks.
   * @param blockDim - The dimensions of a thread block.
   * @param sharedMemBytes - The amount of shared memory to allocate per block.
   * @param args - The arguments to pass to the kernel (e.g., CudaTensors).
   */
  launch(
    gridDim: { x: number; y?: number; z?: number },
    blockDim: { x: number; y?: number; z?: number },
    sharedMemBytes: number,
    args: CudaTensor[]
  ): Promise<void>;
}

/**
 * Defines the contract for the native CUDA runtime bridge.
 * This is the central API for all GPU operations.
 */
export interface CudaRuntime {
  /**
   * Compiles a string of CUDA C++ code into a launchable kernel.
   * @param kernelCode - The CUDA C++ source code.
   * @returns A promise that resolves to a CudaKernel object.
   */
  compile(kernelCode: string): Promise<CudaKernel>;

  /**
   * Allocates a block of memory on the GPU.
   * @param byteLength - The number of bytes to allocate.
   * @param shape - The dimensions of the tensor.
   * @param dtype - The data type of the tensor elements.
   * @returns A promise that resolves to a CudaTensor representing the GPU memory.
   */
  malloc(byteLength: number, shape?: number[], dtype?: "float32" | "int32"): Promise<CudaTensor>;

  /**
   * Copies data from host (CPU) memory to device (GPU) memory.
   * @param dest - The destination CudaTensor on the GPU.
   * @param src - The source Buffer on the CPU.
   * @param byteLength - The number of bytes to copy.
   * @param destOffset - The offset in bytes in the destination tensor.
   * @param srcOffset - The offset in bytes in the source buffer.
   */
  memcpyHostToDevice(
    dest: CudaTensor,
    src: Buffer,
    byteLength?: number,
    destOffset?: number,
    srcOffset?: number
  ): Promise<void>;

  /**
   * Copies data from device (GPU) memory to host (CPU) memory.
   * @param dest - The destination Buffer on the CPU.
   * @param src - The source CudaTensor on the GPU.
   * @param byteLength - The number of bytes to copy.
   * @param destOffset - The offset in bytes in the destination buffer.
   * @param srcOffset - The offset in bytes in the source tensor.
   */
  memcpyDeviceToHost(
    dest: Buffer,
    src: CudaTensor,
    byteLength?: number,
    destOffset?: number,
    srcOffset?: number
  ): Promise<void>;

  /**
   * Copies data from one device (GPU) memory location to another.
   * @param dest - The destination CudaTensor on the GPU.
   * @param src - The source CudaTensor on the GPU.
   * @param byteLength - The number of bytes to copy.
   */
  memcpyDeviceToDevice(dest: CudaTensor, src: CudaTensor, byteLength?: number): Promise<void>;

  /**
   * Fills a device memory region with a specific byte value.
   * @param tensor - The CudaTensor to fill.
   * @param value - The byte value to set.
   * @param byteLength - The number of bytes to fill.
   */
  memset(tensor: CudaTensor, value: number, byteLength?: number): Promise<void>;

  /**
   * Retrieves information about the available GPU devices.
   */
  getDeviceInfo(): Promise<any>;
}

/**
 * A mock implementation of the CudaRuntime for testing and development.
 * This allows the graph logic to be built and tested without a native addon.
 */
export class MockCudaRuntime implements CudaRuntime {
  private nextTensorId = 0;
  private nextKernelId = 0;
  private memory = new Map<string, Buffer>();

  async compile(kernelCode: string): Promise<CudaKernel> {
    const kernelId = `mock_kernel_${this.nextKernelId++}`;
    console.log(`[MockCudaRuntime] Compiling kernel ${kernelId}:\n--- KERNEL CODE ---\n${kernelCode}\n--------------------`);
    return {
      id: kernelId,
      launch: async (grid, block, shared, args) => {
        console.log(`[MockCudaRuntime] Launching kernel ${kernelId} with ${args.length} args.`);
        // In a real scenario, this would trigger GPU execution.
        // Here we can add mock logic if needed, e.g., logging tensor contents.
      },
    };
  }

  async malloc(byteLength: number, shape: number[] = [byteLength / 4], dtype: "float32" | "int32" = "float32"): Promise<CudaTensor> {
    const tensorId = `mock_tensor_${this.nextTensorId++}`;
    this.memory.set(tensorId, Buffer.alloc(byteLength));
    console.log(`[MockCudaRuntime] Allocated ${byteLength} bytes for tensor ${tensorId}`);
    return {
      id: tensorId,
      byteLength,
      shape,
      dtype,
      free: async () => {
        this.memory.delete(tensorId);
        console.log(`[MockCudaRuntime] Freed tensor ${tensorId}`);
      },
    };
  }

  async memcpyHostToDevice(dest: CudaTensor, src: Buffer, byteLength?: number, destOffset?: number, srcOffset?: number): Promise<void> {
    const destBuffer = this.memory.get(dest.id);
    if (!destBuffer) throw new Error(`Tensor ${dest.id} not found`);
    src.copy(destBuffer, destOffset, srcOffset, (srcOffset || 0) + (byteLength || src.byteLength));
    console.log(`[MockCudaRuntime] Copied ${byteLength || src.byteLength} bytes from host to device tensor ${dest.id}`);
  }

  async memcpyDeviceToHost(dest: Buffer, src: CudaTensor, byteLength?: number, destOffset?: number, srcOffset?: number): Promise<void> {
    const srcBuffer = this.memory.get(src.id);
    if (!srcBuffer) throw new Error(`Tensor ${src.id} not found`);
    srcBuffer.copy(dest, destOffset, srcOffset, (srcOffset || 0) + (byteLength || srcBuffer.byteLength));
    console.log(`[MockCudaRuntime] Copied ${byteLength || srcBuffer.byteLength} bytes from device tensor ${src.id} to host`);
  }

  async memcpyDeviceToDevice(dest: CudaTensor, src: CudaTensor, byteLength?: number): Promise<void> {
    const srcBuffer = this.memory.get(src.id);
    const destBuffer = this.memory.get(dest.id);
    if (!srcBuffer) throw new Error(`Tensor ${src.id} not found`);
    if (!destBuffer) throw new Error(`Tensor ${dest.id} not found`);
    srcBuffer.copy(destBuffer, 0, 0, byteLength || src.byteLength);
    console.log(`[MockCudaRuntime] Copied ${byteLength || src.byteLength} bytes from device tensor ${src.id} to ${dest.id}`);
  }

  async memset(tensor: CudaTensor, value: number, byteLength?: number): Promise<void> {
    const buffer = this.memory.get(tensor.id);
    if (!buffer) throw new Error(`Tensor ${tensor.id} not found`);
    buffer.fill(value, 0, byteLength || buffer.byteLength);
    console.log(`[MockCudaRuntime] Set ${byteLength || buffer.byteLength} bytes of tensor ${tensor.id} to ${value}`);
  }

  async getDeviceInfo(): Promise<any> {
    return {
      mockDevice: true,
      name: "Mock CUDA Device",
      totalMemory: 1024 * 1024 * 1024, // 1GB
    };
  }
}
