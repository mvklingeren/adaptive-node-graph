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
 * Represents a CUDA stream for asynchronous execution.
 */
export interface CudaStream {
  /** A unique identifier for the CUDA stream. */
  readonly id: string;
  /** Synchronizes the stream (waits for all operations to complete). */
  synchronize(): Promise<void>;
  /** Destroys the stream and frees resources. */
  destroy(): Promise<void>;
}

/**
 * Represents a cuBLAS handle for optimized linear algebra operations.
 */
export interface CublasHandle {
  /** A unique identifier for the cuBLAS handle. */
  readonly id: string;
  /** Sets the stream for this cuBLAS handle. */
  setStream(stream: CudaStream): Promise<void>;
  /** Destroys the handle and frees resources. */
  destroy(): Promise<void>;
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
   * @param stream - Optional CUDA stream for asynchronous execution.
   */
  launch(
    gridDim: { x: number; y?: number; z?: number },
    blockDim: { x: number; y?: number; z?: number },
    sharedMemBytes: number,
    args: CudaTensor[],
    stream?: CudaStream
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
   * @param filename - The name of the file being compiled.
   * @returns A promise that resolves to a CudaKernel object.
   */
  compile(kernelCode: string, filename: string): Promise<CudaKernel>;

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

  // ============================================================================
  // CUDA Streams Support
  // ============================================================================

  /**
   * Creates a new CUDA stream for asynchronous execution.
   * @returns A promise that resolves to a CudaStream object.
   */
  createStream(): Promise<CudaStream>;

  /**
   * Destroys a CUDA stream and frees its resources.
   * @param stream - The stream to destroy.
   */
  destroyStream(stream: CudaStream): Promise<void>;

  /**
   * Sets the current stream for subsequent operations.
   * @param stream - The stream to set as current, or null for default stream.
   */
  setCurrentStream(stream: CudaStream | null): void;
}

/**
 * Extended runtime interface that includes cuBLAS support.
 * This interface extends CudaRuntime with optimized linear algebra operations.
 */
export interface CublasRuntime extends CudaRuntime {
  // ============================================================================
  // cuBLAS Handle Management
  // ============================================================================

  /**
   * Creates a new cuBLAS handle for optimized linear algebra operations.
   * @returns A promise that resolves to a CublasHandle object.
   */
  createCublasHandle(): Promise<CublasHandle>;

  /**
   * Destroys a cuBLAS handle and frees its resources.
   * @param handle - The handle to destroy.
   */
  destroyCublasHandle(handle: CublasHandle): Promise<void>;

  // ============================================================================
  // Optimized GEMM Operations
  // ============================================================================

  /**
   * Performs a general matrix-matrix multiplication using cuBLAS.
   * C = alpha * op(A) * op(B) + beta * C
   * @param handle - The cuBLAS handle.
   * @param stream - The CUDA stream for asynchronous execution.
   * @param transa - Transpose operation for matrix A ('N' = no transpose, 'T' = transpose).
   * @param transb - Transpose operation for matrix B ('N' = no transpose, 'T' = transpose).
   * @param m - Number of rows of matrix op(A) and C.
   * @param n - Number of columns of matrix op(B) and C.
   * @param k - Number of columns of op(A) and rows of op(B).
   * @param alpha - Scalar multiplier for A*B.
   * @param A - Input matrix A.
   * @param lda - Leading dimension of A.
   * @param B - Input matrix B.
   * @param ldb - Leading dimension of B.
   * @param beta - Scalar multiplier for C.
   * @param C - Input/output matrix C.
   * @param ldc - Leading dimension of C.
   */
  gemmEx(
    handle: CublasHandle,
    stream: CudaStream,
    transa: 'N' | 'T',
    transb: 'N' | 'T',
    m: number,
    n: number,
    k: number,
    alpha: number,
    A: CudaTensor,
    lda: number,
    B: CudaTensor,
    ldb: number,
    beta: number,
    C: CudaTensor,
    ldc: number
  ): Promise<void>;

  /**
   * Performs batched general matrix-matrix multiplication using cuBLAS.
   * Performs multiple GEMM operations in a single call for better performance.
   * @param handle - The cuBLAS handle.
   * @param stream - The CUDA stream for asynchronous execution.
   * @param transa - Transpose operation for matrix A ('N' = no transpose, 'T' = transpose).
   * @param transb - Transpose operation for matrix B ('N' = no transpose, 'T' = transpose).
   * @param m - Number of rows of matrix op(A) and C.
   * @param n - Number of columns of matrix op(B) and C.
   * @param k - Number of columns of op(A) and rows of op(B).
   * @param alpha - Scalar multiplier for A*B.
   * @param A - Array of input matrices A.
   * @param lda - Leading dimension of A matrices.
   * @param B - Array of input matrices B.
   * @param ldb - Leading dimension of B matrices.
   * @param beta - Scalar multiplier for C.
   * @param C - Array of input/output matrices C.
   * @param ldc - Leading dimension of C matrices.
   * @param batchCount - Number of matrices in the batch.
   */
  gemmBatchedEx(
    handle: CublasHandle,
    stream: CudaStream,
    transa: 'N' | 'T',
    transb: 'N' | 'T',
    m: number,
    n: number,
    k: number,
    alpha: number,
    A: CudaTensor[],
    lda: number,
    B: CudaTensor[],
    ldb: number,
    beta: number,
    C: CudaTensor[],
    ldc: number,
    batchCount: number
  ): Promise<void>;

  /**
   * Performs strided batched GEMM for tensors with regular stride patterns.
   * More efficient than gemmBatchedEx when matrices are stored contiguously.
   * @param handle - The cuBLAS handle.
   * @param stream - The CUDA stream for asynchronous execution.
   * @param transa - Transpose operation for matrix A.
   * @param transb - Transpose operation for matrix B.
   * @param m - Number of rows of matrix op(A) and C.
   * @param n - Number of columns of matrix op(B) and C.
   * @param k - Number of columns of op(A) and rows of op(B).
   * @param alpha - Scalar multiplier for A*B.
   * @param A - Input tensor containing batch of matrices A.
   * @param lda - Leading dimension of A matrices.
   * @param strideA - Stride between consecutive A matrices.
   * @param B - Input tensor containing batch of matrices B.
   * @param ldb - Leading dimension of B matrices.
   * @param strideB - Stride between consecutive B matrices.
   * @param beta - Scalar multiplier for C.
   * @param C - Input/output tensor containing batch of matrices C.
   * @param ldc - Leading dimension of C matrices.
   * @param strideC - Stride between consecutive C matrices.
   * @param batchCount - Number of matrices in the batch.
   */
  gemmStridedBatchedEx(
    handle: CublasHandle,
    stream: CudaStream,
    transa: 'N' | 'T',
    transb: 'N' | 'T',
    m: number,
    n: number,
    k: number,
    alpha: number,
    A: CudaTensor,
    lda: number,
    strideA: number,
    B: CudaTensor,
    ldb: number,
    strideB: number,
    beta: number,
    C: CudaTensor,
    ldc: number,
    strideC: number,
    batchCount: number
  ): Promise<void>;
}

/**
 * A mock implementation of the CudaRuntime for testing and development.
 * This allows the graph logic to be built and tested without a native addon.
 */
export class MockCudaRuntime implements CudaRuntime {
  private nextTensorId = 0;
  private nextKernelId = 0;
  private nextStreamId = 0;
  private memory = new Map<string, Buffer>();
  private currentStream: CudaStream | null = null;

  async compile(kernelCode: string, filename: string): Promise<CudaKernel> {
    const kernelId = `mock_kernel_${this.nextKernelId++}`;
    console.log(`[MockCudaRuntime] Compiling kernel ${kernelId}:\n--- KERNEL CODE ---\n${kernelCode}\n--------------------`);
    
    // In a real implementation, we would save the kernel code to the specified filename
    // For the mock, we can just log it.
    console.log(`Kernel code written to ${filename}`);
    
    return {
      id: kernelId,
      launch: async (grid, block, shared, args, stream) => {
        const streamInfo = stream ? ` on stream ${stream.id}` : ' on default stream';
        console.log(`[MockCudaRuntime] Launching kernel ${kernelId} with ${args.length} args${streamInfo}.`);
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

  // ============================================================================
  // CUDA Streams Support
  // ============================================================================

  async createStream(): Promise<CudaStream> {
    const streamId = `mock_stream_${this.nextStreamId++}`;
    console.log(`[MockCudaRuntime] Created stream ${streamId}`);
    
    return {
      id: streamId,
      synchronize: async () => {
        console.log(`[MockCudaRuntime] Synchronizing stream ${streamId}`);
        // In a real implementation, this would wait for all operations on the stream to complete
      },
      destroy: async () => {
        console.log(`[MockCudaRuntime] Destroyed stream ${streamId}`);
        // In a real implementation, this would free the stream resources
      },
    };
  }

  async destroyStream(stream: CudaStream): Promise<void> {
    console.log(`[MockCudaRuntime] Destroying stream ${stream.id}`);
    await stream.destroy();
  }

  setCurrentStream(stream: CudaStream | null): void {
    this.currentStream = stream;
    const streamInfo = stream ? stream.id : 'default stream';
    console.log(`[MockCudaRuntime] Set current stream to ${streamInfo}`);
  }
}

/**
 * A mock implementation of the CublasRuntime for testing and development.
 * This extends MockCudaRuntime with cuBLAS functionality.
 */
export class MockCublasRuntime extends MockCudaRuntime implements CublasRuntime {
  private nextHandleId = 0;

  // ============================================================================
  // cuBLAS Handle Management
  // ============================================================================

  async createCublasHandle(): Promise<CublasHandle> {
    const handleId = `mock_cublas_handle_${this.nextHandleId++}`;
    console.log(`[MockCublasRuntime] Created cuBLAS handle ${handleId}`);
    
    return {
      id: handleId,
      setStream: async (stream: CudaStream) => {
        console.log(`[MockCublasRuntime] Set cuBLAS handle ${handleId} to use stream ${stream.id}`);
      },
      destroy: async () => {
        console.log(`[MockCublasRuntime] Destroyed cuBLAS handle ${handleId}`);
      },
    };
  }

  async destroyCublasHandle(handle: CublasHandle): Promise<void> {
    console.log(`[MockCublasRuntime] Destroying cuBLAS handle ${handle.id}`);
    await handle.destroy();
  }

  // ============================================================================
  // Optimized GEMM Operations
  // ============================================================================

  async gemmEx(
    handle: CublasHandle,
    stream: CudaStream,
    transa: 'N' | 'T',
    transb: 'N' | 'T',
    m: number,
    n: number,
    k: number,
    alpha: number,
    A: CudaTensor,
    lda: number,
    B: CudaTensor,
    ldb: number,
    beta: number,
    C: CudaTensor,
    ldc: number
  ): Promise<void> {
    console.log(`[MockCublasRuntime] cuBLAS GEMM: ${transa}${transb} [${m}x${n}x${k}] alpha=${alpha} beta=${beta}`);
    console.log(`  Handle: ${handle.id}, Stream: ${stream.id}`);
    console.log(`  A: ${A.id} (${A.shape.join('x')}), B: ${B.id} (${B.shape.join('x')}), C: ${C.id} (${C.shape.join('x')})`);
    // In a real implementation, this would call cublasGemmEx
  }

  async gemmBatchedEx(
    handle: CublasHandle,
    stream: CudaStream,
    transa: 'N' | 'T',
    transb: 'N' | 'T',
    m: number,
    n: number,
    k: number,
    alpha: number,
    A: CudaTensor[],
    lda: number,
    B: CudaTensor[],
    ldb: number,
    beta: number,
    C: CudaTensor[],
    ldc: number,
    batchCount: number
  ): Promise<void> {
    console.log(`[MockCublasRuntime] cuBLAS Batched GEMM: ${transa}${transb} [${m}x${n}x${k}] batch=${batchCount}`);
    console.log(`  Handle: ${handle.id}, Stream: ${stream.id}`);
    console.log(`  A: ${A.length} tensors, B: ${B.length} tensors, C: ${C.length} tensors`);
    // In a real implementation, this would call cublasGemmBatchedEx
  }

  async gemmStridedBatchedEx(
    handle: CublasHandle,
    stream: CudaStream,
    transa: 'N' | 'T',
    transb: 'N' | 'T',
    m: number,
    n: number,
    k: number,
    alpha: number,
    A: CudaTensor,
    lda: number,
    strideA: number,
    B: CudaTensor,
    ldb: number,
    strideB: number,
    beta: number,
    C: CudaTensor,
    ldc: number,
    strideC: number,
    batchCount: number
  ): Promise<void> {
    console.log(`[MockCublasRuntime] cuBLAS Strided Batched GEMM: ${transa}${transb} [${m}x${n}x${k}] batch=${batchCount}`);
    console.log(`  Handle: ${handle.id}, Stream: ${stream.id}`);
    console.log(`  A: ${A.id} stride=${strideA}, B: ${B.id} stride=${strideB}, C: ${C.id} stride=${strideC}`);
    // In a real implementation, this would call cublasGemmStridedBatchedEx
  }
}
