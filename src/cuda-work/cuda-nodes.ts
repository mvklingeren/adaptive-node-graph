// ============================================================================
// Standard CUDA Node Library
// ============================================================================
// This file provides a library of pre-defined, reusable CudaNode subclasses
// for common computational tasks. These nodes are designed to be composed
// into a CudaGraph and fused into a single, efficient kernel by the
// CudaGraphCompiler.
// ============================================================================

import { CudaNode } from "./cuda-graph.js";
import { tsToCuda } from "./ts-to-cuda.js";
import * as ts from "typescript";

// ============================================================================
// Custom CUDA Node from TypeScript
// ============================================================================

export class CustomCudaNode extends CudaNode {
  private readonly tsCode: string;

  constructor(tsCode: string) {
    const { functionName, deviceCode } = tsToCuda(tsCode);
    super(deviceCode, functionName);
    this.name = functionName;
    this.tsCode = tsCode;
    this.extractInputsAndOutputs();

    this.setShapeResolver((inputs) => {
      const resolvedShapes = new Map<string, { shape: number[] }>();
      const firstInput = inputs.values().next().value;
      if (firstInput) {
        for (const outputName of this.outputs.keys()) {
          resolvedShapes.set(outputName, { shape: firstInput.shape });
        }
      }
      return resolvedShapes;
    });
  }

  private extractInputsAndOutputs(): void {
    const sourceFile = ts.createSourceFile(
      "temp.ts",
      this.tsCode,
      ts.ScriptTarget.Latest,
      true
    );

    const visit = (node: ts.Node) => {
      if (ts.isFunctionDeclaration(node)) {
        const params = node.parameters;
        if (params.length > 0) {
          // All but the last are inputs
          for (let i = 0; i < params.length - 1; i++) {
            const p = params[i];
            const name = p.name.getText(sourceFile);
            this.addInput(name, [-1], "float32");
          }
          // Last one is output
          const outputParam = params[params.length - 1];
          const outputName = outputParam.name.getText(sourceFile);
          this.addOutput(outputName, [-1], "float32");
        }
      }
      ts.forEachChild(node, visit);
    };

    visit(sourceFile);
  }
}

// ============================================================================
// Element-wise Add Node
// ============================================================================
// Element-wise Add Node
// ============================================================================

const addDeviceCode = `
__device__ void add(Tensor<float> C, const Tensor<float> A, const Tensor<float> B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Add boundary checks for safety
    C(i) = A(i) + B(i);
}
`;

export class AddNode extends CudaNode {
  constructor() {
    super(addDeviceCode, "add");
    this.name = "Add";
    this.addInput("A", [-1], "float32");
    this.addInput("B", [-1], "float32");
    this.addOutput("C", [-1], "float32");

    // Shape resolver: Output shape is the same as the input shapes.
    this.setShapeResolver((inputs) => {
      const shapeA = inputs.get("A")!.shape;
      const shapeB = inputs.get("B")!.shape;
      // Basic validation: for element-wise, shapes must match.
      if (JSON.stringify(shapeA) !== JSON.stringify(shapeB)) {
        throw new Error(`Shape mismatch for AddNode: A is ${shapeA}, B is ${shapeB}`);
      }
      return new Map([["C", { shape: shapeA }]]);
    });
  }
}

// ============================================================================
// Element-wise Multiply Node
// ============================================================================

const multiplyDeviceCode = `
__device__ void multiply(Tensor<float> C, const Tensor<float> A, const Tensor<float> B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Add boundary checks for safety
    C(i) = A(i) * B(i);
}
`;

export class MultiplyNode extends CudaNode {
  constructor() {
    super(multiplyDeviceCode, "multiply");
    this.name = "Multiply";
    this.addInput("A", [-1], "float32");
    this.addInput("B", [-1], "float32");
    this.addOutput("C", [-1], "float32");

    // Shape resolver: Output shape is the same as the input shapes.
    this.setShapeResolver((inputs) => {
      const shapeA = inputs.get("A")!.shape;
      const shapeB = inputs.get("B")!.shape;
      if (JSON.stringify(shapeA) !== JSON.stringify(shapeB)) {
        throw new Error(`Shape mismatch for MultiplyNode: A is ${shapeA}, B is ${shapeB}`);
      }
      return new Map([["C", { shape: shapeA }]]);
    });
  }
}

// ============================================================================
// ReLU (Rectified Linear Unit) Activation Node
// ============================================================================

const reluDeviceCode = `
__device__ void relu(Tensor<float> output, const Tensor<float> input) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Add boundary checks for safety
    float val = input(i);
    output(i) = val > 0 ? val : 0;
}
`;

export class ReluNode extends CudaNode {
  constructor() {
    super(reluDeviceCode, "relu");
    this.name = "ReLU";
    this.addInput("input", [-1], "float32");
    this.addOutput("output", [-1], "float32");

    // Shape resolver: Output shape is identical to input shape.
    this.setShapeResolver((inputs) => {
      const inputShape = inputs.get("input")!.shape;
      return new Map([["output", { shape: inputShape }]]);
    });
  }
}

// ============================================================================
// Matrix Multiplication Node (MatMul)
// ============================================================================
// Performs C = A * B where A is (M, K) and B is (K, N), resulting in C (M, N).

const matmulDeviceCode = `
__device__ void matmul(Tensor<float> C, const Tensor<float> A, const Tensor<float> B) {
    int m = blockIdx.y * blockDim.y + threadIdx.y; // Row index for C
    int n = blockIdx.x * blockDim.x + threadIdx.x; // Col index for C

    int M = C.shape[0];
    int N = C.shape[1];
    int K = A.shape[1];

    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A(m, k) * B(k, n);
        }
        C(m, n) = sum;
    }
}
`;

export class MatMulNode extends CudaNode {
  constructor() {
    super(matmulDeviceCode, "matmul");
    this.name = "MatMul";
    // A: [M, K], B: [K, N]
    this.addInput("A", [-1, -1], "float32");
    this.addInput("B", [-1, -1], "float32");
    // C: [M, N]
    this.addOutput("C", [-1, -1], "float32");

    this.setShapeResolver((inputs) => {
      const shapeA = inputs.get("A")!.shape; // [M, K]
      const shapeB = inputs.get("B")!.shape; // [K, N]

      if (shapeA.length !== 2 || shapeB.length !== 2) {
        throw new Error("MatMul inputs must be 2D tensors.");
      }
      if (shapeA[1] !== shapeB[0]) {
        throw new Error(`Inner dimension mismatch for MatMul: A is [${shapeA}] and B is [${shapeB}]`);
      }

      const M = shapeA[0];
      const N = shapeB[1];
      return new Map([["C", { shape: [M, N] }]]);
    });
  }
}

// ============================================================================
// Tensor Core Enhanced Matrix Multiplication Node
// ============================================================================
// High-performance matrix multiplication using Tensor Cores on Volta+ GPUs
// with automatic fallback to standard CUDA cores for compatibility.

const tensorCoreMatmulDeviceCode = `
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// GPU capability detection
__device__ bool supportsTensorCores() {
    // This will be determined at compile time based on compute capability
    #if __CUDA_ARCH__ >= 700
    return true;
    #else
    return false;
    #endif
}

// Tensor Core implementation using WMMA API
__global__ void tensor_core_matmul(
    Tensor<float> C, 
    const Tensor<float> A, 
    const Tensor<float> B
) {
    // Matrix dimensions
    const int M = C.shape[0];
    const int N = C.shape[1]; 
    const int K = A.shape[1];
    
    // WMMA tile size (16x16x16 for Volta+)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate which tile this thread block is responsible for
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Calculate the matrix coordinates for this warp's tile
    const int cRow = warpM * WMMA_M;
    const int cCol = warpN * WMMA_N;
    
    // Check if this warp has work to do
    if (cRow >= M || cCol >= N) return;
    
    #if __CUDA_ARCH__ >= 700
    // Tensor Core path for Volta+ GPUs
    
    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // Initialize accumulator fragment to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Shared memory for FP32 to FP16 conversion
    __shared__ half A_shared[WMMA_M * WMMA_K];
    __shared__ half B_shared[WMMA_K * WMMA_N];
    
    // Loop over K dimension in WMMA_K chunks
    for (int i = 0; i < K; i += WMMA_K) {
        // Convert FP32 input to FP16 and load into shared memory
        // Each thread in the warp handles multiple elements
        for (int t = threadIdx.x; t < WMMA_M * WMMA_K; t += blockDim.x) {
            int row = t / WMMA_K;
            int col = t % WMMA_K;
            int globalRow = cRow + row;
            int globalCol = i + col;
            
            if (globalRow < M && globalCol < K) {
                A_shared[t] = __float2half(A(globalRow, globalCol));
            } else {
                A_shared[t] = __float2half(0.0f);
            }
        }
        
        for (int t = threadIdx.x; t < WMMA_K * WMMA_N; t += blockDim.x) {
            int row = t / WMMA_N;
            int col = t % WMMA_N;
            int globalRow = i + row;
            int globalCol = cCol + col;
            
            if (globalRow < K && globalCol < N) {
                B_shared[t] = __float2half(B(globalRow, globalCol));
            } else {
                B_shared[t] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Load fragments from shared memory
        wmma::load_matrix_sync(a_frag, A_shared, WMMA_K);
        wmma::load_matrix_sync(b_frag, B_shared, WMMA_N);
        
        // Perform matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        __syncthreads();
    }
    
    // Store result back to global memory
    float C_frag[WMMA_M * WMMA_N];
    wmma::store_matrix_sync(C_frag, acc_frag, WMMA_N, wmma::mem_row_major);
    
    // Copy from fragment to global memory
    for (int t = threadIdx.x; t < WMMA_M * WMMA_N; t += blockDim.x) {
        int row = t / WMMA_N;
        int col = t % WMMA_N;
        int globalRow = cRow + row;
        int globalCol = cCol + col;
        
        if (globalRow < M && globalCol < N) {
            C(globalRow, globalCol) = C_frag[t];
        }
    }
    
    #else
    // Fallback to standard CUDA cores for older GPUs
    
    // Use shared memory tiling for better performance
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A(row, t * TILE_SIZE + tx);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B(t * TILE_SIZE + ty, col);
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C(row, col) = sum;
    }
    
    #endif
}

// Alternative cuBLAS-based implementation for maximum performance
__global__ void cublas_matmul_wrapper(
    Tensor<float> C,
    const Tensor<float> A,
    const Tensor<float> B
) {
    // This kernel serves as a placeholder for cuBLAS integration
    // In a real implementation, this would call cublasGemmEx
    // with appropriate Tensor Core optimizations
    
    // For now, fall back to the tensor core implementation
    tensor_core_matmul<<<gridDim, blockDim>>>(C, A, B);
}
`;

export class TensorCoreMatMulNode extends CudaNode {
  private useTensorCores: boolean;
  private precision: 'fp16' | 'bf16' | 'fp32';
  private useCuBLAS: boolean;

  constructor(
    useTensorCores: boolean = true, 
    precision: 'fp16' | 'bf16' | 'fp32' = 'fp16',
    useCuBLAS: boolean = false
  ) {
    const functionName = useCuBLAS ? "cublas_matmul_wrapper" : "tensor_core_matmul";
    super(tensorCoreMatmulDeviceCode, functionName);
    
    this.name = "TensorCoreMatMul";
    this.useTensorCores = useTensorCores;
    this.precision = precision;
    this.useCuBLAS = useCuBLAS;
    
    // Input/Output configuration
    // For now, we keep FP32 interface and handle conversion internally
    this.addInput("A", [-1, -1], "float32");
    this.addInput("B", [-1, -1], "float32");
    this.addOutput("C", [-1, -1], "float32");

    this.setShapeResolver((inputs) => {
      const shapeA = inputs.get("A")!.shape; // [M, K]
      const shapeB = inputs.get("B")!.shape; // [K, N]

      if (shapeA.length !== 2 || shapeB.length !== 2) {
        throw new Error("TensorCoreMatMul inputs must be 2D tensors.");
      }
      if (shapeA[1] !== shapeB[0]) {
        throw new Error(`Inner dimension mismatch for TensorCoreMatMul: A is [${shapeA}] and B is [${shapeB}]`);
      }

      const M = shapeA[0];
      const N = shapeB[1];
      
      // Log optimization info
      console.log(`[TensorCoreMatMul] Matrix dimensions: ${M}x${shapeA[1]} * ${shapeA[1]}x${N} = ${M}x${N}`);
      console.log(`[TensorCoreMatMul] Using Tensor Cores: ${this.useTensorCores}, Precision: ${this.precision}`);
      
      return new Map([["C", { shape: [M, N] }]]);
    });
  }

  // Override the kernel call generation to use optimal grid/block configuration
  getKernelCall(
    outputTensorNames: Map<string, string>,
    inputTensorNames: Map<string, string>
  ): string {
    const outputArgs = Array.from(this.outputs.keys()).map(name => outputTensorNames.get(name));
    const inputArgs = Array.from(this.inputs.keys()).map(name => inputTensorNames.get(name));
    const paramArgs = Array.from(this.parameters.keys());

    const allArgs = [...outputArgs, ...inputArgs, ...paramArgs].join(", ");
    
    // Calculate optimal grid/block dimensions for Tensor Cores
    // For WMMA, we need warps to handle 16x16 tiles
    const blockDim = "dim3(32, 1, 1)"; // One warp per block
    const gridDim = "dim3((N + 15) / 16, (M + 15) / 16, 1)"; // Tile the matrix
    
    return `${this.functionName}<<<${gridDim}, ${blockDim}>>>(${allArgs});`;
  }
}
