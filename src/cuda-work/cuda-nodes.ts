// ============================================================================
// Standard CUDA Node Library
// ============================================================================
// This file provides a library of pre-defined, reusable CudaNode subclasses
// for common computational tasks. These nodes are designed to be composed
// into a CudaGraph and fused into a single, efficient kernel by the
// CudaGraphCompiler.
// ============================================================================

import { CudaNode } from "./cuda-graph.js";

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
