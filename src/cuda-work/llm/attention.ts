// ============================================================================
// Attention Mechanism
// ============================================================================

import { CudaNode, CudaGraph } from "../cuda-graph";
import { CudaRuntime, CudaTensor } from "../cuda-abstractions";
import { Layer, NeuralGraph } from "../neural-network.js";
import { SoftmaxLayer } from "./llm-layers.js";

// ============================================================================
// Scaled Dot-Product Attention
// ============================================================================

export class ScaledDotProductAttention implements Layer {
  constructor(
    private runtime: CudaRuntime,
    private embedDim: number,
    private numHeads: number
  ) {}

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const [q, k, v] = inputs;
    const headDim = this.embedDim / this.numHeads;
    const scale = 1.0 / Math.sqrt(headDim);

    // Instantiate layers for the attention mechanism
    const matmul_qk = new BatchedMatMul(true); // Transpose K for QK^T
    const scaleLayer = new ScaleLayer(scale);
    const softmaxLayer = new SoftmaxLayer();
    const matmul_sv = new BatchedMatMul(false); // No transpose for SV

    // Build the graph segment for attention with proper connections
    const qkNode = matmul_qk.addToGraph(graph, q, k);
    const scaleNode = graph.addLayer(scaleLayer, qkNode);
    const softmaxNode = graph.addLayer(softmaxLayer, scaleNode);
    const svNode = matmul_sv.addToGraph(graph, softmaxNode, v);

    return svNode;
  }
}

export class ScaleLayer implements Layer {
  constructor(private scale: number) {}

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const deviceCode = `
      /**
       * @cuda global
       */
      void scale_forward(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3];
        
        for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
            output.data[i] = input.data[i] * ${this.scale};
        }
      }
    `;

    const scaleNode = new CudaNode(deviceCode, "scale_forward")
      .addInput("input", [-1, -1, -1, -1], "float32")
      .addOutput("output", [-1, -1, -1, -1], "float32")
      .setShapeResolver((inputs) => {
        const inputShape = inputs.get("input")!.shape;
        return new Map([["output", { shape: inputShape }]]);
      });

    graph.addNode(scaleNode);
    return scaleNode;
  }
}

export class BatchedMatMul implements Layer {
  private functionName: string;

  constructor(private transposeB: boolean = false) {
    this.functionName = this.transposeB ? "batched_matmul_transpose_b" : "batched_matmul";
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    // Corrected indexing for transpose
    const b_access = this.transposeB 
      ? "b(batch_idx, head_idx, col, k)" 
      : "b(batch_idx, head_idx, k, col)";

    const deviceCode = `
      /**
       * @cuda global
       */
      void ${this.functionName}(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int batch_idx = blockIdx.z;
        int head_idx = blockIdx.y;
        int row = blockIdx.x;
        int col = threadIdx.x;

        if (batch_idx < a.shape[0] && head_idx < a.shape[1] && row < a.shape[2] && col < output.shape[3]) {
          float sum = 0.0f;
          for (int k = 0; k < a.shape[3]; ++k) {
            sum += a(batch_idx, head_idx, row, k) * ${b_access};
          }
          output(batch_idx, head_idx, row, col) = sum;
        }
      }
    `;

    const matmulNode = new CudaNode(deviceCode, this.functionName)
      .addInput("a", [-1, -1, -1, -1], "float32")
      .addInput("b", [-1, -1, -1, -1], "float32")
      .addOutput("output", [-1, -1, -1, -1], "float32")
      .setShapeResolver((inputs) => {
        const aShape = inputs.get("a")!.shape;
        const bShape = inputs.get("b")!.shape;
        const [batchSize, numHeads, m, k] = aShape;
        const n = this.transposeB ? bShape[2] : bShape[3];
        return new Map([["output", { shape: [batchSize, numHeads, m, n] }]]);
      });

    graph.addNode(matmulNode);
    if (inputs.length > 1) {
        graph.connect(inputs[0], 'output', matmulNode, 'a');
        graph.connect(inputs[1], 'output', matmulNode, 'b');
    }
    return matmulNode;
  }
}

// ============================================================================
// Utility Layers for Attention
// ============================================================================

export class SplitHeads implements Layer {
  constructor(private embedDim: number, private numHeads: number) {}

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const headDim = this.embedDim / this.numHeads;
    const deviceCode = `
      /**
       * @cuda global
       */
      void split_heads_forward(Tensor<float> output, Tensor<float> input) {
        // Input: [batch, seq_len, embed_dim]
        // Output: [batch, num_heads, seq_len, head_dim]
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int head_idx = blockIdx.x;
        int feature_idx = threadIdx.x;

        int num_heads = output.shape[1];
        int head_dim = output.shape[3];

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1] && head_idx < num_heads && feature_idx < head_dim) {
          int embed_idx = head_idx * head_dim + feature_idx;
          output(batch_idx, head_idx, seq_idx, feature_idx) = input(batch_idx, seq_idx, embed_idx);
        }
      }
    `;

    const splitHeadsNode = new CudaNode(deviceCode, "split_heads_forward")
      .addInput("input", [-1, -1, this.embedDim], "float32")
      .addOutput("output", [-1, this.numHeads, -1, headDim], "float32")
      .setShapeResolver((inputs) => {
        const inputShape = inputs.get("input")!.shape;
        const [batchSize, seqLen] = inputShape;
        return new Map([
          ["output", { shape: [batchSize, this.numHeads, seqLen, headDim] }],
        ]);
      });

    graph.addNode(splitHeadsNode);
    return splitHeadsNode;
  }
}

export class ConcatHeads implements Layer {
  constructor(private embedDim: number, private numHeads: number) {}

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const headDim = this.embedDim / this.numHeads;
    const deviceCode = `
      /**
       * @cuda global
       */
      void concat_heads_forward(Tensor<float> output, Tensor<float> input) {
        // Input: [batch, num_heads, seq_len, head_dim]
        // Output: [batch, seq_len, embed_dim]
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int head_idx = blockIdx.x;
        int feature_idx = threadIdx.x;

        int num_heads = input.shape[1];
        int head_dim = input.shape[3];

        if (batch_idx < output.shape[0] && seq_idx < output.shape[1] && head_idx < num_heads && feature_idx < head_dim) {
          int embed_idx = head_idx * head_dim + feature_idx;
          output(batch_idx, seq_idx, embed_idx) = input(batch_idx, head_idx, seq_idx, feature_idx);
        }
      }
    `;

    const concatHeadsNode = new CudaNode(deviceCode, "concat_heads_forward")
      .addInput("input", [-1, this.numHeads, -1, headDim], "float32")
      .addOutput("output", [-1, -1, this.embedDim], "float32")
      .setShapeResolver((inputs) => {
        const inputShape = inputs.get("input")!.shape;
        const [batchSize, , seqLen] = inputShape;
        return new Map([
          ["output", { shape: [batchSize, seqLen, this.embedDim] }],
        ]);
      });

    graph.addNode(concatHeadsNode);
    return concatHeadsNode;
  }
}
