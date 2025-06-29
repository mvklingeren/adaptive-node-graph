// ============================================================================
// Foundational Layers for Transformer Models
// ============================================================================
// This file contains the building blocks for creating transformer-based
// neural networks, such as Layer Normalization and Softmax.
// ============================================================================

import { CudaNode, CudaGraph } from "../cuda-graph";
import { CudaRuntime, CudaTensor } from "../cuda-abstractions";
import { Layer, NeuralGraph, DenseLayer } from "../neural-network";
import {
  ScaledDotProductAttention,
  SplitHeads,
  ConcatHeads,
} from "./attention";

// ============================================================================
// Layer Normalization Layer
// ============================================================================

export class LayerNormLayer implements Layer {
  private gamma!: CudaTensor;
  private beta!: CudaTensor;

  constructor(
    private runtime: CudaRuntime,
    private normalizedShape: number[],
    private epsilon: number = 1e-5
  ) {}

  async initialize(): Promise<void> {
    const shapeSize = this.normalizedShape.reduce((a, b) => a * b, 1);
    this.gamma = await this.runtime.malloc(shapeSize * 4, this.normalizedShape, "float32");
    this.beta = await this.runtime.malloc(shapeSize * 4, this.normalizedShape, "float32");
    // TODO: Initialize gamma to 1s and beta to 0s.
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const deviceCode = `
      /**
       * @cuda global
       */
      __global__ void layer_norm_forward(
        Tensor<float> output,
        Tensor<float> input,
        Tensor<float> gamma,
        Tensor<float> beta
      ) {
        // This kernel processes one feature vector (e.g., one token's embedding) per block.
        // Grid: (batch_size, seq_len)
        // Block: (feature_count)
        extern __shared__ float shared_mem[];
        int batch_idx = blockIdx.y;
        int seq_idx = blockIdx.x;
        int feature_count = input.shape[2];
        int tid = threadIdx.x;

        // Step 1: Calculate mean
        float sum = 0.0f;
        for (int i = tid; i < feature_count; i += blockDim.x) {
            sum += input(batch_idx, seq_idx, i);
        }
        shared_mem[tid] = sum;
        __syncthreads();

        // Parallel reduction for mean
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_mem[tid] += shared_mem[tid + s];
            }
            __syncthreads();
        }
        float mean = shared_mem[0] / feature_count;

        // Step 2: Calculate variance
        sum = 0.0f;
        for (int i = tid; i < feature_count; i += blockDim.x) {
            float dev = input(batch_idx, seq_idx, i) - mean;
            sum += dev * dev;
        }
        shared_mem[tid] = sum;
        __syncthreads();
        
        // Parallel reduction for variance
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_mem[tid] += shared_mem[tid + s];
            }
            __syncthreads();
        }
        float variance = shared_mem[0] / feature_count;
        float rsqrt_variance = rsqrtf(variance + ${this.epsilon});

        // Step 3: Normalize
        for (int i = tid; i < feature_count; i += blockDim.x) {
            float normalized = (input(batch_idx, seq_idx, i) - mean) * rsqrt_variance;
            output(batch_idx, seq_idx, i) = normalized * gamma(i) + beta(i);
        }
      }
    `;

    const layerNormNode = new CudaNode(deviceCode, "layer_norm_forward")
      .addInput("input", [-1, ...this.normalizedShape], "float32")
      .addOutput("output", [-1, ...this.normalizedShape], "float32")
      .addParameter("gamma", this.gamma)
      .addParameter("beta", this.beta)
      .setShapeResolver((inputs) => {
        const inputShape = inputs.get("input")!.shape;
        return new Map([["output", { shape: inputShape }]]);
      });

    graph.addNode(layerNormNode);
    return layerNormNode;
  }
}

// ============================================================================
// Softmax Layer
// ============================================================================

export class SoftmaxLayer implements Layer {
  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const deviceCode = `
      /**
       * @cuda global
       */
      __global__ void softmax_forward(Tensor<float> output, Tensor<float> input) {
        // This kernel computes softmax over the last dimension.
        // It handles both 2D tensors [batch, features] and 4D tensors [batch, heads, seq, seq]
        extern __shared__ float shared_mem[];
        int tid = threadIdx.x;
        
        if (input.dims == 2) {
          // Handle 2D case: [batch, features]
          int batch_idx = blockIdx.x;
          int size = input.shape[1];
          
          // 1. Find max for numerical stability
          float max_val = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              max_val = fmaxf(max_val, input(batch_idx, i));
          }
          shared_mem[tid] = max_val;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]); }
              __syncthreads();
          }
          max_val = shared_mem[0];

          // 2. Calculate sum of exps
          float sum_exp = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              sum_exp += expf(input(batch_idx, i) - max_val);
          }
          shared_mem[tid] = sum_exp;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] += shared_mem[tid + s]; }
              __syncthreads();
          }
          sum_exp = shared_mem[0];

          // 3. Calculate softmax
          for (int i = tid; i < size; i += blockDim.x) {
              output(batch_idx, i) = expf(input(batch_idx, i) - max_val) / sum_exp;
          }
        } else {
          // Handle 4D case: [batch, heads, seq, seq]
          int batch_idx = blockIdx.z;
          int head_idx = blockIdx.y;
          int row_idx = blockIdx.x;
          int size = input.shape[3];

          // 1. Find max for numerical stability
          float max_val = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              max_val = fmaxf(max_val, input(batch_idx, head_idx, row_idx, i));
          }
          shared_mem[tid] = max_val;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]); }
              __syncthreads();
          }
          max_val = shared_mem[0];

          // 2. Calculate sum of exps
          float sum_exp = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              sum_exp += expf(input(batch_idx, head_idx, row_idx, i) - max_val);
          }
          shared_mem[tid] = sum_exp;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] += shared_mem[tid + s]; }
              __syncthreads();
          }
          sum_exp = shared_mem[0];

          // 3. Calculate softmax
          for (int i = tid; i < size; i += blockDim.x) {
              output(batch_idx, head_idx, row_idx, i) = expf(input(batch_idx, head_idx, row_idx, i) - max_val) / sum_exp;
          }
        }
      }
    `;

    const softmaxNode = new CudaNode(deviceCode, "softmax_forward")
      .addInput("input", [-1, -1], "float32") // Start with 2D, will be resolved dynamically
      .addOutput("output", [-1, -1], "float32")
      .setShapeResolver((inputs) => {
        const inputShape = inputs.get("input")!.shape;
        return new Map([["output", { shape: inputShape }]]);
      });

    graph.addNode(softmaxNode);
    return softmaxNode;
  }
}

// ============================================================================
// Multi-Head Attention Layer
// ============================================================================

export class MultiHeadAttentionLayer implements Layer {
  private static instanceCounter = 0;
  private readonly id: number;

  private Wq!: DenseLayer;
  private Wk!: DenseLayer;
  private Wv!: DenseLayer;
  private Wo!: DenseLayer;

  private splitHeadsQ!: SplitHeads;
  private splitHeadsK!: SplitHeads;
  private splitHeadsV!: SplitHeads;
  private concatHeads!: ConcatHeads;
  private attention!: ScaledDotProductAttention;

  constructor(
    private runtime: CudaRuntime,
    private embedDim: number,
    private numHeads: number
  ) {
    this.id = MultiHeadAttentionLayer.instanceCounter++;
    if (this.embedDim % this.numHeads !== 0) {
      throw new Error("embedDim must be divisible by numHeads");
    }

    // Initialize the projection layers
    this.Wq = new DenseLayer(runtime, embedDim, embedDim);
    this.Wk = new DenseLayer(runtime, embedDim, embedDim);
    this.Wv = new DenseLayer(runtime, embedDim, embedDim);
    this.Wo = new DenseLayer(runtime, embedDim, embedDim);

    // Initialize attention utility layers
    this.splitHeadsQ = new SplitHeads(embedDim, numHeads);
    this.splitHeadsK = new SplitHeads(embedDim, numHeads);
    this.splitHeadsV = new SplitHeads(embedDim, numHeads);
    this.concatHeads = new ConcatHeads(embedDim, numHeads);
    this.attention = new ScaledDotProductAttention(runtime, embedDim, numHeads);
  }

  async initialize(): Promise<void> {
    const layers = [this.Wq, this.Wk, this.Wv, this.Wo];
    for (const layer of layers) {
      if (layer.initialize) {
        await layer.initialize();
      }
    }
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const inputNode = inputs[0];

    // 1. Project Q, K, V
    const qProjNode = graph.addLayer(this.Wq, inputNode);
    const kProjNode = graph.addLayer(this.Wk, inputNode);
    const vProjNode = graph.addLayer(this.Wv, inputNode);

    // 2. Split heads
    const qSplitNode = graph.addLayer(this.splitHeadsQ, qProjNode);
    const kSplitNode = graph.addLayer(this.splitHeadsK, kProjNode);
    const vSplitNode = graph.addLayer(this.splitHeadsV, vProjNode);

    // 3. Scaled dot-product attention
    const attentionNode = this.attention.addToGraph(
      graph,
      qSplitNode,
      kSplitNode,
      vSplitNode
    );

    // 4. Concatenate heads
    const concatNode = graph.addLayer(this.concatHeads, attentionNode);

    // 5. Final output projection
    const outputNode = graph.addLayer(this.Wo, concatNode);

    return outputNode;
  }
}

// ============================================================================
// Element-wise Add Layer
// ============================================================================

export class AddLayer implements Layer {
  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const deviceCode = `
      /**
       * @cuda global
       */
      __global__ void add_forward(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // This is a simplified approach. A robust implementation would handle
        // arbitrary dimensions and calculate total size on the host.
        int size = a.shape[0] * a.shape[1] * a.shape[2];

        for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
            // This assumes flattened indexing. A better way is to reconstruct multidim index.
            // For now, we'll assume the shapes are identical and can be treated as flat arrays.
            output.data[i] = a.data[i] + b.data[i];
        }
      }
    `;

    const addNode = new CudaNode(deviceCode, "add_forward")
      .addInput("a", [-1, -1, -1], "float32")
      .addInput("b", [-1, -1, -1], "float32")
      .addOutput("output", [-1, -1, -1], "float32")
      .setShapeResolver((inputs) => {
        // Output shape is the same as the input shapes (assuming they match)
        const aShape = inputs.get("a")!.shape;
        return new Map([["output", { shape: aShape }]]);
      });

    graph.addNode(addNode);
    if (inputs.length > 1) {
      graph.connect(inputs[0], "output", addNode, "a");
      graph.connect(inputs[1], "output", addNode, "b");
    }
    return addNode;
  }
}
