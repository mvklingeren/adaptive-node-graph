// ============================================================================
// Embedding and Positional Encoding
// ============================================================================

import { CudaNode, CudaGraph } from "../cuda-graph";
import { CudaRuntime, CudaTensor } from "../cuda-abstractions";
import { Layer, NeuralGraph } from "../neural-network";

// ============================================================================
// Embedding Layer
// ============================================================================

export class EmbeddingLayer implements Layer {
  private embeddings!: CudaTensor;

  constructor(
    private runtime: CudaRuntime,
    private vocabSize: number,
    private embedDim: number,
    private maxLen: number
  ) {}

  async initialize(): Promise<void> {
    this.embeddings = await this.runtime.malloc(
      this.vocabSize * this.embedDim * 4,
      [this.vocabSize, this.embedDim],
      "float32"
    );
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const deviceCode = `
      /**
       * @cuda global
       */
      __global__ void embedding_forward(Tensor<float> output, Tensor<int> input, Tensor<float> embeddings) {
        int batch_idx = blockIdx.y;
        int seq_idx = blockIdx.x;

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1]) {
          int token_id = input(batch_idx, seq_idx);
          for (int i = threadIdx.x; i < output.shape[2]; i += blockDim.x) {
            output(batch_idx, seq_idx, i) = embeddings(token_id, i);
          }
        }
      }
    `;

    const embeddingNode = new CudaNode(deviceCode, "embedding_forward")
      .addInput("input", [-1, this.maxLen], "int32")
      .addOutput("output", [-1, this.maxLen, this.embedDim], "float32")
      .addParameter("embeddings", this.embeddings)
      .setShapeResolver((inputs) => {
        const inputShape = inputs.get("input")!.shape;
        const batchSize = inputShape[0];
        // Only resolve if we have a concrete batch size
        if (batchSize === -1) {
          // Return empty map if batch size is still dynamic
          return new Map();
        }
        // Resolve output shape when we have a concrete batch size
        return new Map([
          ["output", { shape: [batchSize, this.maxLen, this.embedDim] }],
        ]);
      });

    graph.addNode(embeddingNode);
    // Connection is now handled by the graph builder (e.g., LanguageModel)
    return embeddingNode;
  }
}

// ============================================================================
// Positional Encoding Layer
// ============================================================================

export class PositionalEncodingLayer implements Layer {
  private frequencies!: CudaTensor;

  constructor(
    private runtime: CudaRuntime,
    private maxLen: number, 
    private embedDim: number
  ) {}

  async initialize(): Promise<void> {
    // Pre-compute frequency values for better accuracy
    // frequencies[i] = 1.0 / pow(10000.0, (2.0 * i) / embed_dim)
    const numFreqs = Math.floor(this.embedDim / 2);
    this.frequencies = await this.runtime.malloc(
      numFreqs * 4, // float32 = 4 bytes
      [numFreqs],
      "float32"
    );

    // Calculate and upload frequency values
    const freqData = new Float32Array(numFreqs);
    for (let i = 0; i < numFreqs; i++) {
      freqData[i] = 1.0 / Math.pow(10000.0, (2.0 * i) / this.embedDim);
    }

    // Copy pre-computed frequencies to GPU
    const freqBuffer = Buffer.from(freqData.buffer);
    await this.runtime.memcpyHostToDevice(this.frequencies, freqBuffer);
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const deviceCode = `
      /**
       * @cuda global
       */
      __global__ void positional_encoding_forward(Tensor<float> output, Tensor<float> input, Tensor<float> frequencies) {
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1] && embed_idx < input.shape[2]) {
          float pos = (float)seq_idx;
          int freq_idx = embed_idx / 2;  // Integer division to get frequency index
          
          // Use pre-computed frequency values for better accuracy
          // This eliminates floating-point accumulation errors from powf() calculations
          float freq = frequencies(freq_idx);
          float val;
          if (embed_idx % 2 == 0) {
            val = sinf(pos * freq);
          } else {
            val = cosf(pos * freq);
          }
          output(batch_idx, seq_idx, embed_idx) = input(batch_idx, seq_idx, embed_idx) + val;
        }
      }
    `;

    const posEncodingNode = new CudaNode(
      deviceCode,
      "positional_encoding_forward"
    )
      .addInput("input", [-1, -1, this.embedDim], "float32")
      .addOutput("output", [-1, -1, this.embedDim], "float32")
      .addParameter("frequencies", this.frequencies)
      .setShapeResolver((inputs) => {
        const inputShape = inputs.get("input")!.shape;
        return new Map([["output", { shape: inputShape }]]);
      });

    graph.addNode(posEncodingNode);
    return posEncodingNode;
  }
}
