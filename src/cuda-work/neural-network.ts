// ============================================================================
// Neural Network Layer
// ============================================================================
// This file provides a high-level API for building neural networks using the
// universal CudaGraph engine. It abstracts the complexities of node and
// parameter management into easy-to-use layer classes.
// ============================================================================

import { CudaNode, CudaGraph } from "./cuda-graph";
import { CudaRuntime, CudaTensor, CublasRuntime, CublasHandle, CudaStream } from "./cuda-abstractions";

// ============================================================================
// NeuralGraph: A Specialized CudaGraph for Neural Networks
// ============================================================================

export class NeuralGraph extends CudaGraph {
  private lastNode: CudaNode | null = null;

  constructor(name: string = "UntitledNeuralGraph") {
    super(name);
  }

  /**
   * Adds a neural network layer to the graph.
   * The layer is automatically connected to the previous layer, assuming
   * a standard 'output' to 'input' connection.
   * @param layer - The layer to add.
   */
  addLayer(layer: Layer, ...inputs: CudaNode[]): CudaNode {
    if (inputs.length === 0 && this.lastNode) {
      inputs.push(this.lastNode);
    }
    
    const currentNode = layer.addToGraph(this, ...inputs);

    // The connection logic is now handled by the individual layers
    // or by the specific graph-building logic (like in LanguageModel).
    if (inputs.length > 0 && layer.constructor.name !== 'AddLayer' && layer.constructor.name !== 'BatchedMatMul') {
        for(const input of inputs) {
            this.connect(input, 'output', currentNode, 'input');
        }
    }

    this.lastNode = currentNode;
    return currentNode;
  }
}

// ============================================================================
// Abstract Layer Definition
// ============================================================================

/**
 * Defines the interface for a neural network layer.
 * A layer is a factory that adds its specific CudaNode(s) to a graph.
 */
export interface Layer {
  /**
   * Adds the layer's logic (as one or more CudaNodes) to the provided graph.
   * @param graph - The NeuralGraph to add the layer to.
   * @param inputs - The list of input nodes for this layer.
   * @returns The final CudaNode of the layer, to be connected to the next layer.
   */
  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode;
}

// ============================================================================
// Concrete Layer Implementations
// ============================================================================

/**
 * A stateless ReLU activation layer.
 * f(x) = max(0, x)
 */
export class ReLULayer implements Layer {
  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const deviceCode = `
      /**
       * @cuda global
       * Optimized ReLU kernel that calculates size dynamically
       */
      __global__ void relu_forward(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = input.total_elements();
        if (idx < total_elements) {
          output.data[idx] = fmaxf(0.0f, input.data[idx]);
        }
      }
    `;
    const reluNode = new CudaNode(deviceCode, "relu_forward")
      .addInput('input', [-1, -1], 'float32') // Dynamic input shape
      .addOutput('output', [-1, -1], 'float32') // Dynamic output shape
      .setShapeResolver(inputs => {
        // A ReLU operation's output shape is always the same as its input shape.
        const inputShape = inputs.get('input')!.shape;
        return new Map([['output', { shape: inputShape }]]);
      });
      
    graph.addNode(reluNode);
    return reluNode;
  }
}

/**
 * A fully connected (dense) layer.
 * f(x) = Wx + b
 * This layer is now context-aware and generates the correct kernel
 * based on the input tensor's dimensionality (2D or 3D).
 */
export class DenseLayer implements Layer {
  private weights!: CudaTensor;
  private bias!: CudaTensor;

  constructor(
    private runtime: CudaRuntime,
    public readonly inputFeatures: number,
    public readonly outputFeatures: number
  ) {}

  async initialize(): Promise<void> {
      const weightShape = [this.inputFeatures, this.outputFeatures];
      const biasShape = [this.outputFeatures];
      this.weights = await this.runtime.malloc(this.inputFeatures * this.outputFeatures * 4, weightShape, "float32");
      this.bias = await this.runtime.malloc(this.outputFeatures * 4, biasShape, "float32");
      // In a real implementation, you would initialize the tensors with random data.
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const inputNode = inputs[0];
    if (!inputNode) {
      throw new Error("DenseLayer requires at least one input node.");
    }
    
    // This is a simplification. In a real graph, we'd need a more robust
    // way to get the output port that connects to this layer.
    const inputShape = inputNode.outputs.get('output')!.shape;
    const is3D = inputShape.length === 3;

    let deviceCode: string;
    let functionName: string;
    let denseNode: CudaNode;

    if (is3D) {
      // --- 3D Kernel for [batch, seq, features] tensors ---
      functionName = 'dense_forward_3d';
      deviceCode = `
      /**
       * @cuda global
       * Performs a dense layer transformation on a 3D tensor.
       * Input: [batch, seq_len, input_features]
       * Output: [batch, seq_len, output_features]
       */
      __global__ void ${functionName}(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights, 
        Tensor<float> bias
      ) {
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < output.shape[0] && seq_idx < output.shape[1] && output_feature_idx < output.shape[2]) {
          float sum = 0.0f;
          for (int k = 0; k < input.shape[2]; ++k) { // Iterate over input_features
            sum += input(batch_idx, seq_idx, k) * weights(k, output_feature_idx);
          }
          output(batch_idx, seq_idx, output_feature_idx) = sum + bias(output_feature_idx);
        }
      }
      `;
      
      denseNode = new CudaNode(deviceCode, functionName)
        .addInput('input', [-1, -1, this.inputFeatures], 'float32')
        .addOutput('output', [-1, -1, this.outputFeatures], 'float32')
        .addParameter('weights', this.weights)
        .addParameter('bias', this.bias)
        .setShapeResolver(inputs => {
          const inputShape = inputs.get('input')!.shape;
          const [batchSize, seqLen] = inputShape;
          return new Map([['output', { shape: [batchSize, seqLen, this.outputFeatures] }]]);
        });

    } else {
      // --- Original 2D Kernel for [batch, features] tensors ---
      functionName = 'dense_forward_2d';
      deviceCode = `
      /**
       * @cuda global
       * Performs a dense layer transformation on a 2D tensor.
       * Input: [batch, input_features]
       * Output: [batch, output_features]
       */
      __global__ void ${functionName}(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights, 
        Tensor<float> bias
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < weights.shape[1]) {
          float sum = 0.0f;
          for (int k = 0; k < input.shape[1]; ++k) { // Iterate over input_features
            sum += input(batch_idx, k) * weights(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias(output_feature_idx);
        }
      }
      `;
      
      denseNode = new CudaNode(deviceCode, functionName)
        .addInput('input', [-1, this.inputFeatures], 'float32')
        .addOutput('output', [-1, this.outputFeatures], 'float32')
        .addParameter('weights', this.weights)
        .addParameter('bias', this.bias)
        .setShapeResolver(inputs => {
          const inputShape = inputs.get('input')!.shape;
          const batchSize = inputShape[0];
          return new Map([['output', { shape: [batchSize, this.outputFeatures] }]]);
        });
    }
    
    // Rename the node for clarity in the compiled graph
    denseNode.name = functionName;
    graph.addNode(denseNode);
    return denseNode;
  }
}

// ============================================================================
// cuBLAS-Optimized Layer Implementations
// ============================================================================

/**
 * A cuBLAS-optimized fully connected (dense) layer.
 * Uses cuBLAS GEMM operations instead of custom CUDA kernels for better performance.
 * f(x) = Wx + b
 */
export class CublasDenseLayer implements Layer {
  private weights!: CudaTensor;
  private bias!: CudaTensor;

  constructor(
    private runtime: CublasRuntime,
    private cublasHandle: CublasHandle,
    private stream: CudaStream,
    public readonly inputFeatures: number,
    public readonly outputFeatures: number
  ) {}

  async initialize(): Promise<void> {
    const weightShape = [this.inputFeatures, this.outputFeatures];
    const biasShape = [this.outputFeatures];
    this.weights = await this.runtime.malloc(this.inputFeatures * this.outputFeatures * 4, weightShape, "float32");
    this.bias = await this.runtime.malloc(this.outputFeatures * 4, biasShape, "float32");
    // In a real implementation, you would initialize the tensors with random data.
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const inputNode = inputs[0];
    if (!inputNode) {
      throw new Error("CublasDenseLayer requires at least one input node.");
    }

    const inputShape = inputNode.outputs.get('output')!.shape;
    const is3D = inputShape.length === 3;

    let functionName: string;
    let denseNode: CudaNode;

    if (is3D) {
      // --- cuBLAS 3D Dense Layer for [batch, seq, features] tensors ---
      functionName = 'cublas_dense_forward_3d';
      
      // Create a special node that represents cuBLAS operation
      denseNode = new CudaNode("", functionName) // Empty device code - will be handled by compiler
        .addInput('input', [-1, -1, this.inputFeatures], 'float32')
        .addOutput('output', [-1, -1, this.outputFeatures], 'float32')
        .addParameter('weights', this.weights)
        .addParameter('bias', this.bias)
        .setShapeResolver(inputs => {
          const inputShape = inputs.get('input')!.shape;
          const [batchSize, seqLen] = inputShape;
          return new Map([['output', { shape: [batchSize, seqLen, this.outputFeatures] }]]);
        });

      // Add metadata for cuBLAS operation
      (denseNode as any).cublasOperation = {
        type: 'gemm',
        transa: 'N',
        transb: 'N',
        alpha: 1.0,
        beta: 1.0, // For bias addition
        handle: this.cublasHandle,
        stream: this.stream
      };

    } else {
      // --- cuBLAS 2D Dense Layer for [batch, features] tensors ---
      functionName = 'cublas_dense_forward_2d';
      
      denseNode = new CudaNode("", functionName)
        .addInput('input', [-1, this.inputFeatures], 'float32')
        .addOutput('output', [-1, this.outputFeatures], 'float32')
        .addParameter('weights', this.weights)
        .addParameter('bias', this.bias)
        .setShapeResolver(inputs => {
          const inputShape = inputs.get('input')!.shape;
          const batchSize = inputShape[0];
          return new Map([['output', { shape: [batchSize, this.outputFeatures] }]]);
        });

      // Add metadata for cuBLAS operation
      (denseNode as any).cublasOperation = {
        type: 'gemm',
        transa: 'N',
        transb: 'N',
        alpha: 1.0,
        beta: 1.0, // For bias addition
        handle: this.cublasHandle,
        stream: this.stream
      };
    }

    denseNode.name = functionName;
    graph.addNode(denseNode);
    return denseNode;
  }
}

/**
 * A cuBLAS-optimized batched matrix multiplication layer.
 * Uses cuBLAS batched GEMM operations for attention mechanisms.
 */
export class CublasBatchedMatMul implements Layer {
  private functionName: string;

  constructor(
    private runtime: CublasRuntime,
    private cublasHandle: CublasHandle,
    private stream: CudaStream,
    private transposeB: boolean = false
  ) {
    this.functionName = this.transposeB ? "cublas_batched_matmul_transpose_b" : "cublas_batched_matmul";
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    if (inputs.length < 2) {
      throw new Error("CublasBatchedMatMul requires at least two input nodes.");
    }

    // Create a special node that represents cuBLAS batched operation
    const matmulNode = new CudaNode("", this.functionName) // Empty device code - will be handled by compiler
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

    // Add metadata for cuBLAS operation
    (matmulNode as any).cublasOperation = {
      type: 'gemmStridedBatched',
      transa: 'N',
      transb: this.transposeB ? 'T' : 'N',
      alpha: 1.0,
      beta: 0.0,
      handle: this.cublasHandle,
      stream: this.stream
    };

    graph.addNode(matmulNode);
    
    // Connect inputs manually since we need specific port names
    if (inputs.length > 1) {
      graph.connect(inputs[0], 'output', matmulNode, 'a');
      graph.connect(inputs[1], 'output', matmulNode, 'b');
    }
    
    return matmulNode;
  }
}

/**
 * Enhanced NeuralGraph with cuBLAS support and stream management.
 */
export class CublasNeuralGraph extends NeuralGraph {
  private streams: CudaStream[] = [];
  private cublasHandles: CublasHandle[] = [];

  constructor(
    name: string = "UntitledCublasNeuralGraph",
    private runtime: CublasRuntime,
    private numStreams: number = 2
  ) {
    super(name);
  }

  /**
   * Initializes the cuBLAS handles and CUDA streams for this graph.
   */
  async initialize(): Promise<void> {
    // Create CUDA streams
    for (let i = 0; i < this.numStreams; i++) {
      const stream = await this.runtime.createStream();
      this.streams.push(stream);
    }

    // Create cuBLAS handles and associate them with streams
    for (let i = 0; i < this.numStreams; i++) {
      const handle = await this.runtime.createCublasHandle();
      await handle.setStream(this.streams[i]);
      this.cublasHandles.push(handle);
    }

    console.log(`[CublasNeuralGraph] Initialized ${this.numStreams} streams and cuBLAS handles`);
  }

  /**
   * Gets a cuBLAS handle for the specified stream index.
   */
  getCublasHandle(streamIndex: number = 0): CublasHandle {
    if (streamIndex >= this.cublasHandles.length) {
      throw new Error(`Stream index ${streamIndex} out of range. Available streams: ${this.cublasHandles.length}`);
    }
    return this.cublasHandles[streamIndex];
  }

  /**
   * Gets a CUDA stream for the specified index.
   */
  getStream(streamIndex: number = 0): CudaStream {
    if (streamIndex >= this.streams.length) {
      throw new Error(`Stream index ${streamIndex} out of range. Available streams: ${this.streams.length}`);
    }
    return this.streams[streamIndex];
  }

  /**
   * Adds a cuBLAS-optimized layer to the graph with automatic stream assignment.
   */
  addCublasLayer(
    layerFactory: (handle: CublasHandle, stream: CudaStream) => Layer,
    streamIndex: number = 0,
    ...inputs: CudaNode[]
  ): CudaNode {
    const handle = this.getCublasHandle(streamIndex);
    const stream = this.getStream(streamIndex);
    const layer = layerFactory(handle, stream);
    
    return this.addLayer(layer, ...inputs);
  }

  /**
   * Cleans up resources when the graph is no longer needed.
   */
  async cleanup(): Promise<void> {
    // Destroy cuBLAS handles
    for (const handle of this.cublasHandles) {
      await this.runtime.destroyCublasHandle(handle);
    }

    // Destroy CUDA streams
    for (const stream of this.streams) {
      await this.runtime.destroyStream(stream);
    }

    this.cublasHandles.length = 0;
    this.streams.length = 0;

    console.log(`[CublasNeuralGraph] Cleaned up all resources`);
  }
}
