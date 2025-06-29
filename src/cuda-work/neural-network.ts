// ============================================================================
// Neural Network Layer
// ============================================================================
// This file provides a high-level API for building neural networks using the
// universal CudaGraph engine. It abstracts the complexities of node and
// parameter management into easy-to-use layer classes.
// ============================================================================

import { CudaNode, CudaGraph } from "./cuda-graph";
import { CudaRuntime, CudaTensor } from "./cuda-abstractions";

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

        if (batch_idx < input.shape[0] && output_feature_idx < output.shape[1]) {
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
