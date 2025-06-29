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
       */
      __global__ void relu_forward(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = 1;
        for (int i = 0; i < input.dims; ++i) {
          size *= input.shape[i];
        }
        if (idx < size) {
          output(idx) = fmaxf(0.0f, input(idx));
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
 * This layer now creates a node that points to a single, reusable __global__ kernel.
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
    // This device code implements a generic, batched matrix-vector multiplication.
    // It's designed to be a __global__ kernel, not a __device__ function.
    const deviceCode = `
      /**
       * @cuda global
       */
      __global__ void dense_forward(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights, 
        Tensor<float> bias
      ) {
        // Each thread computes one output element.
        // Grid: (output_features / threads_per_block, batch_size)
        // Block: (threads_per_block)
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < output.shape[1]) {
          float sum = 0.0f;
          for (int k = 0; k < input.shape[1]; ++k) {
            sum += input(batch_idx, k) * weights(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias(output_feature_idx);
        }
      }
    `;

    const denseNode = new CudaNode(deviceCode, `dense_forward`)
      .addInput('input', [-1, this.inputFeatures], 'float32')
      .addOutput('output', [-1, this.outputFeatures], 'float32')
      // Use standardized parameter names for the reusable kernel
      .addParameter('weights', this.weights)
      .addParameter('bias', this.bias)
      .setShapeResolver(inputs => {
        const inputShape = inputs.get('input')!.shape;
        const batchSize = inputShape[0]; // Preserve batch size
        return new Map([['output', { shape: [batchSize, this.outputFeatures] }]]);
      });

    graph.addNode(denseNode);
    return denseNode;
  }
}
