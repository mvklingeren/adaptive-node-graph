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
  addLayer(layer: Layer): this {
    const currentNode = layer.addToGraph(this);

    if (this.lastNode) {
      // Connect the default output of the last node to the default input of the current node.
      this.connect(this.lastNode, 'output', currentNode, 'input');
    }

    this.lastNode = currentNode;
    return this;
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
   * @returns The final CudaNode of the layer, to be connected to the next layer.
   */
  addToGraph(graph: NeuralGraph): CudaNode;
}

// ============================================================================
// Concrete Layer Implementations
// ============================================================================

/**
 * A stateless ReLU activation layer.
 * f(x) = max(0, x)
 */
export class ReLULayer implements Layer {
  addToGraph(graph: NeuralGraph): CudaNode {
    const deviceCode = `
      __device__ void relu_forward(Tensor<float> output, Tensor<float> input) {
        int size = 1;
        for (int i = 0; i < input.dims; ++i) {
          size *= input.shape[i];
        }
        // This is a simplified loop for element-wise operation.
        // A real implementation would use the thread index (idx).
        for (int i = 0; i < size; ++i) {
            output.data[i] = fmaxf(0.0f, input.data[i]);
        }
      }
    `;
    const reluNode = new CudaNode(deviceCode, "relu_forward")
      .addInput('input', [-1, -1], 'float32') // Dynamic shape
      .addOutput('output', [-1, -1], 'float32');
      
    graph.addNode(reluNode);
    return reluNode;
  }
}

/**
 * A fully connected (dense) layer.
 * f(x) = Wx + b
 * Note: This implementation is simplified for a single neuron for clarity.
 * A real implementation would use matrix multiplication.
 */
export class DenseLayer implements Layer {
  private static instanceCounter = 0;
  private readonly id: number;
  private weights!: CudaTensor;
  private bias!: CudaTensor;

  constructor(
    private runtime: CudaRuntime,
    public readonly inputFeatures: number,
    public readonly outputFeatures: number
  ) {
    this.id = DenseLayer.instanceCounter++;
  }

  async initialize(): Promise<void> {
      const weightShape = [this.inputFeatures, this.outputFeatures];
      const biasShape = [this.outputFeatures];
      this.weights = await this.runtime.malloc(this.inputFeatures * this.outputFeatures * 4, weightShape, "float32");
      this.bias = await this.runtime.malloc(this.outputFeatures * 4, biasShape, "float32");
      // In a real implementation, you would initialize the tensors with random data.
  }

  addToGraph(graph: NeuralGraph): CudaNode {
    const weightParamName = `weights_${this.id}`;
    const biasParamName = `bias_${this.id}`;

    // This device code implements a matrix-vector multiplication.
    // It assumes the input is a vector (dims == 1) and weights is a matrix (dims == 2).
    const deviceCode = `
      __device__ void dense_forward_${this.id}(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> ${weightParamName}, 
        Tensor<float> ${biasParamName}
      ) {
        int input_size = input.shape[0];
        int output_size = output.shape[0];

        for (int i = 0; i < output_size; ++i) {
          float sum = 0.0f;
          for (int j = 0; j < input_size; ++j) {
            // W is row-major: W[i, j] is at index i * input_size + j
            sum += input.data[j] * ${weightParamName}.data[i * input_size + j];
          }
          output.data[i] = sum + ${biasParamName}.data[i];
        }
      }
    `;

    const denseNode = new CudaNode(deviceCode, `dense_forward_${this.id}`)
      .addInput('input', [this.inputFeatures], 'float32')
      .addOutput('output', [this.outputFeatures], 'float32')
      .addParameter(weightParamName, this.weights)
      .addParameter(biasParamName, this.bias);

    graph.addNode(denseNode);
    return denseNode;
  }
}
