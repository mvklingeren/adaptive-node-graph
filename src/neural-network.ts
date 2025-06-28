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
  private lastLayer: CudaNode | null = null;

  constructor(name: string = "UntitledNeuralGraph") {
    super(name);
  }

  /**
   * Adds a neural network layer to the graph.
   * The layer is automatically connected to the previous layer.
   * @param layer - The layer to add.
   */
  addLayer(layer: Layer): this {
    // A layer is responsible for creating and adding its own CudaNode(s)
    // to the graph, and returning the final output node of the layer.
    const outputNode = layer.addToGraph(this);

    if (this.lastLayer) {
      this.connect(this.lastLayer, outputNode);
    }

    this.lastLayer = outputNode;
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
  private static instanceCounter = 0;
  private readonly id: number;

  constructor() {
    this.id = ReLULayer.instanceCounter++;
  }

  addToGraph(graph: NeuralGraph): CudaNode {
    const deviceCode = `
      __device__ float relu_forward(float x) {
        return fmaxf(0.0f, x);
      }
    `;
    const reluNode = new CudaNode(
      deviceCode,
      "relu_forward",
      `relu_out_${this.id}`, // Unique output variable name
      ["x"] // Input variable name
    );
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
  private weights: CudaTensor;
  private bias: CudaTensor;

  constructor(
    private runtime: CudaRuntime,
    public readonly inputFeatures: number,
    public readonly outputFeatures: number
  ) {
    this.id = DenseLayer.instanceCounter++;
    // In a real scenario, you would initialize these tensors with random values.
    // For now, we just allocate them.
    this.weights = {} as CudaTensor; // Placeholder
    this.bias = {} as CudaTensor; // Placeholder
  }

  async initialize(): Promise<void> {
      // Allocate memory for weights and bias on the GPU
      this.weights = await this.runtime.malloc(this.inputFeatures * this.outputFeatures * 4); // 4 bytes per float
      this.bias = await this.runtime.malloc(this.outputFeatures * 4);
      // Here you would typically initialize the tensors with random data.
  }

  addToGraph(graph: NeuralGraph): CudaNode {
    const weightParamName = `weights_${this.id}`;
    const biasParamName = `bias_${this.id}`;

    // This device code is a simplification for a single value.
    // A real implementation would perform a dot product.
    const deviceCode = `
      __device__ float dense_forward_${this.id}(float input, const float* ${weightParamName}, const float* ${biasParamName}) {
        // Simple multiplication, not a dot product, for this example.
        return input * ${weightParamName}[0] + ${biasParamName}[0];
      }
    `;

    const denseNode = new CudaNode(
      deviceCode,
      `dense_forward_${this.id}`,
      `dense_out_${this.id}`,
      ["input"]
    );

    // Add the weight and bias tensors as parameters to the node with unique names.
    denseNode.addParameter(weightParamName, this.weights);
    denseNode.addParameter(biasParamName, this.bias);

    graph.addNode(denseNode);
    return denseNode;
  }
}
