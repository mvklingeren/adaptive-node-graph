// ============================================================================
// Transformer Block
// ============================================================================

import { CudaNode, CudaGraph } from "../cuda-graph.js";
import { CudaRuntime, CudaTensor } from "../cuda-abstractions.js";
import { Layer, NeuralGraph, DenseLayer, ReLULayer } from "../neural-network.js";
import {
  MultiHeadAttentionLayer,
  LayerNormLayer,
  AddLayer,
} from "./llm-layers.js";

// ============================================================================
// Transformer Block
// ============================================================================

export class TransformerBlock implements Layer {
  private attention: MultiHeadAttentionLayer;
  private norm1: LayerNormLayer;
  private norm2: LayerNormLayer;
  private ffn1: DenseLayer;
  private relu: ReLULayer;
  private ffn2: DenseLayer;
  private add1: AddLayer;
  private add2: AddLayer;

  constructor(
    private runtime: CudaRuntime,
    private embedDim: number,
    private numHeads: number,
    private ffnHiddenDim: number
  ) {
    this.attention = new MultiHeadAttentionLayer(runtime, embedDim, numHeads);
    this.norm1 = new LayerNormLayer(runtime, [embedDim]);
    this.norm2 = new LayerNormLayer(runtime, [embedDim]);
    this.add1 = new AddLayer();
    this.add2 = new AddLayer();
    this.ffn1 = new DenseLayer(runtime, embedDim, ffnHiddenDim);
    this.relu = new ReLULayer();
    this.ffn2 = new DenseLayer(runtime, ffnHiddenDim, embedDim);
  }

  async initialize(): Promise<void> {
    const layers = [
      this.attention,
      this.norm1,
      this.norm2,
      this.ffn1,
      this.ffn2,
    ];
    for (const layer of layers) {
      if (layer.initialize) {
        await layer.initialize();
      }
    }
  }

  addToGraph(graph: NeuralGraph, ...inputs: CudaNode[]): CudaNode {
    const inputNode = inputs[0];

    // 1. Multi-Head Attention
    const attentionNode = graph.addLayer(this.attention, inputNode);

    // 2. Add & Norm
    const add1Node = graph.addLayer(this.add1, inputNode, attentionNode);
    graph.connect(inputNode, "output", add1Node, "a");
    graph.connect(attentionNode, "output", add1Node, "b");
    const norm1Node = graph.addLayer(this.norm1, add1Node);

    // 3. Feed Forward Network
    const ffn1Node = graph.addLayer(this.ffn1, norm1Node);
    const reluNode = graph.addLayer(this.relu, ffn1Node);
    const ffn2Node = graph.addLayer(this.ffn2, reluNode);

    // 4. Add & Norm
    const add2Node = graph.addLayer(this.add2, norm1Node, ffn2Node);
    graph.connect(norm1Node, "output", add2Node, "a");
    graph.connect(ffn2Node, "output", add2Node, "b");
    const norm2Node = graph.addLayer(this.norm2, add2Node);

    return norm2Node;
  }
}
