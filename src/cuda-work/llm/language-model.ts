// ============================================================================
// Language Model
// ============================================================================

import { CudaRuntime } from "../cuda-abstractions";
import { NeuralGraph, DenseLayer, Layer } from "../neural-network";
import { TransformerBlock } from "./transformer";
import { EmbeddingLayer, PositionalEncodingLayer } from "./embedding";
import { SoftmaxLayer } from "./llm-layers";

// ============================================================================
// Language Model
// ============================================================================

export class LanguageModel extends NeuralGraph {
  private layers: (Layer & { initialize?: () => Promise<void> })[] = [];
  private runtime: CudaRuntime;
  private vocabSize: number;
  private embedDim: number;
  private numHeads: number;
  private numLayers: number;
  private ffnHiddenDim: number;
  private maxLen: number;

  constructor(
    runtime: CudaRuntime,
    vocabSize: number,
    embedDim: number,
    numHeads: number,
    numLayers: number,
    ffnHiddenDim: number,
    maxLen: number
  ) {
    super("LanguageModel");
    this.runtime = runtime;
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.numHeads = numHeads;
    this.numLayers = numLayers;
    this.ffnHiddenDim = ffnHiddenDim;
    this.maxLen = maxLen;

    // Create layer instances but do not build the graph yet
    this.layers.push(
      new EmbeddingLayer(
        this.runtime,
        this.vocabSize,
        this.embedDim,
        this.maxLen
      )
    );
    this.layers.push(new PositionalEncodingLayer(this.runtime, this.maxLen, this.embedDim));
    for (let i = 0; i < this.numLayers; ++i) {
      this.layers.push(
        new TransformerBlock(
          this.runtime,
          this.embedDim,
          this.numHeads,
          this.ffnHiddenDim
        )
      );
    }
    this.layers.push(
      new DenseLayer(this.runtime, this.embedDim, this.vocabSize)
    );
    this.layers.push(new SoftmaxLayer());
  }

  async initialize(): Promise<void> {
    for (const layer of this.layers) {
      if (layer.initialize) {
        await layer.initialize();
      }
    }
  }

  build(): void {
    // The graph's input is implicitly the input to the first layer.
    // The CudaGraph logic will identify any unconnected CudaNode inputs as graph inputs.
    const embeddingNode = this.addLayer(this.layers[0]);
    const posEncodingNode = this.addLayer(this.layers[1], embeddingNode);

    let lastNode = posEncodingNode;
    // Transformer Blocks
    for (let i = 0; i < this.numLayers; i++) {
      const transformerBlock = this.layers[2 + i];
      lastNode = this.addLayer(transformerBlock, lastNode);
    }

    // Final Dense Layer and Softmax
    const denseLayer = this.layers[this.layers.length - 2];
    const softmaxLayer = this.layers[this.layers.length - 1];
    const denseNode = this.addLayer(denseLayer, lastNode);
    this.addLayer(softmaxLayer, denseNode);
  }
}
