import {
  AdaptiveNode,
  Graph,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/machine-learning.ts
var DataPreprocessor = class extends AdaptiveNode {
  constructor() {
    super((data) => new Float32Array(data));
    this.register(Array, this.preprocessArray.bind(this));
    this.register(Float32Array, this.preprocessFloat32.bind(this));
    this.register(ImageData, this.preprocessImage.bind(this));
  }
  preprocessArray(data) {
    const arr = new Float32Array(data);
    const mean = arr.reduce((a, b) => a + b) / arr.length;
    const std = Math.sqrt(
      arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length
    );
    return arr.map((val) => (val - mean) / std);
  }
  preprocessFloat32(data) {
    return this.preprocessArray(Array.from(data));
  }
  preprocessImage(data) {
    const gray = new Float32Array(data.width * data.height);
    for (let i = 0; i < gray.length; i++) {
      const idx = i * 4;
      gray[i] = (data.data[idx] * 0.299 + data.data[idx + 1] * 0.587 + data.data[idx + 2] * 0.114) / 255;
    }
    return gray;
  }
};
var featureExtractor = createProcessor((data) => {
  const features = new Float32Array(5);
  features[0] = Math.min(...data);
  features[1] = Math.max(...data);
  features[2] = data.reduce((a, b) => a + b) / data.length;
  features[3] = Math.sqrt(
    data.reduce((sum, val) => sum + val * val, 0) / data.length
  );
  features[4] = data.reduce(
    (sum, val, i) => i > 0 ? sum + Math.abs(val - data[i - 1]) : sum,
    0
  );
  return features;
}, "feature-extractor");
var model = createProcessor((features) => {
  const sum = features.reduce((a, b) => a + b);
  if (sum > 10) {
    return { class: "positive", confidence: 0.85 };
  } else if (sum < -10) {
    return { class: "negative", confidence: 0.9 };
  } else {
    return { class: "neutral", confidence: 0.75 };
  }
}, "ml-model");
var postProcessor = new AdaptiveNode((prediction) => prediction).register(Object, (pred) => {
  if (pred.confidence < 0.5) {
    return { ...pred, class: "uncertain" };
  }
  return pred;
}).setLabel("post-processor");
var graph = new Graph();
var preprocessor = new DataPreprocessor().setLabel("preprocessor");
graph.addNode(preprocessor);
graph.addNode(featureExtractor);
graph.addNode(model);
graph.addNode(postProcessor);
graph.connect(preprocessor, featureExtractor);
graph.connect(featureExtractor, model);
graph.connect(model, postProcessor);
var testData = [
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  new Float32Array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]),
  Array.from({ length: 100 }, () => Math.random() * 10 - 5)
];
for (const data of testData) {
  const result = await graph.execute(data, preprocessor.id);
  console.log("Prediction:", result);
}
