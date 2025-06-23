import {
  Graph,
  OscillatorNode,
  createFloat32MultiplyNode
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/oscillator.ts
var graph = new Graph();
var osc = new OscillatorNode();
var multiply = createFloat32MultiplyNode();
graph.addNode(osc);
graph.addNode(multiply);
graph.connect(osc, multiply);
var result = await graph.execute({
  frequency: 440,
  amplitude: 0.5,
  sampleRate: 44100
});
