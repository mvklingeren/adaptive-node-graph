import {
  AdaptiveNode,
  Graph,
  OscillatorNode,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/audio-processing.ts
var envelope = createProcessor((time) => {
  if (time < 0.1) return time / 0.1;
  if (time < 0.2) return 1;
  if (time < 0.8) return 0.7;
  return 0.7 * (1 - (time - 0.8) / 0.2);
}, "envelope");
var filter = new AdaptiveNode((samples2) => {
  const filtered2 = new Float32Array(samples2.length);
  filtered2[0] = samples2[0];
  for (let i = 1; i < samples2.length; i++) {
    filtered2[i] = filtered2[i - 1] * 0.9 + samples2[i] * 0.1;
  }
  return filtered2;
}).setLabel("lowpass");
var graph = new Graph();
var osc = new OscillatorNode();
var gain = createProcessor(
  ([samples2, gainValue]) => samples2.map((s) => s * gainValue),
  "gain"
);
graph.addNode(osc);
graph.addNode(envelope);
graph.addNode(filter);
graph.addNode(gain);
var params = {
  frequency: 440,
  amplitude: 0.5,
  sampleRate: 44100,
  length: 44100,
  waveform: "sawtooth"
};
var samples = await osc.process(params);
var filtered = await filter.process(samples);
var envelopeValues = await Promise.all(
  Array.from({ length: 44100 }, (_, i) => envelope.process(i / 44100))
);
var output = filtered?.map((sample, i) => sample * envelopeValues[i]);
