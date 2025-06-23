import { Graph, AdaptiveNode, OscillatorNode, createProcessor } from '../core';

// Envelope generator
const envelope = createProcessor<number, number>((time) => {
  // Simple ADSR envelope
  if (time < 0.1) return time / 0.1;           // Attack
  if (time < 0.2) return 1;                    // Decay to sustain
  if (time < 0.8) return 0.7;                  // Sustain
  return 0.7 * (1 - (time - 0.8) / 0.2);      // Release
}, 'envelope');

// Filter node
const filter = new AdaptiveNode<Float32Array, Float32Array>((samples) => {
  // Simple low-pass filter
  const filtered = new Float32Array(samples.length);
  filtered[0] = samples[0];
  
  for (let i = 1; i < samples.length; i++) {
    filtered[i] = filtered[i-1] * 0.9 + samples[i] * 0.1;
  }
  
  return filtered;
}).setLabel('lowpass');

// Build synthesizer graph
const graph = new Graph();
const osc = new OscillatorNode();
const gain = createProcessor<[Float32Array, number], Float32Array>(
  ([samples, gainValue]) => samples.map(s => s * gainValue),
  'gain'
);

graph.addNode(osc);
graph.addNode(envelope);
graph.addNode(filter);
graph.addNode(gain);

// Generate 1 second of audio
const params = {
  frequency: 440,
  amplitude: 0.5,
  sampleRate: 44100,
  length: 44100,
  waveform: 'sawtooth' as const
};

const samples = await osc.process(params);
const filtered = await filter.process(samples);

// Apply envelope
const envelopeValues = await Promise.all(
  Array.from({ length: 44100 }, (_, i) => envelope.process(i / 44100))
);
const output = filtered.map((sample, i) => sample * envelopeValues[i]);