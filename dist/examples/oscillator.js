import { Graph, createFloat32MultiplyNode, OscillatorNode } from '../core';
const graph = new Graph();
const osc = new OscillatorNode();
const multiply = createFloat32MultiplyNode();
graph.addNode(osc);
graph.addNode(multiply);
graph.connect(osc, multiply);
const result = await graph.execute({
    frequency: 440,
    amplitude: 0.5,
    sampleRate: 44100
});
