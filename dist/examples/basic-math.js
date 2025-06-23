import { Graph, createProcessor } from '../core';
const graph = new Graph();
// Create nodes
const input = createProcessor(x => x, 'input');
const add5 = createProcessor(x => x + 5, 'add5');
const multiply2 = createProcessor(x => x * 2, 'multiply2');
const output = createProcessor(x => console.log('Result:', x), 'output');
// Build graph: (input + 5) * 2
graph
    .addNode(input)
    .addNode(add5)
    .addNode(multiply2)
    .addNode(output);
graph.connect(input, add5);
graph.connect(add5, multiply2);
graph.connect(multiply2, output);
// Execute
await graph.execute(10); // Result: 30
