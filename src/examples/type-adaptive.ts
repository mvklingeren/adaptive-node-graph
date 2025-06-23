import { AdaptiveNode } from '../core';
import { createProcessor } from '../core';
import { Graph } from '../core';

// Node that handles different data types
const smartProcessor = new AdaptiveNode<any, string>(
  (input) => `Unknown type: ${typeof input}`
)
  .register(Number, (num) => `Number: ${num.toFixed(2)}`)
  .register(String, (str) => `String: "${str.toUpperCase()}"`)
  .register(Array, (arr) => `Array[${arr.length}]: ${arr.join(', ')}`)
  .register(Date, (date) => `Date: ${date.toISOString()}`)
  .setLabel('smart-processor');

const logger = createProcessor<string, void>(
  (msg) => console.log(msg),
  'logger'
);

const graph = new Graph();
graph.addNode(smartProcessor);
graph.addNode(logger);
graph.connect(smartProcessor, logger);

// Test with different types
await smartProcessor.process(42);           // "Number: 42.00"
await smartProcessor.process("hello");      // "String: "HELLO""
await smartProcessor.process([1, 2, 3]);    // "Array[3]: 1, 2, 3"
await smartProcessor.process(new Date());   // "Date: 2024-01-20T..."