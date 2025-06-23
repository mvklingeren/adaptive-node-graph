import {
  AdaptiveNode,
  Graph,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/type-adaptive.ts
var smartProcessor = new AdaptiveNode(
  (input) => `Unknown type: ${typeof input}`
).register(Number, (num) => `Number: ${num.toFixed(2)}`).register(String, (str) => `String: "${str.toUpperCase()}"`).register(Array, (arr) => `Array[${arr.length}]: ${arr.join(", ")}`).register(Date, (date) => `Date: ${date.toISOString()}`).setLabel("smart-processor");
var logger = createProcessor(
  (msg) => console.log(msg),
  "logger"
);
var graph = new Graph();
graph.addNode(smartProcessor);
graph.addNode(logger);
graph.connect(smartProcessor, logger);
await smartProcessor.process(42);
await smartProcessor.process("hello");
await smartProcessor.process([1, 2, 3]);
await smartProcessor.process(/* @__PURE__ */ new Date());
