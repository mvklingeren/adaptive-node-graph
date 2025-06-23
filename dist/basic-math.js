import {
  Graph,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/basic-math.ts
var graph = new Graph();
var input = createProcessor((x) => x, "input");
var add5 = createProcessor((x) => x + 5, "add5");
var multiply2 = createProcessor((x) => x * 2, "multiply2");
var output = createProcessor((x) => console.log("Result:", x), "output");
graph.addNode(input).addNode(add5).addNode(multiply2).addNode(output);
graph.connect(input, add5);
graph.connect(add5, multiply2);
graph.connect(multiply2, output);
await graph.execute(10);
