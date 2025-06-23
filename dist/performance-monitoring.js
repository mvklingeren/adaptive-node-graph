import {
  AdaptiveNode
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/performance-monitoring.ts
function expensiveOperation(input) {
  const iterations = input?.iterations || 1e6;
  let result = 0;
  for (let i = 0; i < iterations; i++) {
    result += Math.sin(i) * Math.cos(i);
  }
  return result;
}
var monitoredNode = new AdaptiveNode((input) => {
  const start = performance.now();
  const result = expensiveOperation(input);
  const duration = performance.now() - start;
  if (duration > 100) {
    console.warn(`Slow operation: ${duration}ms`);
  }
  return result;
}).setLabel("monitored");
var stats = monitoredNode.getPerformanceStats();
console.log("Performance stats:", stats);
