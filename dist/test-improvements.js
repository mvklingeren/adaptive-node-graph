import {
  AdaptiveNode,
  Graph,
  SubGraphNode,
  TestNode,
  createCacheNode,
  createDelayNode,
  createErrorLoggerNode,
  createErrorRecoveryNode,
  createLoadBalancerNode,
  createThrottleNode
} from "./chunks/chunk-PSE6PIOY.js";

// src/test-improvements.ts
console.log("=== Error Handling Demo ===");
async function errorHandlingDemo() {
  const graph = new Graph();
  const unreliableNode = new AdaptiveNode((input) => {
    if (input > 5) {
      throw new Error(`Value ${input} is too high!`);
    }
    return input * 2;
  }).setLabel("unreliable");
  const errorLogger = createErrorLoggerNode();
  const errorRecovery = createErrorRecoveryNode(0);
  graph.addNode(unreliableNode);
  graph.addNode(errorLogger);
  graph.addNode(errorRecovery);
  graph.connectError(unreliableNode, errorLogger);
  graph.connect(errorLogger, errorRecovery);
  console.log("Processing 3:", await unreliableNode.process(3));
  console.log("Processing 10:", await unreliableNode.process(10));
}
console.log("\n=== Type Safety Demo ===");
function typeSafetyDemo() {
  const numberNode = new AdaptiveNode((n) => n.toString());
  const stringNode = new AdaptiveNode((s) => s.length > 0);
  const graph = new Graph();
  graph.addNode(numberNode);
  graph.addNode(stringNode);
  graph.connect(numberNode, stringNode);
  console.log("Type-safe connections established");
}
console.log("\n=== Async Flow Control Demo ===");
async function flowControlDemo() {
  const slowNode = new AdaptiveNode(
    async (input) => {
      console.log(`Processing ${input}...`);
      await new Promise((resolve) => setTimeout(resolve, 100));
      return input * 2;
    },
    { maxConcurrent: 2 }
    // Only 2 concurrent operations
  ).setLabel("slowNode");
  const promises = [];
  for (let i = 0; i < 5; i++) {
    promises.push(slowNode.process(i));
  }
  console.log("Started 5 operations with max concurrency of 2");
  const results = await Promise.all(promises);
  console.log("Results:", results);
}
console.log("\n=== Time-based Operators Demo ===");
async function timeBasedDemo() {
  const graph = new Graph();
  const delayNode = createDelayNode(500);
  const throttleNode = createThrottleNode(1e3);
  console.log("Testing delay node...");
  const start = Date.now();
  await delayNode.process("delayed");
  console.log(`Delayed by ${Date.now() - start}ms`);
  console.log("\nTesting throttle node...");
  for (let i = 0; i < 5; i++) {
    const result = await throttleNode.process(i);
    console.log(`Throttle result ${i}:`, result);
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
}
console.log("\n=== Testing Utilities Demo ===");
async function testingDemo() {
  const testNode1 = new TestNode();
  const testNode2 = new TestNode((n) => n * 2);
  const graph = new Graph();
  graph.addNode(testNode1);
  graph.addNode(testNode2);
  graph.connect(testNode1, testNode2);
  await testNode1.process(5);
  await testNode1.process(10);
  try {
    testNode1.assertReceived([5, 10]);
    testNode2.assertProcessed([10, 20]);
    testNode1.assertNoErrors();
    console.log("All tests passed!");
  } catch (error) {
    console.error("Test failed:", error);
  }
}
console.log("\n=== Circuit Breaker Demo ===");
async function circuitBreakerDemo() {
  const failingNode = new AdaptiveNode(
    () => {
      throw new Error("Always fails");
    },
    {
      circuitBreakerThreshold: 3,
      circuitBreakerResetTime: 2e3
    }
  ).setLabel("failing");
  const errorLogger = createErrorLoggerNode();
  const graph = new Graph();
  graph.addNode(failingNode);
  graph.addNode(errorLogger);
  graph.connectError(failingNode, errorLogger);
  console.log("Attempting to trigger circuit breaker...");
  for (let i = 0; i < 5; i++) {
    await failingNode.process(i);
    console.log(`Attempt ${i + 1} completed`);
  }
  console.log("Circuit breaker should be open now");
  console.log("Waiting 2 seconds for circuit breaker reset...");
  await new Promise((resolve) => setTimeout(resolve, 2100));
  console.log("Trying again after reset...");
  await failingNode.process(99);
}
console.log("\n=== Load Balancer Demo ===");
async function loadBalancerDemo() {
  const workers = [
    new AdaptiveNode((n) => `Worker1: ${n}`).setLabel("worker1"),
    new AdaptiveNode((n) => `Worker2: ${n}`).setLabel("worker2"),
    new AdaptiveNode((n) => `Worker3: ${n}`).setLabel("worker3")
  ];
  const loadBalancer = createLoadBalancerNode(workers, "round-robin");
  console.log("Round-robin load balancing:");
  for (let i = 0; i < 6; i++) {
    const result = await loadBalancer.process(i);
    console.log(result);
  }
}
console.log("\n=== Cache Node Demo ===");
async function cacheDemo() {
  let computeCount = 0;
  const expensiveCompute = async (n) => {
    computeCount++;
    console.log(`Computing ${n}... (computation #${computeCount})`);
    await new Promise((resolve) => setTimeout(resolve, 100));
    return n * n;
  };
  const cachedNode = createCacheNode(expensiveCompute, 1e3, 10);
  console.log("First calls:");
  console.log(await cachedNode.process(5));
  console.log(await cachedNode.process(10));
  console.log("\nRepeated calls (cached):");
  console.log(await cachedNode.process(5));
  console.log(await cachedNode.process(10));
  console.log("\nWaiting for cache expiry...");
  await new Promise((resolve) => setTimeout(resolve, 1100));
  console.log("\nAfter cache expiry:");
  console.log(await cachedNode.process(5));
}
console.log("\n=== Sub-graph Demo ===");
async function subGraphDemo() {
  const subGraph = new Graph();
  const add1 = new AdaptiveNode((n) => n + 1).setLabel("add1");
  const add2 = new AdaptiveNode((n) => n + 2).setLabel("add2");
  subGraph.addNode(add1);
  subGraph.addNode(add2);
  subGraph.connect(add1, add2);
  const subGraphNode = new SubGraphNode(subGraph, add1.id, add2.id);
  const result = await subGraphNode.process(10);
  console.log("Sub-graph result:", result);
}
async function runAllDemos() {
  try {
    await errorHandlingDemo();
    typeSafetyDemo();
    await flowControlDemo();
    await timeBasedDemo();
    await testingDemo();
    await circuitBreakerDemo();
    await loadBalancerDemo();
    await cacheDemo();
    await subGraphDemo();
    console.log("\n=== All demos completed successfully! ===");
  } catch (error) {
    console.error("Demo error:", error);
  }
}
runAllDemos().catch(console.error);
export {
  runAllDemos
};
