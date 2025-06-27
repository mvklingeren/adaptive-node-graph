import {
  AdaptiveNode,
  Graph,
  SubGraphNode,
  TestNode,
  createCacheNode,
  createDebounceNode,
  createDelayNode,
  createErrorRecoveryNode,
  createLoadBalancerNode,
  createThrottleNode
} from "./chunks/chunk-X2WD6DAB.js";

// src/test-improvements.ts
import { strict as assert } from "assert";
var tests = {};
async function runAllTests() {
  let pass = 0;
  let fail = 0;
  for (const testName in tests) {
    try {
      await tests[testName]();
      console.log(`\u2713 ${testName}`);
      pass++;
    } catch (error) {
      console.error(`\u2717 ${testName}`);
      console.error(error);
      fail++;
    }
  }
  console.log(`
Tests complete: ${pass} passed, ${fail} failed.`);
  if (fail > 0) {
    process.exit(1);
  }
}
tests["Error Handling and Recovery"] = async () => {
  const graph = new Graph();
  const unreliableNode = new AdaptiveNode((input) => {
    if (input > 5) throw new Error("Value too high");
    return input * 2;
  }).setName("unreliable");
  const errorCapture = new TestNode().setName("errorCapture");
  const recoveryNode = createErrorRecoveryNode(-1);
  const finalOutput = new TestNode().setName("finalOutput");
  graph.addNode(unreliableNode).addNode(errorCapture).addNode(recoveryNode).addNode(finalOutput);
  graph.connect(unreliableNode, finalOutput);
  graph.connectError(unreliableNode, errorCapture);
  graph.connectError(unreliableNode, recoveryNode);
  graph.connect(recoveryNode, finalOutput);
  await graph.execute(3, unreliableNode.id);
  finalOutput.assertReceived([6]);
  errorCapture.assertReceived([]);
  finalOutput.reset();
  await graph.execute(10, unreliableNode.id);
  assert.equal(errorCapture.receivedInputs.length, 1, "Error was not captured");
  finalOutput.assertReceived([-1]);
};
tests["Async Flow Control with maxConcurrent"] = async () => {
  let concurrentCount = 0;
  let maxConcurrent = 0;
  const slowNode = new AdaptiveNode(
    async (input) => {
      concurrentCount++;
      maxConcurrent = Math.max(maxConcurrent, concurrentCount);
      await new Promise((resolve) => setTimeout(resolve, 50));
      concurrentCount--;
      return input;
    },
    { maxConcurrent: 2 }
  ).setName("slowNode");
  const promises = Array.from({ length: 5 }, (_, i) => slowNode.process(i));
  await Promise.all(promises);
  assert.equal(maxConcurrent, 2, "Concurrency limit was not respected");
};
tests["Time-based Operators (Delay, Throttle, Debounce)"] = async () => {
  const delayNode = createDelayNode(100);
  const start = Date.now();
  await delayNode.process(null);
  const duration = Date.now() - start;
  assert(duration >= 100, `Delay was too short: ${duration}ms`);
  const throttleNode = createThrottleNode(100);
  const throttleOutput = new TestNode().setName("throttleOutput");
  throttleNode.outlets[0].connections.push({
    target: throttleOutput,
    transfer: async (data) => throttleOutput.process(data)
  });
  await throttleNode.process(1);
  await throttleNode.process(2);
  await new Promise((resolve) => setTimeout(resolve, 110));
  await throttleNode.process(3);
  throttleOutput.assertReceived([1, 3]);
  const debounceNode = createDebounceNode(100);
  const debounceOutput = new TestNode().setName("debounceOutput");
  debounceNode.outlets[0].connections.push({
    target: debounceOutput,
    transfer: async (data) => debounceOutput.process(data)
  });
  debounceNode.process(1);
  debounceNode.process(2);
  await new Promise((resolve) => setTimeout(resolve, 50));
  debounceNode.process(3);
  await new Promise((resolve) => setTimeout(resolve, 110));
  debounceOutput.assertReceived([3]);
};
tests["Circuit Breaker"] = async () => {
  const failingNode = new AdaptiveNode(
    () => {
      throw new Error("Failure");
    },
    { circuitBreakerThreshold: 2, circuitBreakerResetTime: 200 }
  ).setName("failing");
  const errorCapture = new TestNode().setName("errorCapture");
  failingNode.outlets[1].connections.push({
    target: errorCapture,
    transfer: async (data) => errorCapture.process(data)
  });
  await failingNode.process(1);
  await failingNode.process(2);
  assert.equal(errorCapture.receivedInputs.length, 2, "Errors not captured before break");
  await failingNode.process(3);
  assert.equal(errorCapture.receivedInputs.length, 3, "Open circuit error not captured");
  assert(errorCapture.receivedInputs[2].error.message.includes("Circuit breaker is open"));
  await new Promise((resolve) => setTimeout(resolve, 210));
  await failingNode.process(4);
  assert.equal(errorCapture.receivedInputs.length, 4, "Error not captured after reset");
  assert.equal(errorCapture.receivedInputs[3].error.message, "Failure");
};
tests["Load Balancer (Round Robin)"] = async () => {
  const worker1 = new TestNode().setName("worker1");
  const worker2 = new TestNode().setName("worker2");
  const loadBalancer = createLoadBalancerNode([worker1, worker2], { strategy: "round-robin" });
  await loadBalancer.process(1);
  await loadBalancer.process(2);
  await loadBalancer.process(3);
  worker1.assertReceived([1, 3]);
  worker2.assertReceived([2]);
};
tests["Cache Node"] = async () => {
  let computeCount = 0;
  const expensiveCompute = async (n) => {
    computeCount++;
    return n * n;
  };
  const cachedNode = createCacheNode(expensiveCompute, { ttl: 200 });
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 1);
  assert.equal(await cachedNode.process(10), 100);
  assert.equal(computeCount, 2);
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 2, "Cache was not used for first repeat");
  assert.equal(await cachedNode.process(10), 100);
  assert.equal(computeCount, 2, "Cache was not used for second repeat");
  await new Promise((resolve) => setTimeout(resolve, 210));
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 3, "Cache did not expire");
};
tests["Sub-graph Execution"] = async () => {
  const subGraph = new Graph();
  const add1 = new AdaptiveNode((n) => n + 1).setName("add1");
  const add2 = new AdaptiveNode((n) => n + 2).setName("add2");
  subGraph.addNode(add1).addNode(add2).connect(add1, add2);
  const subGraphNode = new SubGraphNode(subGraph, add1.id, add2.id);
  const mainGraph = new Graph().addNode(subGraphNode);
  const output = new TestNode().setName("output");
  mainGraph.addNode(output).connect(subGraphNode, output);
  await mainGraph.execute(10, subGraphNode.id);
  output.assertReceived([13]);
};
runAllTests().catch((err) => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
