// test-improvements.ts
// Verifiable tests for the new features in core.ts

import {
  AdaptiveNode,
  Graph,
  TestNode,
  createDelayNode,
  createThrottleNode,
  createDebounceNode,
  createErrorRecoveryNode,
  createLoadBalancerNode,
  createCacheNode,
  SubGraphNode,
  NodeError,
} from "./core";
import { strict as assert } from "assert";

// ============================================================================
// Test Runner
// ============================================================================

const tests: { [key: string]: () => Promise<void> } = {};

async function runAllTests() {
  let pass = 0;
  let fail = 0;

  for (const testName in tests) {
    try {
      await tests[testName]();
      console.log(`✓ ${testName}`);
      pass++;
    } catch (error) {
      console.error(`✗ ${testName}`);
      console.error(error);
      fail++;
    }
  }

  console.log(`\nTests complete: ${pass} passed, ${fail} failed.`);
  if (fail > 0) {
    process.exit(1);
  }
}

// ============================================================================
// 1. Error Handling and Recovery Test
// ============================================================================

tests["Error Handling and Recovery"] = async () => {
  const graph = new Graph();

  const unreliableNode = new AdaptiveNode<number, number>((input) => {
    if (input > 5) throw new Error("Value too high");
    return input * 2;
  }).setName("unreliable");

  const errorCapture = new TestNode<NodeError>().setName("errorCapture");
  const recoveryNode = createErrorRecoveryNode(-1); // Recover with -1
  const finalOutput = new TestNode<number>().setName("finalOutput");

  graph
    .addNode(unreliableNode)
    .addNode(errorCapture)
    .addNode(recoveryNode)
    .addNode(finalOutput);

  graph.connect(unreliableNode, finalOutput); // Main data path
  graph.connectError(unreliableNode, errorCapture);
  graph.connectError(unreliableNode, recoveryNode);
  graph.connect(recoveryNode, finalOutput); // Recovered data path

  // Test success path
  await graph.execute(3, unreliableNode.id);
  finalOutput.assertReceived([6]);
  errorCapture.assertReceived([]);

  // Test error path
  finalOutput.reset();
  await graph.execute(10, unreliableNode.id);
  assert.equal(errorCapture.receivedInputs.length, 1, "Error was not captured");
  finalOutput.assertReceived([-1]); // Should receive the recovered value
};

// ============================================================================
// 2. Async Flow Control Test
// ============================================================================

tests["Async Flow Control with maxConcurrent"] = async () => {
  let concurrentCount = 0;
  let maxConcurrent = 0;

  const slowNode = new AdaptiveNode<number, number>(
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

// ============================================================================
// 3. Time-based Operators Test
// ============================================================================

tests["Time-based Operators (Delay, Throttle, Debounce)"] = async () => {
  // Test Delay
  const delayNode = createDelayNode(100);
  const start = Date.now();
  await delayNode.process(null);
  const duration = Date.now() - start;
  assert(duration >= 100, `Delay was too short: ${duration}ms`);

  // Test Throttle
  const throttleNode = createThrottleNode(100);
  const throttleOutput = new TestNode<number>().setName("throttleOutput");
  throttleNode.outlets[0].connections.push({
    target: throttleOutput,
    transfer: async (data: number) => throttleOutput.process(data),
  } as any);

  await throttleNode.process(1); // Should pass
  await throttleNode.process(2); // Should be skipped
  await new Promise((resolve) => setTimeout(resolve, 110));
  await throttleNode.process(3); // Should pass

  throttleOutput.assertReceived([1, 3]);

  // Test Debounce
  const debounceNode = createDebounceNode(100);
  const debounceOutput = new TestNode<number>().setName("debounceOutput");
  debounceNode.outlets[0].connections.push({
    target: debounceOutput,
    transfer: async (data: number) => debounceOutput.process(data),
  } as any);

  debounceNode.process(1);
  debounceNode.process(2);
  await new Promise((resolve) => setTimeout(resolve, 50));
  debounceNode.process(3); // This should be the one that resolves
  await new Promise((resolve) => setTimeout(resolve, 110));

  debounceOutput.assertReceived([3]);
};

// ============================================================================
// 4. Circuit Breaker Test
// ============================================================================

tests["Circuit Breaker"] = async () => {
  const failingNode = new AdaptiveNode<number, number>(
    () => { throw new Error("Failure"); },
    { circuitBreakerThreshold: 2, circuitBreakerResetTime: 200 }
  ).setName("failing");

  const errorCapture = new TestNode<NodeError>().setName("errorCapture");
  failingNode.outlets[1].connections.push({
    target: errorCapture,
    transfer: async (data: NodeError) => errorCapture.process(data),
  } as any);

  // Trigger the breaker
  await failingNode.process(1);
  await failingNode.process(2);
  assert.equal(errorCapture.receivedInputs.length, 2, "Errors not captured before break");

  // Breaker should be open
  await failingNode.process(3);
  assert.equal(errorCapture.receivedInputs.length, 3, "Open circuit error not captured");
  assert(errorCapture.receivedInputs[2].error.message.includes("Circuit breaker is open"));

  // Wait for reset
  await new Promise((resolve) => setTimeout(resolve, 210));

  // Breaker should be closed
  await failingNode.process(4);
  assert.equal(errorCapture.receivedInputs.length, 4, "Error not captured after reset");
  assert.equal(errorCapture.receivedInputs[3].error.message, "Failure");
};

// ============================================================================
// 5. Load Balancer Test
// ============================================================================

tests["Load Balancer (Round Robin)"] = async () => {
  const worker1 = new TestNode<number>().setName("worker1");
  const worker2 = new TestNode<number>().setName("worker2");
  const loadBalancer = createLoadBalancerNode([worker1, worker2], { strategy: "round-robin" });

  await loadBalancer.process(1);
  await loadBalancer.process(2);
  await loadBalancer.process(3);

  worker1.assertReceived([1, 3]);
  worker2.assertReceived([2]);
};

// ============================================================================
// 6. Cache Node Test
// ============================================================================

tests["Cache Node"] = async () => {
  let computeCount = 0;
  const expensiveCompute = async (n: number) => {
    computeCount++;
    return n * n;
  };

  const cachedNode = createCacheNode(expensiveCompute, { ttl: 200 });

  // First calls
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 1);
  assert.equal(await cachedNode.process(10), 100);
  assert.equal(computeCount, 2);

  // Cached calls
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 2, "Cache was not used for first repeat");
  assert.equal(await cachedNode.process(10), 100);
  assert.equal(computeCount, 2, "Cache was not used for second repeat");

  // Wait for expiry
  await new Promise((resolve) => setTimeout(resolve, 210));

  // Recompute after expiry
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 3, "Cache did not expire");
};

// ============================================================================
// 7. Sub-graph Test
// ============================================================================

tests["Sub-graph Execution"] = async () => {
  const subGraph = new Graph();
  const add1 = new AdaptiveNode<number, number>((n) => n + 1).setName("add1");
  const add2 = new AdaptiveNode<number, number>((n) => n + 2).setName("add2");
  subGraph.addNode(add1).addNode(add2).connect(add1, add2);

  const subGraphNode = new SubGraphNode(subGraph, add1.id, add2.id);
  const mainGraph = new Graph().addNode(subGraphNode);
  const output = new TestNode<number>().setName("output");
  mainGraph.addNode(output).connect(subGraphNode, output);

  await mainGraph.execute(10, subGraphNode.id);

  output.assertReceived([13]); // 10 + 1 + 2
};

// ============================================================================
// Run all tests
// ============================================================================

runAllTests().catch((err) => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
