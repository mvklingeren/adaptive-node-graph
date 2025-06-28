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
  const delayGraph = new Graph();
  const delayInput = new TestNode<number>().setName("delayInput");
  const delayNode = createDelayNode(100);
  const delayOutput = new TestNode<number>().setName("delayOutput");
  delayGraph.addNode(delayInput).addNode(delayNode).addNode(delayOutput);
  delayGraph.connect(delayInput, delayNode);
  delayGraph.connect(delayNode, delayOutput);

  const start = Date.now();
  await delayGraph.execute(1, delayInput.id);
  const duration = Date.now() - start;
  assert(duration >= 100, `Delay was too short: ${duration}ms`);
  delayOutput.assertReceived([1]);

  // Test Throttle
  const throttleGraph = new Graph();
  const throttleInput = new AdaptiveNode<number, number>((n) => n).setName("throttleInput");
  const throttleNode = createThrottleNode<number>(100);
  const throttleOutput = new TestNode<number | null>().setName("throttleOutput");
  throttleGraph.addNode(throttleInput).addNode(throttleNode).addNode(throttleOutput);
  throttleGraph.connect(throttleInput, throttleNode);
  throttleGraph.connect(throttleNode, throttleOutput);

  await throttleGraph.execute(1, throttleInput.id); // Should pass
  await throttleGraph.execute(2, throttleInput.id); // Should be skipped
  await new Promise((resolve) => setTimeout(resolve, 110));
  await throttleGraph.execute(3, throttleInput.id); // Should pass
  assert.deepStrictEqual(
    throttleOutput.receivedInputs.filter((i) => i !== null),
    [1, 3]
  );

  // Test Debounce
  const debounceGraph = new Graph();
  const debounceInput = new AdaptiveNode<number, number>((n) => n).setName("debounceInput");
  const debounceNode = createDebounceNode<number>(100);
  const debounceOutput = new TestNode<number | null>().setName("debounceOutput");
  debounceGraph.addNode(debounceInput).addNode(debounceNode).addNode(debounceOutput);
  debounceGraph.connect(debounceInput, debounceNode);
  debounceGraph.connect(debounceNode, debounceOutput);

  // Fire multiple inputs in quick succession
  debounceGraph.execute(1, debounceInput.id);
  debounceGraph.execute(2, debounceInput.id);
  await new Promise((resolve) => setTimeout(resolve, 50));
  debounceGraph.execute(3, debounceInput.id); // This should be the one that resolves

  // Wait for debounce time to pass
  await new Promise((resolve) => setTimeout(resolve, 110));
  assert.deepStrictEqual(
    debounceOutput.receivedInputs.filter((i) => i !== null),
    [3]
  );
};

// ============================================================================
// 4. Circuit Breaker Test
// ============================================================================

tests["Circuit Breaker"] = async () => {
  const graph = new Graph();
  const failingNode = new AdaptiveNode<number, number>(
    () => {
      throw new Error("Failure");
    },
    { circuitBreakerThreshold: 2, circuitBreakerResetTime: 200 }
  ).setName("failing");

  const errorCapture = new TestNode<NodeError>().setName("errorCapture");
  graph.addNode(failingNode).addNode(errorCapture);
  graph.connectError(failingNode, errorCapture);

  // Trigger the breaker
  await graph.execute(1, failingNode.id);
  await graph.execute(2, failingNode.id);
  assert.equal(errorCapture.receivedInputs.length, 2, "Errors not captured before break");

  // Breaker should be open
  await graph.execute(3, failingNode.id);
  assert.equal(errorCapture.receivedInputs.length, 3, "Open circuit error not captured");
  assert(
    errorCapture.receivedInputs[2].error.message.includes(
      "Circuit breaker is open"
    )
  );

  // Wait for reset
  await new Promise((resolve) => setTimeout(resolve, 210));

  // Breaker should be closed
  await graph.execute(4, failingNode.id);
  assert.equal(errorCapture.receivedInputs.length, 4, "Error not captured after reset");
  assert.equal(errorCapture.receivedInputs[3].error.message, "Failure");
};

// ============================================================================
// 5. Load Balancer Test
// ============================================================================

tests["Load Balancer (Round Robin)"] = async () => {
  const graph = new Graph();
  const worker1 = new TestNode<number>().setName("worker1");
  const worker2 = new TestNode<number>().setName("worker2");
  const loadBalancer = createLoadBalancerNode([worker1, worker2], {
    strategy: "round-robin",
  });

  graph.addNode(loadBalancer).addNode(worker1).addNode(worker2);
  // Note: The load balancer internally processes to its workers,
  // so we don't connect them in the graph here. We just need to execute it.

  await graph.execute(1, loadBalancer.id);
  await graph.execute(2, loadBalancer.id);
  await graph.execute(3, loadBalancer.id);

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
