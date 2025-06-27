// test-complex.ts
// A complex graph test with 20+ nodes, multiple paths, cycles, and advanced features.

import {
  Graph,
  AdaptiveNode,
  TestNode,
  SubGraphNode,
  createProcessor,
  createDelayNode,
  createCacheNode,
  createLoadBalancerNode,
  createAddNode,
  createMultiplyNode,
  createGateNode,
  createMergeNode,
  createSplitNode,
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

  console.log(`\nComplex tests complete: ${pass} passed, ${fail} failed.`);
  if (fail > 0) {
    process.exit(1);
  }
}

// ============================================================================
// Complex Graph Test
// ============================================================================

tests["Complex Graph with Cycles and Advanced Features"] = async () => {
  const graph = new Graph();

  // --- Test Nodes to Capture Output ---
  const finalOutput = new TestNode<any>().setName("finalOutput");
  const errorOutput = new TestNode<NodeError>().setName("errorOutput");
  const pathA_Output = new TestNode<number>().setName("pathA_Output");
  const pathB_Output = new TestNode<any>().setName("pathB_Output");
  const feedback_Output = new TestNode<number>().setName("feedback_Output");

  // --- Nodes (20+ nodes) ---
  // 1. Input
  const input = new AdaptiveNode<number, number>((n) => n).setName("input");

  // 2. Splitter
  const splitter = createSplitNode(3); // Splits into 3 paths

  // --- Path A: Heavy Computation with Caching ---
  // 3. Path A Processor
  let pathA_ComputeCount = 0;
  const pathA_processorFn = async (n: number): Promise<number> => {
    pathA_ComputeCount++;
    await new Promise(resolve => setTimeout(resolve, 50)); // Simulate work
    return n * 2;
  };
  const pathA_Processor = createProcessor<number, number>(pathA_processorFn, "pathA_Processor");

  // 4. Path A Cache
  const pathA_Cache = createCacheNode(pathA_processorFn, { ttl: 5000 });

  // 5. Path A Adder
  const pathA_Add = createProcessor<number, number>((n) => n + 5, "pathA_Add");

  // --- Path B: Sub-graph and Load Balancer ---
  // 6. Sub-graph
  const subGraph = new Graph();
  const sub_add = new AdaptiveNode<number, number[]>((n) => [n, 5]);
  const sub_adder = createAddNode();
  subGraph.addNode(sub_add).addNode(sub_adder).connect(sub_add, sub_adder);
  const subGraphNode = new SubGraphNode<number, number>(subGraph, sub_add.id, sub_adder.id).setName("subGraph");

  // 7, 8, 9. Load Balancer Workers
  const worker1 = createProcessor<number, string>((n) => `w1:${n}`, "worker1");
  const worker2 = createProcessor<number, string>((n) => `w2:${n}`, "worker2");
  const worker3 = createProcessor<number, string>((n) => { throw new Error("Worker 3 fails"); }, "worker3");
  
  // 10. Load Balancer
  const loadBalancer = createLoadBalancerNode([worker1, worker2, worker3], { strategy: "round-robin" });

  // --- Path C: Feedback Loop (Cycle) ---
  // 11. Feedback Gate
  const feedbackGate = createGateNode(); // Initially closed

  // 12. Feedback Delay
  const feedbackDelay = createDelayNode(100);

  // 13. Feedback Processor
  const feedbackProcessor = createProcessor<number, number>((n) => n - 1, "feedbackProcessor");

  // --- Merging and Final Processing ---
  // 14. Main Merger
  const merger = createMergeNode<number | string>(2);

  // 15. Final Processor
  const finalProcessor = createProcessor<any[], any>((arr) => arr.sort(), "finalProcessor");

  // 16-21: Add all nodes to graph
  graph
    .addNode(input).addNode(splitter)
    .addNode(pathA_Cache).addNode(pathA_Add).addNode(pathA_Output)
    .addNode(subGraphNode).addNode(loadBalancer).addNode(pathB_Output)
    .addNode(feedbackGate).addNode(feedbackDelay).addNode(feedbackProcessor).addNode(feedback_Output)
    .addNode(merger).addNode(finalProcessor).addNode(finalOutput)
    .addNode(errorOutput);
  
  // --- Connections ---
  // Input to Splitter
  graph.connect(input, splitter);

  // Path A
  graph.connect(splitter, pathA_Cache, undefined, 0, 0);
  graph.connect(pathA_Cache, pathA_Add);
  graph.connect(pathA_Add, pathA_Output);
  graph.connect(pathA_Add, merger);

  // Path B
  graph.connect(splitter, subGraphNode, undefined, 1, 0);
  graph.connect(subGraphNode, loadBalancer);
  graph.connect(loadBalancer, pathB_Output);
  graph.connect(loadBalancer, merger);
  graph.connectError(loadBalancer, errorOutput); // Capture worker3 error

  // Path C (Feedback Loop)
  graph.connect(splitter, feedbackGate as AdaptiveNode<any, any>, undefined, 2, 0);
  graph.connect(feedbackGate as AdaptiveNode<any, any>, feedbackProcessor as AdaptiveNode<any, number>);
  graph.connect(feedbackProcessor, feedbackDelay);
  graph.connect(feedbackDelay, feedback_Output);
  graph.connect(feedbackDelay, feedbackGate as AdaptiveNode<any, any>, (n: number) => [n > 1, n] as any); // Cycle back to the gate

  // Final path
  graph.connect(merger, finalProcessor as AdaptiveNode<any, any>);
  graph.connect(finalProcessor, finalOutput);

  // --- Execution and Assertions ---
  // Run 1: Should go to worker1
  await graph.execute(10, input.id);
  pathA_Output.assertReceived([25]); // (10 * 2) + 5
  pathB_Output.assertReceived(["w1:15"]); // subGraphNode passes 15, LB sends to w1
  assert.equal(pathA_ComputeCount, 1, "Path A should compute once");

  // Run 2: Should go to worker2 (and test cache)
  pathB_Output.reset();
  await graph.execute(10, input.id);
  pathB_Output.assertReceived(["w2:15"]); // subGraphNode passes 15, LB sends to w2
  assert.equal(pathA_ComputeCount, 1, "Path A cache was not used");

  // Run 3: Should go to worker3 and fail
  pathB_Output.reset();
  await graph.execute(10, input.id);
  pathB_Output.assertReceived([]); // No output on failure
  assert.equal(errorOutput.receivedInputs.length, 1, "Worker error not caught");

  // Third run (with feedback loop)
  feedbackGate.setInitialValue([true, 5]); // Open the gate with a new value
  await graph.execute(null, feedbackGate.id);

  // Wait for feedback loop to run a few times
  await new Promise(resolve => setTimeout(resolve, 500));
  graph.stop(); // Stop the graph to prevent infinite loops

  feedback_Output.assertReceived([4, 3, 2, 1]); // 5 -> 4 -> 3 -> 2 -> 1
  
  console.log("Complex graph executed successfully.");
};

// ============================================================================
// Run all tests
// ============================================================================

runAllTests().catch((err) => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
