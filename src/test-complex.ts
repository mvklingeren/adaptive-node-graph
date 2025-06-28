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
  const merger = createMergeNode<any>(2);
  const finalProcessor = createProcessor<any[], any>((arr) => arr.sort(), "finalProcessor");

  // Add nodes to graph
  graph
    .addNode(input).addNode(splitter)
    .addNode(pathA_Cache).addNode(pathA_Add).addNode(pathA_Output)
    .addNode(subGraphNode).addNode(loadBalancer).addNode(pathB_Output)
    .addNode(feedbackGate).addNode(feedbackDelay).addNode(feedbackProcessor).addNode(feedback_Output)
    .addNode(merger).addNode(finalProcessor).addNode(finalOutput)
    .addNode(errorOutput);

  // --- Connections ---
  // Create a transformer for the feedback loop to convert number to [boolean, number]
  const feedbackTransformer = createProcessor<number, [boolean, number]>(
    (n: number) => [n > 1, n],
    "feedbackTransformer"
  );
  graph.addNode(feedbackTransformer);

  // Input to Splitter
  graph.connect(input, splitter);

  // Path A
  graph.connect(splitter, pathA_Cache, undefined, 0);
  graph.connect(pathA_Cache, pathA_Add);
  graph.connect(pathA_Add, pathA_Output);

  // Path B
  graph.connect(splitter, subGraphNode, undefined, 1);
  graph.connect(subGraphNode, loadBalancer);
  graph.connect(loadBalancer, pathB_Output);
  graph.connectError(loadBalancer, errorOutput);

  // Path C (Feedback Loop)
  graph.connect(splitter, feedbackGate as AdaptiveNode<any, any>, undefined, 2);
  graph.connect(feedbackGate as AdaptiveNode<any, any>, feedbackProcessor as AdaptiveNode<any, number>);
  graph.connect(feedbackProcessor, feedbackDelay);
  graph.connect(feedbackDelay, feedback_Output);
  // The cycle is created by feeding the output of the delay back into the gate via a transformer node
  graph.connect(feedbackDelay, feedbackTransformer);
  graph.connect(feedbackTransformer, feedbackGate as AdaptiveNode<any, any>);

  // --- Execution and Assertions ---

  // --- Test Path A & B (Linear Flow) ---
  console.log("Testing linear paths A and B...");
  // Run 1: Should go to worker1
  await graph.execute(10, input.id);
  pathA_Output.assertReceived([25]); // (10 * 2) + 5
  pathB_Output.assertReceived(["w1:15"]); // subGraphNode passes 15, LB sends to w1
  assert.equal(pathA_ComputeCount, 1, "Path A should compute once");

  // Run 2: Should go to worker2 (and test cache)
  pathA_Output.reset();
  pathB_Output.reset();
  await graph.execute(10, input.id);
  pathA_Output.assertReceived([25]); // Cached result
  pathB_Output.assertReceived(["w2:15"]); // LB sends to w2
  assert.equal(pathA_ComputeCount, 1, "Path A cache was not used");

  // Run 3: Should go to worker3 and fail
  pathB_Output.reset();
  await graph.execute(12, input.id); // Use a different value to avoid cache
  pathB_Output.assertReceived([]); // No output on failure
  assert.equal(errorOutput.receivedInputs.length, 1, "Worker error not caught");
  assert.equal(errorOutput.receivedInputs[0].error.message, "Worker 3 fails");
  assert.equal(pathA_ComputeCount, 2, "Path A should recompute for new value");

  // --- Test Merging ---
  console.log("Testing merge node...");
  const mergeGraph = new Graph();
  const source1 = new TestNode<string>().setName("source1");
  const source2 = new TestNode<number>().setName("source2");
  const mergeNode = createMergeNode<string | number>(2);
  const mergeOutput = new TestNode<any[]>().setName("mergeOutput");
  mergeGraph.addNode(source1).addNode(source2).addNode(mergeNode).addNode(mergeOutput);
  mergeGraph.connect(source1, mergeNode);
  mergeGraph.connect(source2, mergeNode);
  mergeGraph.connect(mergeNode, mergeOutput);

  // Execute sources in parallel to trigger merge
  await Promise.all([
    mergeGraph.execute("hello", source1.id),
    mergeGraph.execute(123, source2.id)
  ]);
  // The order is not guaranteed, so we check for content
  assert.equal(mergeOutput.receivedInputs.length, 1);
  assert.deepStrictEqual(mergeOutput.receivedInputs[0].sort(), ["hello", 123].sort());


  // --- Test Feedback Loop ---
  console.log("Testing feedback loop...");
  const feedbackGraph = new Graph();
  const feedbackInput = new AdaptiveNode<[boolean, number], any>(([pass, data]) => pass ? data : null).setName("feedbackInput");
  const fbGate = createGateNode();
  const fbProcessor = createProcessor<number, number>((n) => n - 1, "fbProcessor");
  const fbDelay = createDelayNode(10); // shorter delay for faster test
  const fbOutput = new TestNode<number>().setName("fbOutput");
  const fbTransformer = createProcessor<number, [boolean, number]>(
    (n: number) => [n > 1, n],
    "fbTransformer"
  );
  feedbackGraph.addNode(feedbackInput).addNode(fbGate).addNode(fbProcessor).addNode(fbDelay).addNode(fbOutput).addNode(fbTransformer);

  feedbackGraph.connect(feedbackInput, fbGate as AdaptiveNode<any, any>);
  feedbackGraph.connect(fbGate as AdaptiveNode<any, any>, fbProcessor);
  feedbackGraph.connect(fbProcessor, fbDelay);
  feedbackGraph.connect(fbDelay, fbOutput);
  feedbackGraph.connect(fbDelay, fbTransformer); // Cycle
  feedbackGraph.connect(fbTransformer, fbGate as AdaptiveNode<any, any>);

  await feedbackGraph.execute([true, 5], feedbackInput.id);
  await new Promise(resolve => setTimeout(resolve, 100)); // Wait for loop
  feedbackGraph.stop();

  fbOutput.assertReceived([4, 3, 2, 1]);

  console.log("Complex graph executed successfully.");
};

// ============================================================================
// Run all tests
// ============================================================================

runAllTests().catch((err) => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
