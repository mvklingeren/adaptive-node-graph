import {
  AdaptiveNode,
  Graph,
  SubGraphNode,
  TestNode,
  createAddNode,
  createCacheNode,
  createDelayNode,
  createGateNode,
  createLoadBalancerNode,
  createMergeNode,
  createProcessor,
  createSplitNode
} from "./chunks/chunk-KMFS2BJO.js";

// src/test-complex.ts
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
Complex tests complete: ${pass} passed, ${fail} failed.`);
  if (fail > 0) {
    process.exit(1);
  }
}
tests["Complex Graph with Cycles and Advanced Features"] = async () => {
  const graph = new Graph();
  const finalOutput = new TestNode().setName("finalOutput");
  const errorOutput = new TestNode().setName("errorOutput");
  const pathA_Output = new TestNode().setName("pathA_Output");
  const pathB_Output = new TestNode().setName("pathB_Output");
  const feedback_Output = new TestNode().setName("feedback_Output");
  const input = new AdaptiveNode((n) => n).setName("input");
  const splitter = createSplitNode(3);
  let pathA_ComputeCount = 0;
  const pathA_processorFn = async (n) => {
    pathA_ComputeCount++;
    await new Promise((resolve) => setTimeout(resolve, 50));
    return n * 2;
  };
  const pathA_Processor = createProcessor(pathA_processorFn, "pathA_Processor");
  const pathA_Cache = createCacheNode(pathA_processorFn, { ttl: 5e3 });
  const pathA_Add = createProcessor((n) => n + 5, "pathA_Add");
  const subGraph = new Graph();
  const sub_add = new AdaptiveNode((n) => [n, 5]);
  const sub_adder = createAddNode();
  subGraph.addNode(sub_add).addNode(sub_adder).connect(sub_add, sub_adder);
  const subGraphNode = new SubGraphNode(subGraph, sub_add.id, sub_adder.id).setName("subGraph");
  const worker1 = createProcessor((n) => `w1:${n}`, "worker1");
  const worker2 = createProcessor((n) => `w2:${n}`, "worker2");
  const worker3 = createProcessor((n) => {
    throw new Error("Worker 3 fails");
  }, "worker3");
  const loadBalancer = createLoadBalancerNode([worker1, worker2, worker3], { strategy: "round-robin" });
  const feedbackGate = createGateNode();
  const feedbackDelay = createDelayNode(100);
  const feedbackProcessor = createProcessor((n) => n - 1, "feedbackProcessor");
  const merger = createMergeNode(2);
  const finalProcessor = createProcessor((arr) => arr.sort(), "finalProcessor");
  graph.addNode(input).addNode(splitter).addNode(pathA_Cache).addNode(pathA_Add).addNode(pathA_Output).addNode(subGraphNode).addNode(loadBalancer).addNode(pathB_Output).addNode(feedbackGate).addNode(feedbackDelay).addNode(feedbackProcessor).addNode(feedback_Output).addNode(merger).addNode(finalProcessor).addNode(finalOutput).addNode(errorOutput);
  const feedbackTransformer = createProcessor(
    (n) => [n > 1, n],
    "feedbackTransformer"
  );
  graph.addNode(feedbackTransformer);
  graph.connect(input, splitter);
  graph.connect(splitter, pathA_Cache, void 0, 0);
  graph.connect(pathA_Cache, pathA_Add);
  graph.connect(pathA_Add, pathA_Output);
  graph.connect(splitter, subGraphNode, void 0, 1);
  graph.connect(subGraphNode, loadBalancer);
  graph.connect(loadBalancer, pathB_Output);
  graph.connectError(loadBalancer, errorOutput);
  graph.connect(splitter, feedbackGate, void 0, 2);
  graph.connect(feedbackGate, feedbackProcessor);
  graph.connect(feedbackProcessor, feedbackDelay);
  graph.connect(feedbackDelay, feedback_Output);
  graph.connect(feedbackDelay, feedbackTransformer);
  graph.connect(feedbackTransformer, feedbackGate);
  console.log("Testing linear paths A and B...");
  await graph.execute(10, input.id);
  pathA_Output.assertReceived([25]);
  pathB_Output.assertReceived(["w1:15"]);
  assert.equal(pathA_ComputeCount, 1, "Path A should compute once");
  pathA_Output.reset();
  pathB_Output.reset();
  await graph.execute(10, input.id);
  pathA_Output.assertReceived([25]);
  pathB_Output.assertReceived(["w2:15"]);
  assert.equal(pathA_ComputeCount, 1, "Path A cache was not used");
  pathB_Output.reset();
  await graph.execute(12, input.id);
  pathB_Output.assertReceived([]);
  assert.equal(errorOutput.receivedInputs.length, 1, "Worker error not caught");
  assert.equal(errorOutput.receivedInputs[0].error.message, "Worker 3 fails");
  assert.equal(pathA_ComputeCount, 2, "Path A should recompute for new value");
  console.log("Testing merge node...");
  const mergeGraph = new Graph();
  const source1 = new TestNode().setName("source1");
  const source2 = new TestNode().setName("source2");
  const mergeNode = createMergeNode(2);
  const mergeOutput = new TestNode().setName("mergeOutput");
  mergeGraph.addNode(source1).addNode(source2).addNode(mergeNode).addNode(mergeOutput);
  mergeGraph.connect(source1, mergeNode);
  mergeGraph.connect(source2, mergeNode);
  mergeGraph.connect(mergeNode, mergeOutput);
  await Promise.all([
    mergeGraph.execute("hello", source1.id),
    mergeGraph.execute(123, source2.id)
  ]);
  assert.equal(mergeOutput.receivedInputs.length, 1);
  assert.deepStrictEqual(mergeOutput.receivedInputs[0].sort(), ["hello", 123].sort());
  console.log("Testing feedback loop...");
  const feedbackGraph = new Graph();
  const feedbackInput = new AdaptiveNode(([pass, data]) => pass ? data : null).setName("feedbackInput");
  const fbGate = createGateNode();
  const fbProcessor = createProcessor((n) => n - 1, "fbProcessor");
  const fbDelay = createDelayNode(10);
  const fbOutput = new TestNode().setName("fbOutput");
  const fbTransformer = createProcessor(
    (n) => [n > 1, n],
    "fbTransformer"
  );
  feedbackGraph.addNode(feedbackInput).addNode(fbGate).addNode(fbProcessor).addNode(fbDelay).addNode(fbOutput).addNode(fbTransformer);
  feedbackGraph.connect(feedbackInput, fbGate);
  feedbackGraph.connect(fbGate, fbProcessor);
  feedbackGraph.connect(fbProcessor, fbDelay);
  feedbackGraph.connect(fbDelay, fbOutput);
  feedbackGraph.connect(fbDelay, fbTransformer);
  feedbackGraph.connect(fbTransformer, fbGate);
  await feedbackGraph.execute([true, 5], feedbackInput.id);
  await new Promise((resolve) => setTimeout(resolve, 100));
  feedbackGraph.stop();
  fbOutput.assertReceived([4, 3, 2, 1]);
  console.log("Complex graph executed successfully.");
};
runAllTests().catch((err) => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
