import {
  AdaptiveNode,
  Graph,
  OscillatorNode,
  TestNode,
  createFloat32MultiplyNode,
  createLoadBalancerNode,
  createProcessor,
  testGraph
} from "./chunks/chunk-KMFS2BJO.js";

// src/test-basics.ts
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
tests["Basic Math Graph"] = async () => {
  const graph = new Graph();
  const input = new TestNode().setName("input");
  const add5 = createProcessor((x) => x + 5, "add5");
  const multiply2 = createProcessor((x) => x * 2, "multiply2");
  const output = new TestNode().setName("output");
  graph.addNode(input).addNode(add5).addNode(multiply2).addNode(output);
  graph.connect(input, add5);
  graph.connect(add5, multiply2);
  graph.connect(multiply2, output);
  await graph.execute(10, input.id);
  output.assertReceived([30]);
  output.assertNoErrors();
};
tests["Conditional Routing"] = async () => {
  const graph = new Graph();
  const tempSensor = createProcessor((input) => input ?? 42, "temp-sensor");
  const alertNode = new TestNode().setName("alert");
  const normalNode = new TestNode().setName("normal");
  const router = new AdaptiveNode(async (temp) => {
    if (temp > 30) {
      await alertNode.process(temp);
    } else {
      await normalNode.process(temp);
    }
    return temp;
  }).setName("router");
  graph.addNode(tempSensor).addNode(router).addNode(alertNode).addNode(normalNode);
  graph.connect(tempSensor, router);
  await graph.execute(null, tempSensor.id);
  alertNode.assertReceived([42]);
  normalNode.assertReceived([]);
  alertNode.assertNoErrors();
  alertNode.reset();
  normalNode.reset();
  tempSensor.setInitialValue(25);
  await graph.execute(null, tempSensor.id);
  alertNode.assertReceived([]);
  normalNode.assertReceived([25]);
  normalNode.assertNoErrors();
};
tests["Error Handling with Recovery"] = async () => {
  const graph = new Graph();
  const failingNode = new AdaptiveNode(() => {
    throw new Error("Processing failed");
  }).setName("failingNode");
  const errorCapture = new TestNode().setName("errorCapture");
  graph.addNode(failingNode).addNode(errorCapture);
  graph.connectError(failingNode, errorCapture);
  await graph.execute("test-input", failingNode.id);
  assert.equal(errorCapture.receivedInputs.length, 1, "Error should be captured");
  const error = errorCapture.receivedInputs[0];
  assert.equal(error.error.message, "Processing failed");
  assert.equal(error.input, "test-input");
  assert.equal(error.nodeId, failingNode.id);
};
tests["Machine Learning Pipeline"] = async () => {
  const graph = new Graph();
  const preprocessor = new AdaptiveNode((data) => new Float32Array(data)).setName("preprocessor");
  const model = new AdaptiveNode(
    (features) => features.reduce((a, b) => a + b, 0) > 10 ? "positive" : "negative"
  ).setName("model");
  const output = new TestNode().setName("output");
  graph.addNode(preprocessor).addNode(model).addNode(output);
  graph.connect(preprocessor, model);
  graph.connect(model, output);
  await graph.execute([5, 6], preprocessor.id);
  output.assertReceived(["positive"]);
  output.reset();
  await graph.execute([1, 2], preprocessor.id);
  output.assertReceived(["negative"]);
  output.assertNoErrors();
};
tests["Multi-protocol Gateway"] = async () => {
  const graph = new Graph();
  const gateway = new AdaptiveNode((req) => ({ status: 400, error: "Bad format" })).register(
    (req) => "method" in req,
    (req) => ({ status: 200, data: `HTTP ${req.method}` })
  ).register(
    (req) => req.type === "ws",
    (req) => ({ status: "ok", data: "WebSocket" })
  ).setName("gateway");
  const output = new TestNode().setName("output");
  graph.addNode(gateway).addNode(output);
  graph.connect(gateway, output);
  await graph.execute({ method: "GET" }, gateway.id);
  output.assertReceived([{ status: 200, data: "HTTP GET" }]);
  output.reset();
  await graph.execute({ type: "ws" }, gateway.id);
  output.assertReceived([{ status: "ok", data: "WebSocket" }]);
  output.assertNoErrors();
};
tests["Oscillator Audio Chain"] = async () => {
  const graph = new Graph();
  const osc = new OscillatorNode();
  const multiply = createFloat32MultiplyNode();
  const extractor = createProcessor(
    (data) => data.samples,
    "extractor"
  );
  const output = new TestNode().setName("output");
  graph.addNode(osc).addNode(extractor).addNode(multiply).addNode(output);
  graph.connect(osc, extractor);
  graph.connect(extractor, multiply);
  graph.connect(multiply, output);
  const params = {
    frequency: 440,
    amplitude: 1,
    // Use 1.0 for easier assertion
    sampleRate: 44100,
    length: 4,
    waveform: "sine"
  };
  await graph.execute(params, osc.id);
  assert.equal(output.receivedInputs.length, 1, "Should receive one output");
  const result = output.receivedInputs[0];
  assert(result instanceof Float32Array, "Output should be Float32Array");
  assert.equal(result.length, 4, "Output length should be correct");
  assert(result.every((val) => Math.abs(val) <= 0.5), "Values should be scaled down");
  output.assertNoErrors();
};
tests["Real-time Stream Load Balancer"] = async () => {
  const graph = new Graph();
  const worker1 = new TestNode().setName("worker1");
  const worker2 = new TestNode().setName("worker2");
  const loadBalancer = createLoadBalancerNode([worker1, worker2], { strategy: "round-robin" });
  graph.addNode(loadBalancer).addNode(worker1).addNode(worker2);
  loadBalancer.setInitialValue({ data: 1 });
  await graph.execute(null, loadBalancer.id);
  loadBalancer.setInitialValue({ data: 2 });
  await graph.execute(null, loadBalancer.id);
  loadBalancer.setInitialValue({ data: 3 });
  await graph.execute(null, loadBalancer.id);
  loadBalancer.setInitialValue({ data: 4 });
  await graph.execute(null, loadBalancer.id);
  worker1.assertReceived([{ data: 1 }, { data: 3 }]);
  worker2.assertReceived([{ data: 2 }, { data: 4 }]);
  worker1.assertNoErrors();
  worker2.assertNoErrors();
};
tests["Transformation Pipeline"] = async () => {
  const validator = createProcessor((user) => {
    if (!user.email.includes("@")) throw new Error("Invalid email");
    return user;
  }, "validator");
  const enricher = createProcessor(
    (user) => ({ ...user, category: user.age < 18 ? "minor" : "adult" }),
    "enricher"
  );
  const privacyFilter = createProcessor(
    (user) => ({ id: user.id, age: user.age, category: user.category }),
    "privacyFilter"
  );
  const graph = new Graph().addNode(validator).addNode(enricher).addNode(privacyFilter);
  graph.connect(validator, enricher);
  graph.connect(enricher, privacyFilter);
  await testGraph(
    graph,
    { id: 1, name: "Test", email: "test@test.com", age: 25 },
    { id: 1, age: 25, category: "adult" },
    validator.id
  );
};
tests["Type Adaptive Node"] = async () => {
  const smartProcessor = new AdaptiveNode((input) => `Unknown: ${typeof input}`).register((input) => typeof input === "number", (num) => `Number: ${num}`).register((input) => typeof input === "string", (str) => `String: ${str}`).setName("smart-processor");
  const output = new TestNode().setName("output");
  const graph = new Graph();
  graph.addNode(smartProcessor);
  graph.addNode(output);
  graph.connect(smartProcessor, output);
  await graph.execute(123, smartProcessor.id);
  output.assertReceived(["Number: 123"]);
  output.reset();
  await graph.execute("hello", smartProcessor.id);
  output.assertReceived(["String: hello"]);
  output.assertNoErrors();
};
runAllTests().catch((err) => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
