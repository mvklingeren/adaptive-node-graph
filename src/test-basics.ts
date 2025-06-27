// test-basics.ts
// Contains all the basic tests from the /examples directory, refactored into
// a verifiable test suite using TestNode and graph.execute().

import {
  Graph,
  AdaptiveNode,
  TestNode,
  OscillatorNode,
  createProcessor,
  createLoadBalancerNode,
  createFloat32MultiplyNode,
  testGraph,
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
    process.exit(1); // Exit with error code if any test fails
  }
}

// ============================================================================
// 1. Basic Math Test
// ============================================================================

tests["Basic Math Graph"] = async () => {
  const graph = new Graph();

  // Create nodes
  const input = new TestNode<number>().setName("input");
  const add5 = createProcessor<number, number>((x) => x + 5, "add5");
  const multiply2 = createProcessor<number, number>((x) => x * 2, "multiply2");
  const output = new TestNode<number>().setName("output");

  // Build graph: (input + 5) * 2
  graph.addNode(input).addNode(add5).addNode(multiply2).addNode(output);

  graph.connect(input, add5);
  graph.connect(add5, multiply2);
  graph.connect(multiply2, output);

  // Execute and assert
  await graph.execute(10, input.id);

  output.assertReceived([30]);
  output.assertNoErrors();
};

// ============================================================================
// 2. Conditional Routing Test
// ============================================================================

tests["Conditional Routing"] = async () => {
  const graph = new Graph();

  // Nodes
  const tempSensor = createProcessor(() => 42, "temp-sensor"); // Fixed temp for predictability
  const alertNode = new TestNode<number>().setName("alert");
  const normalNode = new TestNode<number>().setName("normal");

  // Custom router node to direct traffic
  const router = new AdaptiveNode<number, void>((temp) => {
    if (temp > 30) {
      alertNode.process(temp);
    } else {
      normalNode.process(temp);
    }
  }).setName("router");

  graph.addNode(tempSensor).addNode(router).addNode(alertNode).addNode(normalNode);
  graph.connect(tempSensor, router);

  // Test high temperature
  await graph.execute(null, tempSensor.id);
  alertNode.assertReceived([42]);
  normalNode.assertReceived([]);
  alertNode.assertNoErrors();

  // Test normal temperature
  alertNode.reset();
  normalNode.reset();
  tempSensor.setInitialValue(25); // Change sensor value
  await graph.execute(null, tempSensor.id);
  alertNode.assertReceived([]);
  normalNode.assertReceived([25]);
  normalNode.assertNoErrors();
};

// ============================================================================
// 3. Error Handling Test
// ============================================================================

tests["Error Handling with Recovery"] = async () => {
  const graph = new Graph();

  const failingNode = new AdaptiveNode<any, any>(() => {
    throw new Error("Processing failed");
  }).setName("failingNode");

  const errorCapture = new TestNode<any>().setName("errorCapture");
  graph.addNode(failingNode).addNode(errorCapture);
  graph.connectError(failingNode, errorCapture);

  await graph.execute("test-input", failingNode.id);

  assert.equal(errorCapture.receivedInputs.length, 1, "Error should be captured");
  const error = errorCapture.receivedInputs[0];
  assert.equal(error.error.message, "Processing failed");
  assert.equal(error.input, "test-input");
  assert.equal(error.nodeId, failingNode.id);
};

// ============================================================================
// 4. Machine Learning Pipeline Test
// ============================================================================

tests["Machine Learning Pipeline"] = async () => {
  const graph = new Graph();

  // Simplified ML nodes for testing
  const preprocessor = new AdaptiveNode<number[], Float32Array>((data) => new Float32Array(data))
    .setName("preprocessor");
  
  const model = new AdaptiveNode<Float32Array, string>((features) =>
    features.reduce((a, b) => a + b, 0) > 10 ? "positive" : "negative"
  ).setName("model");

  const output = new TestNode<string>().setName("output");

  graph.addNode(preprocessor).addNode(model).addNode(output);
  graph.connect(preprocessor, model);
  graph.connect(model, output);

  // Test positive case
  await graph.execute([5, 6], preprocessor.id);
  output.assertReceived(["positive"]);

  // Test negative case
  output.reset();
  await graph.execute([1, 2], preprocessor.id);
  output.assertReceived(["negative"]);
  output.assertNoErrors();
};

// ============================================================================
// 5. Multi-protocol Gateway Test
// ============================================================================

tests["Multi-protocol Gateway"] = async () => {
  const graph = new Graph();

  // Simplified gateway
  const gateway = new AdaptiveNode<any, any>((req) => ({ status: 400, error: "Bad format" }))
    .register(
      (req): req is { method: string } => "method" in req,
      (req) => ({ status: 200, data: `HTTP ${req.method}` })
    )
    .register(
      (req): req is { type: "ws" } => req.type === "ws",
      (req) => ({ status: "ok", data: "WebSocket" })
    )
    .setName("gateway");

  const output = new TestNode<any>().setName("output");
  graph.addNode(gateway).addNode(output);
  graph.connect(gateway, output);

  // Test HTTP
  await graph.execute({ method: "GET" }, gateway.id);
  output.assertReceived([{ status: 200, data: "HTTP GET" }]);

  // Test WebSocket
  output.reset();
  await graph.execute({ type: "ws" }, gateway.id);
  output.assertReceived([{ status: "ok", data: "WebSocket" }]);
  output.assertNoErrors();
};

// ============================================================================
// 6. Oscillator and Audio Chain Test
// ============================================================================

tests["Oscillator Audio Chain"] = async () => {
  const graph = new Graph();

  const osc = new OscillatorNode();
  const multiply = createFloat32MultiplyNode(); // Multiplies by 0.5
  const extractor = createProcessor<{ samples: Float32Array; nextPhase: number }, Float32Array>(
    (data) => data.samples,
    "extractor"
  );
  const output = new TestNode<Float32Array>().setName("output");

  graph.addNode(osc).addNode(extractor).addNode(multiply).addNode(output);
  graph.connect(osc, extractor);
  graph.connect(extractor, multiply);
  graph.connect(multiply, output);

  const params = {
    frequency: 440,
    amplitude: 1.0, // Use 1.0 for easier assertion
    sampleRate: 44100,
    length: 4,
    waveform: "sine" as const,
  };

  await graph.execute(params, osc.id);

  assert.equal(output.receivedInputs.length, 1, "Should receive one output");
  const result = output.receivedInputs[0];
  assert(result instanceof Float32Array, "Output should be Float32Array");
  assert.equal(result.length, 4, "Output length should be correct");
  // Check if values are multiplied by 0.5
  assert(result.every(val => Math.abs(val) <= 0.5), "Values should be scaled down");
  output.assertNoErrors();
};

// ============================================================================
// 7. Real-time Stream Load Balancer Test
// ============================================================================

tests["Real-time Stream Load Balancer"] = async () => {
  const graph = new Graph();

  const worker1 = new TestNode<any>().setName("worker1");
  const worker2 = new TestNode<any>().setName("worker2");
  const loadBalancer = createLoadBalancerNode([worker1, worker2], { strategy: "round-robin" });
  
  graph.addNode(loadBalancer).addNode(worker1).addNode(worker2);

  // The load balancer itself is the entry point
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

// ============================================================================
// 8. Transformation Pipeline Test
// ============================================================================

tests["Transformation Pipeline"] = async () => {
  interface User { id: number; name: string; email: string; age: number; }
  interface EnrichedUser extends User { category: string; }

  const validator = createProcessor<User, User>((user) => {
    if (!user.email.includes("@")) throw new Error("Invalid email");
    return user;
  }, "validator");

  const enricher = createProcessor<User, EnrichedUser>(
    (user) => ({ ...user, category: user.age < 18 ? "minor" : "adult" }),
    "enricher"
  );

  const privacyFilter = createProcessor<EnrichedUser, Partial<EnrichedUser>>(
    (user) => ({ id: user.id, age: user.age, category: user.category }),
    "privacyFilter"
  );

  const graph = new Graph()
    .addNode(validator)
    .addNode(enricher)
    .addNode(privacyFilter);
  
  graph.connect(validator, enricher);
  graph.connect(enricher, privacyFilter);

  await testGraph(
    graph,
    { id: 1, name: "Test", email: "test@test.com", age: 25 },
    { id: 1, age: 25, category: "adult" },
    validator.id
  );
};

// ============================================================================
// 9. Type Adaptive Node Test
// ============================================================================

tests["Type Adaptive Node"] = async () => {
  const smartProcessor = new AdaptiveNode<any, string>((input) => `Unknown: ${typeof input}`)
    .register((input): input is number => typeof input === 'number', (num) => `Number: ${num}`)
    .register((input): input is string => typeof input === 'string', (str) => `String: ${str}`)
    .setName("smart-processor");

  const output = new TestNode<string>().setName("output");
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


// ============================================================================
// Run all tests
// ============================================================================

runAllTests().catch(err => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
