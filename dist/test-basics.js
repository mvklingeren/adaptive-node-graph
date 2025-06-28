// src/core.ts
import pino from "pino";
var HandledNodeError = class extends Error {
  constructor(message) {
    super(message);
    this.name = "HandledNodeError";
  }
};
var AdaptiveNode = class {
  constructor(defaultProcessor, options = {}) {
    this.defaultProcessor = defaultProcessor;
    this.options = options;
    this.maxConcurrent = options.maxConcurrent || 10;
    this.circuitBreakerThreshold = options.circuitBreakerThreshold || 5;
    this.circuitBreakerResetTime = options.circuitBreakerResetTime || 6e4;
    this.id = `node_${crypto.randomUUID()}`;
    this.name = this.constructor.name;
    this.setupInOut();
  }
  processors = /* @__PURE__ */ new Map();
  performanceStats = /* @__PURE__ */ new Map();
  // Flow control
  processing = 0;
  maxConcurrent = 10;
  queue = [];
  // Error handling
  errorCount = 0;
  lastErrorTime = 0;
  circuitBreakerThreshold = 5;
  circuitBreakerResetTime = 6e4;
  // 1 minute
  isCircuitOpen = false;
  // Store last result for sub-graphs
  lastResult = null;
  isEmitting = true;
  onProcess = null;
  initialValue = null;
  id;
  name;
  inlets = [];
  outlets = [];
  // Type markers for compile-time type checking
  inputType;
  outputType;
  setupInOut() {
    this.inlets = [{ accept: (data) => this.process(data) }];
    const dataOutletConnections = [];
    const errorOutletConnections = [];
    this.outlets = [
      {
        send: async (data, graph, hooks) => {
          const promises = dataOutletConnections.map(
            (conn) => conn.transfer(data, graph, hooks)
          );
          await Promise.all(promises);
        },
        connections: dataOutletConnections
      },
      {
        send: async (error, graph, hooks) => {
          const promises = errorOutletConnections.map(
            (conn) => conn.transfer(error, graph, hooks)
          );
          await Promise.all(promises);
        },
        connections: errorOutletConnections
      }
    ];
  }
  register(predicate, processor) {
    this.processors.set(predicate, processor);
    return this;
  }
  async process(input, graph, hooks) {
    if (graph?.isStopped) {
      return null;
    }
    hooks?.onNodeStart?.(this.id);
    if (this.isEmitting && this.onProcess) {
      this.onProcess(this.id, input);
    }
    if (input === null && this.initialValue !== null) {
      input = this.initialValue;
    }
    if (this.isCircuitOpen) {
      const now = Date.now();
      if (now - this.lastErrorTime > this.circuitBreakerResetTime) {
        this.isCircuitOpen = false;
        this.errorCount = 0;
      } else {
        const error = new Error("Circuit breaker is open");
        this.handleError(error, input, graph, hooks);
        throw error;
      }
    }
    if (this.processing >= this.maxConcurrent) {
      await new Promise((resolve) => this.queue.push(resolve));
    }
    this.processing++;
    try {
      const result = await this.executeProcessor(input, graph, hooks);
      if (this.outlets[0]) {
        await this.outlets[0].send(result, graph, hooks);
      }
      hooks?.onNodeComplete?.(this.id, result);
      return result;
    } catch (error) {
      this.handleError(error, input, graph, hooks);
      throw error;
    } finally {
      this.processing--;
      if (this.queue.length > 0) {
        this.queue.shift()?.();
      }
    }
  }
  async executeProcessor(input, _graph, hooks) {
    const start = performance.now();
    let processorName = "default";
    let selectedProcessor = this.defaultProcessor;
    processorName = "default";
    for (const [predicate, processor] of this.processors) {
      if (predicate(input)) {
        selectedProcessor = processor;
        processorName = `predicate:${predicate.toString()}`;
        break;
      }
    }
    const result = await selectedProcessor(input);
    this.recordPerformance(processorName, performance.now() - start);
    this.lastResult = result;
    this.errorCount = 0;
    return result;
  }
  handleError(error, input, graph, hooks) {
    this.errorCount++;
    this.lastErrorTime = Date.now();
    if (this.errorCount >= this.circuitBreakerThreshold) {
      this.isCircuitOpen = true;
      const logger = hooks?.logger || pino();
      logger.error(
        { nodeId: this.id, errorCount: this.errorCount },
        `Circuit breaker opened for node ${this.id} after ${this.errorCount} errors`
      );
    }
    this.sendError(error, input, graph, hooks);
  }
  sendError(error, input, graph, hooks) {
    const nodeError = {
      error,
      input,
      nodeId: this.id,
      timestamp: Date.now()
    };
    if (this.outlets[1]?.connections.length > 0) {
      this.outlets[1].send(nodeError, graph, hooks);
    } else {
      const logger = hooks?.logger || pino();
      logger.error({ error: nodeError }, `Unhandled error in node ${this.id}`);
    }
  }
  recordPerformance(processorName, duration) {
    if (!this.performanceStats.has(processorName)) {
      this.performanceStats.set(processorName, []);
    }
    const stats = this.performanceStats.get(processorName);
    stats.push(duration);
    if (stats.length > 100) {
      stats.shift();
    }
  }
  getPerformanceStats() {
    const result = /* @__PURE__ */ new Map();
    for (const [name, durations] of this.performanceStats) {
      const avg = durations.reduce((a, b) => a + b, 0) / durations.length;
      const min = Math.min(...durations);
      const max = Math.max(...durations);
      result.set(name, { avg, min, max });
    }
    return result;
  }
  getLastResult() {
    return this.lastResult;
  }
  setName(name) {
    this.name = name;
    return this;
  }
  setInitialValue(value) {
    this.initialValue = value;
    return this;
  }
  setEmitting(isEmitting, onProcess) {
    this.isEmitting = isEmitting;
    if (onProcess) {
      this.onProcess = onProcess;
    }
    return this;
  }
  // New methods for flow control
  setMaxConcurrent(max) {
    this.maxConcurrent = max;
    return this;
  }
  resetCircuitBreaker() {
    this.isCircuitOpen = false;
    this.errorCount = 0;
    return this;
  }
  /**
   * Cleans up any resources used by the node, like timers or intervals.
   * Override in subclasses for custom cleanup logic.
   */
  destroy() {
  }
};
var Connection = class {
  constructor(source, sourceOutlet, target, targetInlet, transformer) {
    this.source = source;
    this.sourceOutlet = sourceOutlet;
    this.target = target;
    this.targetInlet = targetInlet;
    this.transformer = transformer;
    this.source.outlets[this.sourceOutlet].connections.push(this);
  }
  async transfer(data, graph, hooks) {
    try {
      const transformedData = this.transformer ? await this.transformer(data) : data;
      await this.target.process(transformedData, graph, hooks);
    } catch (error) {
      if (error instanceof HandledNodeError) {
        throw error;
      }
      const connError = {
        error,
        input: data,
        nodeId: `connection-${this.source.id}-${this.target.id}`,
        timestamp: Date.now()
      };
      const logger = hooks?.logger || pino();
      logger.error(
        {
          sourceNode: this.source.id,
          targetNode: this.target.id,
          error: connError.error
        },
        `Error in connection from ${this.source.id} to ${this.target.id}`
      );
      this.target.outlets[1]?.send(connError, graph, hooks);
    }
  }
  disconnect() {
    const outlet = this.source.outlets[this.sourceOutlet];
    if (outlet) {
      const index = outlet.connections.indexOf(this);
      if (index > -1) {
        outlet.connections.splice(index, 1);
      }
    }
  }
};
var Graph = class {
  nodes = /* @__PURE__ */ new Map();
  connections = /* @__PURE__ */ new Set();
  executionOrder = [];
  _isDirty = true;
  isStopped = false;
  addNode(node) {
    this.nodes.set(node.id, node);
    this._isDirty = true;
    return this;
  }
  removeNode(nodeId) {
    const node = this.nodes.get(nodeId);
    if (!node) return this;
    node.destroy();
    for (const conn of this.connections) {
      if (conn.source.id === nodeId || conn.target.id === nodeId) {
        conn.disconnect();
        this.connections.delete(conn);
      }
    }
    this.nodes.delete(nodeId);
    this._isDirty = true;
    return this;
  }
  getNode(nodeId) {
    return this.nodes.get(nodeId);
  }
  // Type-safe connect method
  connect(source, target, transformer, sourceOutlet = 0, targetInlet = 0) {
    const connection = new Connection(
      source,
      sourceOutlet,
      target,
      targetInlet,
      transformer
    );
    this.connections.add(connection);
    this._isDirty = true;
    return connection;
  }
  // Connect error outlets
  connectError(source, errorHandler) {
    const connection = new Connection(
      source,
      1,
      // error outlet
      errorHandler,
      0,
      // default inlet
      void 0
    );
    this.connections.add(connection);
    this._isDirty = true;
    return connection;
  }
  disconnect(connection) {
    connection.disconnect();
    this.connections.delete(connection);
    this._isDirty = true;
    return this;
  }
  getExecutionOrder(hooks) {
    if (this._isDirty) {
      this.updateExecutionOrder(hooks);
    }
    return this.executionOrder;
  }
  updateExecutionOrder(hooks) {
    const logger = hooks?.logger || pino();
    const order = [];
    const visiting = /* @__PURE__ */ new Set();
    const visited = /* @__PURE__ */ new Set();
    const visit = (node) => {
      if (visited.has(node.id)) {
        return;
      }
      if (visiting.has(node.id)) {
        logger.warn(
          { nodeId: node.id },
          `Cycle detected in graph involving node ${node.id}`
        );
      } else {
        visiting.add(node.id);
        for (const conn of this.connections) {
          if (conn.target.id === node.id && conn.sourceOutlet === 0) {
            const sourceNode = this.nodes.get(conn.source.id);
            if (sourceNode) {
              visit(sourceNode);
            }
          }
        }
        visiting.delete(node.id);
      }
      visiting.add(node.id);
      for (const conn of this.connections) {
        if (conn.target.id === node.id && conn.sourceOutlet === 0) {
          const sourceNode = this.nodes.get(conn.source.id);
          if (sourceNode) {
            visit(sourceNode);
          }
        }
      }
      visiting.delete(node.id);
      visited.add(node.id);
      order.push(node);
    };
    for (const node of this.nodes.values()) {
      if (!visited.has(node.id)) {
        visit(node);
      }
    }
    this.executionOrder = order;
    this._isDirty = false;
  }
  async execute(input, startNodeId, hooks) {
    this.isStopped = false;
    const executionOrder = this.getExecutionOrder(hooks);
    const processNode = async (node, nodeInput) => {
      try {
        await node.process(nodeInput, this, hooks);
      } catch (error) {
        const logger = hooks?.logger || pino();
        logger.warn(
          { err: error, nodeId: node.id },
          `Caught error during graph execution at node ${node.id}. Execution continues.`
        );
        throw error;
      }
    };
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Start node ${startNodeId} not found`);
      const nodeInput = input instanceof Map ? input.get(startNodeId) : input;
      try {
        await processNode(startNode, nodeInput);
      } catch (e) {
      }
      const lastNode2 = executionOrder[executionOrder.length - 1];
      return lastNode2?.getLastResult();
    }
    const entryNodes = executionOrder.filter((node) => {
      const hasIncomingConnection = Array.from(this.connections).some(
        (conn) => conn.target.id === node.id && conn.sourceOutlet === 0
      );
      const hasInitialValue = node.initialValue !== null;
      return !hasIncomingConnection || hasInitialValue;
    });
    if (entryNodes.length === 0 && this.nodes.size > 0) {
      const firstNode = executionOrder[0];
      if (firstNode) {
        entryNodes.push(firstNode);
      } else {
        throw new Error(
          "Graph execution failed: No entry nodes found and execution order is empty."
        );
      }
    }
    if (entryNodes.length > 1 && !(input instanceof Map)) {
      throw new Error(
        `Multiple entry nodes found (${entryNodes.map((n) => n.id).join(
          ", "
        )}). Please specify a startNodeId or provide a Map of inputs.`
      );
    }
    const promises = entryNodes.map(async (node) => {
      const nodeInput = input instanceof Map ? input.get(node.id) : input;
      await processNode(node, nodeInput);
    });
    const results = await Promise.allSettled(promises);
    results.forEach((result) => {
      if (result.status === "rejected") {
        const logger = hooks?.logger || pino();
        logger.error({ reason: result.reason }, "An error occurred in an execution branch.");
      }
    });
    const lastNode = executionOrder[executionOrder.length - 1];
    return lastNode?.getLastResult();
  }
  stop() {
    this.isStopped = true;
  }
  // Parallel execution for independent nodes
  async executeParallel(initialInputs, hooks) {
    this.isStopped = false;
    const results = /* @__PURE__ */ new Map();
    const dependencies = this.calculateDependencies();
    const inDegree = /* @__PURE__ */ new Map();
    const queue = [];
    for (const node of this.nodes.values()) {
      const deps = dependencies.get(node.id) || /* @__PURE__ */ new Set();
      inDegree.set(node.id, deps.size);
      if (deps.size === 0) {
        queue.push(node);
      }
    }
    let processedCount = 0;
    while (processedCount < this.nodes.size) {
      if (this.isStopped) break;
      if (queue.length === 0) {
        const logger = hooks?.logger || pino();
        logger.warn(
          {
            totalNodes: this.nodes.size,
            executedNodes: results.size
          },
          "Parallel execution stalled. A cycle may be present or graph is disconnected."
        );
        break;
      }
      const node = queue.shift();
      processedCount++;
      const nodeDependencies = dependencies.get(node.id) || /* @__PURE__ */ new Set();
      const aggregatedInputs = Array.from(nodeDependencies).map(
        (depId) => results.get(depId)
      );
      const input = initialInputs.get(node.id) ?? (aggregatedInputs.length === 1 ? aggregatedInputs[0] : aggregatedInputs);
      const result = await node.process(input, this, hooks);
      results.set(node.id, result);
      for (const conn of this.connections) {
        if (conn.source.id === node.id) {
          const targetId = conn.target.id;
          const currentInDegree = (inDegree.get(targetId) || 1) - 1;
          inDegree.set(targetId, currentInDegree);
          if (currentInDegree === 0) {
            const targetNode = this.nodes.get(targetId);
            if (targetNode) queue.push(targetNode);
          }
        }
      }
    }
    if (results.size < this.nodes.size && !this.isStopped) {
      const logger = hooks?.logger || pino();
      logger.warn(
        {
          totalNodes: this.nodes.size,
          executedNodes: results.size
        },
        "Parallel execution did not execute all nodes."
      );
    }
    return results;
  }
  calculateDependencies() {
    const deps = /* @__PURE__ */ new Map();
    for (const conn of this.connections) {
      if (conn.sourceOutlet === 0) {
        if (!deps.has(conn.target.id)) {
          deps.set(conn.target.id, /* @__PURE__ */ new Set());
        }
        deps.get(conn.target.id).add(conn.source.id);
      }
    }
    return deps;
  }
};
var TestNode = class extends AdaptiveNode {
  receivedInputs = [];
  processedOutputs = [];
  errors = [];
  constructor(processor) {
    const capturingProcessor = async (input) => {
      this.receivedInputs.push(input);
      const output = await (processor ? processor(input) : input);
      this.processedOutputs.push(output);
      return output;
    };
    super(capturingProcessor);
    this.setName("test");
    const errorCollector = new AdaptiveNode((error) => {
      this.errors.push(error);
    }).setName("testErrorCollector");
    new Connection(this, 1, errorCollector, 0);
  }
  reset() {
    this.receivedInputs = [];
    this.processedOutputs = [];
    this.errors = [];
  }
  assertReceived(expected) {
    if (JSON.stringify(this.receivedInputs) !== JSON.stringify(expected)) {
      throw new Error(
        `Expected inputs ${JSON.stringify(expected)}, got ${JSON.stringify(
          this.receivedInputs
        )}`
      );
    }
  }
  assertProcessed(expected) {
    if (JSON.stringify(this.processedOutputs) !== JSON.stringify(expected)) {
      throw new Error(
        `Expected outputs ${JSON.stringify(expected)}, got ${JSON.stringify(
          this.processedOutputs
        )}`
      );
    }
  }
  assertNoErrors() {
    if (this.errors.length > 0) {
      throw new Error(
        `Expected no errors, got ${this.errors.length}: ${this.errors.map((e) => e.error.message).join(", ")}`
      );
    }
  }
};
async function testGraph(graph, input, expectedOutput, startNodeId) {
  const output = await graph.execute(input, startNodeId);
  if (JSON.stringify(output) !== JSON.stringify(expectedOutput)) {
    throw new Error(
      `Expected output ${JSON.stringify(expectedOutput)}, got ${JSON.stringify(
        output
      )}`
    );
  }
}
var createFloat32MultiplyNode = () => new AdaptiveNode((input) => {
  const result = new Float32Array(input?.length);
  for (let i = 0; i < input?.length; i++) {
    result[i] = input[i] * 0.5;
  }
  return result;
}).setName("f32*");
var LoadBalancerNode = class extends AdaptiveNode {
  constructor(nodes, strategy = "round-robin", healthCheckInterval = 3e4, logger = pino()) {
    super(async () => null);
    this.nodes = nodes;
    this.strategy = strategy;
    this.logger = logger;
    this.nodeHealth = new Map(
      nodes.map((n) => [n.id, { healthy: true, lastCheck: 0 }])
    );
    this.healthCheckTimer = setInterval(() => {
      const now = Date.now();
      for (const [nodeId, health] of this.nodeHealth.entries()) {
        if (!health.healthy && now - health.lastCheck > healthCheckInterval) {
          this.logger.info(
            `Retrying health check for unhealthy node ${nodeId}`
          );
          health.healthy = true;
          health.lastCheck = now;
        }
      }
    }, healthCheckInterval);
  }
  index = 0;
  nodeHealth;
  healthCheckTimer;
  async process(input, graph, hooks) {
    hooks?.onNodeStart?.(this.id);
    const logger = hooks?.logger || this.logger;
    const healthyNodes = this.nodes.filter(
      (node) => this.nodeHealth.get(node.id)?.healthy
    );
    if (healthyNodes.length === 0) {
      const error = new Error("No healthy nodes available");
      this.sendError(error, input, graph, hooks);
      throw error;
    }
    let selectedNode;
    if (this.strategy === "round-robin") {
      this.index = (this.index || 0) % healthyNodes.length;
      selectedNode = healthyNodes[this.index];
      this.index++;
    } else {
      const randomIndex = Math.floor(Math.random() * healthyNodes.length);
      selectedNode = healthyNodes[randomIndex];
    }
    try {
      const result = await selectedNode.process(input, graph, hooks);
      await this.outlets[0].send(result, graph, hooks);
      hooks?.onNodeComplete?.(this.id, result);
      return result;
    } catch (error) {
      logger.warn(
        { err: error, nodeId: selectedNode.id },
        `Node ${selectedNode.id} failed in load balancer. Marking as unhealthy.`
      );
      const health = this.nodeHealth.get(selectedNode.id);
      if (health) {
        health.healthy = false;
        health.lastCheck = Date.now();
      }
      this.sendError(error, input, graph, hooks);
      hooks?.onNodeComplete?.(this.id, null);
      return null;
    }
  }
  destroy() {
    clearInterval(this.healthCheckTimer);
    super.destroy();
  }
};
function createLoadBalancerNode(nodes, options = {}) {
  const { strategy = "round-robin", healthCheckInterval = 3e4 } = options;
  return new LoadBalancerNode(nodes, strategy, healthCheckInterval).setName(
    `loadBalance(${strategy})`
  );
}
var OscillatorNode = class extends AdaptiveNode {
  constructor() {
    super((params) => {
      const {
        frequency,
        amplitude,
        sampleRate,
        length = 128,
        waveform = "sine",
        phase = 0
      } = params;
      const samples = new Float32Array(length);
      const phaseIncrement = 2 * Math.PI * frequency / sampleRate;
      let currentPhase = phase;
      for (let i = 0; i < samples.length; i++) {
        switch (waveform) {
          case "sine":
            samples[i] = Math.sin(currentPhase) * amplitude;
            break;
          case "square":
            samples[i] = (Math.sin(currentPhase) > 0 ? 1 : -1) * amplitude;
            break;
          case "sawtooth":
            samples[i] = (currentPhase / Math.PI - 1) * amplitude;
            break;
          case "triangle":
            samples[i] = (2 * Math.abs(
              2 * (currentPhase / (2 * Math.PI) - Math.floor(currentPhase / (2 * Math.PI) + 0.5))
            ) - 1) * amplitude;
            break;
        }
        currentPhase += phaseIncrement;
      }
      const nextPhase = currentPhase % (2 * Math.PI);
      return { samples, nextPhase };
    });
    this.setName("oscillator");
  }
};
function createProcessor(fn, name) {
  const node = new AdaptiveNode(fn);
  if (name) node.setName(name);
  return node;
}

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
