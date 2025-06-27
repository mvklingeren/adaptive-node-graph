// src/core.ts
import pino from "pino";
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
        send: (error, graph, hooks) => {
          errorOutletConnections.forEach((conn) => {
            conn.transfer(error, graph, hooks);
          });
        },
        connections: errorOutletConnections
      }
    ];
  }
  register(type, processor) {
    this.processors.set(type, processor);
    return this;
  }
  async process(input, graph, hooks) {
    if (graph?.isStopped) return null;
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
        this.sendError(
          new Error("Circuit breaker is open"),
          input,
          graph,
          hooks
        );
        return null;
      }
    }
    if (this.processing >= this.maxConcurrent) {
      await new Promise((resolve) => this.queue.push(resolve));
    }
    this.processing++;
    let result = null;
    try {
      result = await this.executeProcessor(input, graph, hooks);
      if (result !== null) {
        if (this.isSplitNode) {
          const count = this.dataOutletCount || 0;
          const promises = this.outlets.slice(0, count).map((outlet) => outlet.send(result, graph, hooks));
          await Promise.all(promises);
        } else if (this.outlets[0]) {
          await this.outlets[0].send(result, graph, hooks);
        }
      }
      return result;
    } catch (error) {
      this.handleError(error, input, graph, hooks);
      return null;
    } finally {
      this.processing--;
      if (this.queue.length > 0) {
        this.queue.shift()?.();
      }
      hooks?.onNodeComplete?.(this.id, result);
    }
  }
  async executeProcessor(input, _graph, _hooks) {
    const start = performance.now();
    let processorName = "default";
    let selectedProcessor = this.defaultProcessor;
    processorName = "default";
    for (const [type, processor] of this.processors) {
      let match = false;
      if (typeof type === "function") {
        if (type.prototype && type.prototype.constructor === type && input instanceof type) {
          match = true;
          processorName = type.name;
        } else if (!type.prototype || !type.prototype.constructor) {
          match = type(input);
          processorName = `predicate:${type.toString()}`;
        }
      }
      if (match) {
        selectedProcessor = processor;
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
  isStopped = false;
  addNode(node) {
    this.nodes.set(node.id, node);
    this.updateExecutionOrder();
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
    this.updateExecutionOrder();
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
    this.updateExecutionOrder();
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
    this.updateExecutionOrder();
    return connection;
  }
  disconnect(connection) {
    connection.disconnect();
    this.connections.delete(connection);
    this.updateExecutionOrder();
    return this;
  }
  getExecutionOrder() {
    this.updateExecutionOrder();
    return this.executionOrder;
  }
  updateExecutionOrder() {
    const order = [];
    const visiting = /* @__PURE__ */ new Set();
    const visited = /* @__PURE__ */ new Set();
    const visit = (node) => {
      if (visited.has(node.id)) {
        return;
      }
      if (visiting.has(node.id)) {
        console.warn(`Cycle detected in graph involving node ${node.id}`);
        return;
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
  }
  async execute(input, startNodeId, hooks) {
    this.isStopped = false;
    const processNode = (node, nodeInput) => {
      return node.process(nodeInput, this, hooks);
    };
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Start node ${startNodeId} not found`);
      const nodeInput = input instanceof Map ? input.get(startNodeId) : input;
      return processNode(startNode, nodeInput);
    }
    const entryNodes = this.executionOrder.filter((node) => {
      const hasIncomingConnection = Array.from(this.connections).some(
        (conn) => conn.target.id === node.id && conn.sourceOutlet === 0
      );
      const hasInitialValue = node.initialValue !== null;
      return !hasIncomingConnection || hasInitialValue;
    });
    if (entryNodes.length === 0 && this.nodes.size > 0) {
      const firstNode = this.executionOrder[0];
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
    const promises = entryNodes.map((node) => {
      const nodeInput = input instanceof Map ? input.get(node.id) : input;
      return processNode(node, nodeInput);
    });
    await Promise.all(promises);
    const lastNode = this.executionOrder[this.executionOrder.length - 1];
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
var createFloat32MultiplyNode = () => new AdaptiveNode((input) => {
  const result = new Float32Array(input?.length);
  for (let i = 0; i < input?.length; i++) {
    result[i] = input[i] * 0.5;
  }
  return result;
}).setName("f32*");
var LoadBalancerNode = class extends AdaptiveNode {
  constructor(nodes, strategy = "round-robin", healthCheckInterval = 3e4) {
    super(async () => null);
    this.nodes = nodes;
    this.strategy = strategy;
    this.nodeHealth = new Map(
      nodes.map((n) => [n.id, { healthy: true, lastCheck: 0 }])
    );
    this.healthCheckTimer = setInterval(() => {
      const now = Date.now();
      for (const [nodeId, health] of this.nodeHealth.entries()) {
        if (!health.healthy && now - health.lastCheck > healthCheckInterval) {
          console.log(`Retrying health check for unhealthy node ${nodeId}`);
          health.healthy = true;
          health.lastCheck = now;
        }
      }
    }, healthCheckInterval);
  }
  index = 0;
  nodeHealth;
  healthCheckTimer;
  async executeProcessor(input) {
    const nodesToTry = this.nodes.filter(
      (n) => this.nodeHealth.get(n.id)?.healthy
    );
    if (nodesToTry.length === 0) {
      throw new Error("No healthy nodes available in load balancer.");
    }
    let selectedNode;
    switch (this.strategy) {
      case "random":
        selectedNode = nodesToTry[Math.floor(Math.random() * nodesToTry.length)];
        break;
      case "least-loaded":
        selectedNode = nodesToTry[0];
        break;
      default:
        this.index = this.index % nodesToTry.length;
        selectedNode = nodesToTry[this.index];
        this.index++;
        break;
    }
    if (!selectedNode) {
      throw new Error("Load balancer could not select a node.");
    }
    try {
      const result = await selectedNode.process(input);
      if (result === null) {
        throw new Error(`Node ${selectedNode.id} returned null`);
      }
      return result;
    } catch (error) {
      console.warn(
        `Node ${selectedNode.id} failed in load balancer. Marking as unhealthy.`,
        error
      );
      const health = this.nodeHealth.get(selectedNode.id);
      if (health) {
        health.healthy = false;
        health.lastCheck = Date.now();
      }
      throw new Error(
        `Node ${selectedNode.id} failed. All other nodes were skipped.`
      );
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
async function audioProcessingDemo() {
  console.log("=== Audio Processing Demo ===");
  const envelope = createProcessor((time) => {
    if (time < 0.1) return time / 0.1;
    if (time < 0.2) return 1;
    if (time < 0.8) return 0.7;
    return 0.7 * (1 - (time - 0.8) / 0.2);
  }, "envelope");
  const filter = new AdaptiveNode((samples) => {
    const filtered2 = new Float32Array(samples.length);
    filtered2[0] = samples[0];
    for (let i = 1; i < samples.length; i++) {
      filtered2[i] = filtered2[i - 1] * 0.9 + samples[i] * 0.1;
    }
    return filtered2;
  }).setName("lowpass");
  const graph = new Graph();
  const osc = new OscillatorNode();
  const gain = createProcessor(
    ([samples, gainValue]) => samples.map((s) => s * gainValue),
    "gain"
  );
  graph.addNode(osc);
  graph.addNode(envelope);
  graph.addNode(filter);
  graph.addNode(gain);
  const params = {
    frequency: 440,
    amplitude: 0.5,
    sampleRate: 44100,
    length: 44100,
    waveform: "sawtooth"
  };
  const oscResult = await osc.process(params);
  if (!oscResult) {
    console.error("Audio processing failed: Oscillator did not produce output.");
    return;
  }
  const filtered = await filter.process(oscResult.samples);
  const envelopeValues = await Promise.all(
    Array.from({ length: 44100 }, (_, i) => envelope.process(i / 44100))
  );
  const output = filtered?.map((sample, i) => sample * (envelopeValues[i] ?? 0));
  console.log("Audio processing demo complete.");
}
async function basicMathDemo() {
  console.log("\n=== Basic Math Demo ===");
  const graph = new Graph();
  const input = createProcessor((x) => x, "input");
  const add5 = createProcessor((x) => x + 5, "add5");
  const multiply2 = createProcessor((x) => x * 2, "multiply2");
  const output = createProcessor(
    (x) => console.log("Result:", x),
    "output"
  );
  graph.addNode(input).addNode(add5).addNode(multiply2).addNode(output);
  graph.connect(input, add5);
  graph.connect(add5, multiply2);
  graph.connect(multiply2, output);
  await graph.execute(10);
}
async function conditionalRoutingDemo() {
  console.log("\n=== Conditional Routing Demo ===");
  const graph = new Graph();
  const tempSensor = createProcessor(() => Math.random() * 50, "temp-sensor");
  const threshold = createProcessor(
    (temp) => temp > 30,
    "threshold"
  );
  const alert = createProcessor(
    (temp) => console.log(`\u{1F525} HIGH TEMP: ${temp}\xB0C`),
    "alert"
  );
  const normal = createProcessor(
    (temp) => console.log(`\u2713 Normal: ${temp}\xB0C`),
    "normal"
  );
  const router = createProcessor(([isHigh, temp]) => {
    if (isHigh) {
      alert.process(temp);
    } else {
      normal.process(temp);
    }
  }, "router");
  graph.addNode(tempSensor).addNode(threshold).addNode(router).addNode(alert).addNode(normal);
  const sensorOutput = await tempSensor.process(null);
  if (sensorOutput !== null) {
    const isHigh = await threshold.process(sensorOutput);
    if (isHigh !== null) {
      await router.process([isHigh, sensorOutput]);
    }
  }
}
function dynamicGraphDemo() {
  console.log("\n=== Dynamic Graph Demo ===");
  console.log("Dynamic graph demo is conceptual and commented out.");
}
async function errorHandlingDemo() {
  console.log("\n=== Error Handling Demo ===");
  const processData = (data) => {
    if (typeof data === "number") {
      return data * 2;
    } else if (typeof data === "string") {
      return data.toUpperCase();
    } else if (Array.isArray(data)) {
      return data.map(processData);
    } else if (data === null || data === void 0) {
      throw new Error("Cannot process null or undefined data");
    } else if (typeof data === "object") {
      return Object.fromEntries(
        Object.entries(data).map(([key, value]) => [
          key,
          processData(value)
        ])
      );
    }
    throw new Error(`Unsupported data type: ${typeof data}`);
  };
  const safeProcessor = createProcessor(
    (input) => {
      try {
        return processData(input);
      } catch (error) {
        console.error("Processing failed:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        return { error: errorMessage, input };
      }
    },
    "safe-processor"
  );
  console.log('Processing { a: "hello", b: [1, 2] }');
  const result1 = await safeProcessor.process({ a: "hello", b: [1, 2] });
  console.log("Result:", result1);
  console.log("Processing null");
  const result2 = await safeProcessor.process(null);
  console.log("Result:", result2);
}
async function machineLearningDemo() {
  console.log("\n=== Machine Learning Demo ===");
  class DataPreprocessor extends AdaptiveNode {
    constructor() {
      super((data) => new Float32Array(data));
      this.register(Array, this.preprocessArray.bind(this));
      this.register(Float32Array, this.preprocessFloat32.bind(this));
    }
    preprocessArray(data) {
      const arr = new Float32Array(data);
      const mean = arr.reduce((a, b) => a + b) / arr.length;
      const std = Math.sqrt(
        arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length
      );
      return arr.map((val) => (val - mean) / std);
    }
    preprocessFloat32(data) {
      return this.preprocessArray(Array.from(data));
    }
  }
  const featureExtractor = createProcessor(
    (data) => {
      const features = new Float32Array(5);
      features[0] = Math.min(...data);
      features[1] = Math.max(...data);
      features[2] = data.reduce((a, b) => a + b) / data.length;
      features[3] = Math.sqrt(
        data.reduce((sum, val) => sum + val * val, 0) / data.length
      );
      features[4] = data.reduce(
        (sum, val, i) => i > 0 ? sum + Math.abs(val - data[i - 1]) : sum,
        0
      );
      return features;
    },
    "feature-extractor"
  );
  const model = createProcessor((features) => {
    const sum = features.reduce((a, b) => a + b);
    if (sum > 10) {
      return { class: "positive", confidence: 0.85 };
    } else if (sum < -10) {
      return { class: "negative", confidence: 0.9 };
    } else {
      return { class: "neutral", confidence: 0.75 };
    }
  }, "ml-model");
  const postProcessor = new AdaptiveNode((prediction) => prediction).register(Object, (pred) => {
    if (pred.confidence < 0.5) {
      return { ...pred, class: "uncertain" };
    }
    return pred;
  }).setName("post-processor");
  const graph = new Graph();
  const preprocessor = new DataPreprocessor().setName("preprocessor");
  graph.addNode(preprocessor);
  graph.addNode(featureExtractor);
  graph.addNode(model);
  graph.addNode(postProcessor);
  graph.connect(preprocessor, featureExtractor);
  graph.connect(featureExtractor, model);
  graph.connect(model, postProcessor);
  const testData = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    new Float32Array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]),
    Array.from({ length: 100 }, () => Math.random() * 10 - 5)
  ];
  for (const data of testData) {
    const result = await graph.execute(data, preprocessor.id);
    console.log("Prediction:", result);
  }
}
async function multiProtocolDemo() {
  console.log("\n=== Multi-protocol Demo ===");
  class APIGateway extends AdaptiveNode {
    rateLimits = /* @__PURE__ */ new Map();
    cache = /* @__PURE__ */ new Map();
    constructor() {
      super((request) => ({ error: "Unknown protocol" }));
      this.register(Object, this.routeByShape.bind(this));
    }
    routeByShape(request) {
      if ("method" in request && "path" in request) {
        return this.handleHTTP(request);
      }
      if (request.type === "ws" && "action" in request) {
        return this.handleWebSocket(request);
      }
      if ("service" in request && "method" in request) {
        return this.handleGRPC(request);
      }
      return { error: "Unrecognized request format" };
    }
    handleHTTP(request) {
      const clientId = request.headers["x-client-id"] || "anonymous";
      if (!this.checkRateLimit(clientId)) {
        return { status: 429, error: "Rate limit exceeded" };
      }
      const cacheKey = `${request.method}:${request.path}`;
      const cached = this.cache.get(cacheKey);
      if (cached && cached.expires > Date.now()) {
        return { status: 200, data: cached.data, cached: true };
      }
      const response = this.routeHTTPRequest(request);
      if (request.method === "GET" && response.status === 200) {
        this.cache.set(cacheKey, {
          data: response.data,
          expires: Date.now() + 6e4
          // 1 minute
        });
      }
      return response;
    }
    handleWebSocket(message) {
      switch (message.action) {
        case "subscribe":
          return { type: "subscription", channel: message.payload.channel };
        case "publish":
          return { type: "broadcast", sent: true };
        default:
          return { type: "error", message: "Unknown action" };
      }
    }
    handleGRPC(request) {
      return {
        service: request.service,
        method: request.method,
        response: `Processed ${request.service}.${request.method}`
      };
    }
    checkRateLimit(clientId) {
      const now = Date.now();
      const window = 6e4;
      const limit = 100;
      if (!this.rateLimits.has(clientId)) {
        this.rateLimits.set(clientId, []);
      }
      const requests2 = this.rateLimits.get(clientId);
      const recentRequests = requests2.filter((time) => now - time < window);
      if (recentRequests.length >= limit) {
        return false;
      }
      recentRequests.push(now);
      this.rateLimits.set(clientId, recentRequests);
      return true;
    }
    routeHTTPRequest(request) {
      if (request.path.startsWith("/api/users")) {
        return { status: 200, data: { users: [] } };
      }
      if (request.path.startsWith("/api/products")) {
        return { status: 200, data: { products: [] } };
      }
      return { status: 404, error: "Not found" };
    }
  }
  const authService = createProcessor((request) => {
    return { ...request, authenticated: true };
  }, "auth-service");
  const loggingService = createProcessor((request) => {
    console.log(`[${(/* @__PURE__ */ new Date()).toISOString()}] Request:`, request);
    return request;
  }, "logging-service");
  const analyticsService = new AdaptiveNode((event) => {
    return event;
  }).register(Object, (event) => {
    if (event.status >= 400) {
      console.log("Error event:", event);
    }
    return event;
  }).setName("analytics");
  const graph = new Graph();
  const gateway = new APIGateway().setName("api-gateway");
  graph.addNode(loggingService);
  graph.addNode(authService);
  graph.addNode(gateway);
  graph.addNode(analyticsService);
  graph.connect(loggingService, authService);
  graph.connect(authService, gateway);
  graph.connect(gateway, analyticsService);
  const requests = [
    // HTTP Request
    {
      method: "GET",
      path: "/api/users/123",
      headers: { "x-client-id": "client-1" }
    },
    // WebSocket Message
    {
      type: "ws",
      action: "subscribe",
      payload: { channel: "updates" }
    },
    // gRPC Request
    {
      service: "UserService",
      method: "GetUser",
      data: { userId: 123 }
    }
  ];
  for (const request of requests) {
    const result = await graph.execute(request, loggingService.id);
    console.log("Result:", result);
  }
}
async function oscillatorDemo() {
  console.log("\n=== Oscillator Demo ===");
  const graph = new Graph();
  const osc = new OscillatorNode();
  const multiply = createFloat32MultiplyNode();
  const extractor = createProcessor(
    (data) => data.samples,
    "extractor"
  );
  graph.addNode(osc);
  graph.addNode(extractor);
  graph.addNode(multiply);
  graph.connect(osc, extractor);
  graph.connect(extractor, multiply);
  const result = await graph.execute({
    frequency: 440,
    amplitude: 0.5,
    sampleRate: 44100
  });
  console.log("Oscillator result:", result);
}
async function performanceMonitoringDemo() {
  console.log("\n=== Performance Monitoring Demo ===");
  function expensiveOperation(input) {
    const iterations = input?.iterations || 1e6;
    let result = 0;
    for (let i = 0; i < iterations; i++) {
      result += Math.sin(i) * Math.cos(i);
    }
    return result;
  }
  const monitoredNode = new AdaptiveNode((input) => {
    const start = performance.now();
    const result = expensiveOperation(input);
    const duration = performance.now() - start;
    if (duration > 100) {
      console.warn(`Slow operation: ${duration}ms`);
    }
    return result;
  }).setName("monitored");
  await monitoredNode.process({ iterations: 2e6 });
  const stats = monitoredNode.getPerformanceStats();
  console.log("Performance stats:", stats);
}
async function realtimeStreamDemo() {
  console.log("\n=== Real-time Stream Demo ===");
  class StreamProcessor extends AdaptiveNode {
    buffer = [];
    windowSize = 100;
    constructor() {
      super((event) => event);
      if (typeof MouseEvent !== "undefined") {
        this.register(MouseEvent, this.processMouseEvent.bind(this));
      }
      if (typeof KeyboardEvent !== "undefined") {
        this.register(KeyboardEvent, this.processKeyboardEvent.bind(this));
      }
      this.register(Object, this.processDataEvent.bind(this));
    }
    processMouseEvent(event) {
      return {
        type: "mouse",
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
      };
    }
    processKeyboardEvent(event) {
      return {
        type: "keyboard",
        key: event.key,
        timestamp: Date.now()
      };
    }
    processDataEvent(event) {
      this.buffer.push(event);
      if (this.buffer.length > this.windowSize) {
        this.buffer.shift();
      }
      return {
        type: "data",
        count: this.buffer.length,
        average: this.computeAverage(),
        timestamp: Date.now()
      };
    }
    computeAverage() {
      if (this.buffer.length === 0) return 0;
      const sum = this.buffer.reduce((acc, val) => acc + (val.value || 0), 0);
      return sum / this.buffer.length;
    }
  }
  const workers = Array.from(
    { length: 4 },
    (_, i) => new StreamProcessor().setName(`worker-${i}`)
  );
  const loadBalancer = createLoadBalancerNode(workers);
  const aggregator = createProcessor((result) => {
    console.log(`Processed event:`, result);
  }, "aggregator");
  const graph = new Graph();
  graph.addNode(loadBalancer);
  workers.forEach((w) => graph.addNode(w));
  graph.addNode(aggregator);
  workers.forEach((worker) => {
    graph.connect(loadBalancer, worker);
    graph.connect(worker, aggregator);
  });
  const events = [];
  if (typeof MouseEvent !== "undefined") {
    events.push(new MouseEvent("click", { clientX: 100, clientY: 200 }));
  }
  events.push({ value: 42 });
  if (typeof KeyboardEvent !== "undefined") {
    events.push(new KeyboardEvent("keydown", { key: "Enter" }));
  }
  events.push({ value: 38 }, { value: 45 });
  for (const event of events) {
    await loadBalancer.process(event);
  }
}
async function transformationPipelineDemo() {
  console.log("\n=== Transformation Pipeline Demo ===");
  const validator = createProcessor((user) => {
    if (!user.email.includes("@")) {
      throw new Error("Invalid email");
    }
    if (user.age < 0 || user.age > 150) {
      throw new Error("Invalid age");
    }
    return user;
  }, "validator");
  const enricher = createProcessor(
    (user) => ({
      ...user,
      category: user.age < 18 ? "minor" : user.age < 65 ? "adult" : "senior"
    }),
    "enricher"
  );
  const privacyFilter = new AdaptiveNode((data) => data).register(Object, (obj) => {
    const filtered = { ...obj };
    if ("email" in filtered) {
      filtered.email = filtered.email.replace(/(.{2}).*(@.*)/, "$1***$2");
    }
    return filtered;
  }).setName("privacy-filter");
  const graph = new Graph();
  graph.addNode(validator);
  graph.addNode(enricher);
  graph.addNode(privacyFilter);
  graph.connect(validator, enricher);
  graph.connect(enricher, privacyFilter);
  const userData = {
    id: 1,
    name: "John Doe",
    email: "john.doe@example.com",
    age: 25
  };
  const result = await graph.execute(userData, validator.id);
  console.log(result);
}
async function typeAdaptiveDemo() {
  console.log("\n=== Type Adaptive Demo ===");
  const smartProcessor = new AdaptiveNode(
    (input) => `Unknown type: ${typeof input}`
  ).register(Number, (num) => `Number: ${num.toFixed(2)}`).register(String, (str) => `String: "${str.toUpperCase()}"`).register(Array, (arr) => `Array[${arr.length}]: ${arr.join(", ")}`).register(Date, (date) => `Date: ${date.toISOString()}`).setName("smart-processor");
  const logger = createProcessor(
    (msg) => console.log(msg),
    "logger"
  );
  const graph = new Graph();
  graph.addNode(smartProcessor);
  graph.addNode(logger);
  graph.connect(smartProcessor, logger);
  await smartProcessor.process(42);
  await smartProcessor.process("hello");
  await smartProcessor.process([1, 2, 3]);
  await smartProcessor.process(/* @__PURE__ */ new Date());
}
async function runAllDemos() {
  try {
    await audioProcessingDemo();
    await basicMathDemo();
    await conditionalRoutingDemo();
    dynamicGraphDemo();
    await errorHandlingDemo();
    await machineLearningDemo();
    await multiProtocolDemo();
    await oscillatorDemo();
    await performanceMonitoringDemo();
    await realtimeStreamDemo();
    await transformationPipelineDemo();
    await typeAdaptiveDemo();
    console.log("\n=== All demos completed successfully! ===");
  } catch (error) {
    console.error("Demo error:", error);
  }
}
runAllDemos().catch(console.error);
export {
  runAllDemos
};
