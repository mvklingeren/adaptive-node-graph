// src/core.ts
var AdaptiveNode = class {
  constructor(defaultProcessor, options = {}) {
    this.defaultProcessor = defaultProcessor;
    this.options = options;
    this.maxConcurrent = options.maxConcurrent || 10;
    this.circuitBreakerThreshold = options.circuitBreakerThreshold || 5;
    this.circuitBreakerResetTime = options.circuitBreakerResetTime || 6e4;
    this.setupInletsOutlets();
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
  id = Math.random().toString(36).substr(2, 9);
  visual = {
    x: 0,
    y: 0,
    label: "",
    inlets: [],
    outlets: []
  };
  // Type markers for compile-time type checking
  inputType;
  outputType;
  setupInletsOutlets() {
    this.visual.inlets = [{
      accept: (data) => this.process(data)
    }];
    this.visual.outlets = [
      // Data outlet
      {
        send: (data) => {
          this.visual.outlets[0].connections.forEach((conn) => {
            conn.transfer(data);
          });
        },
        connections: []
      },
      // Error outlet
      {
        send: (error) => {
          this.visual.outlets[1].connections.forEach((conn) => {
            conn.transfer(error);
          });
        },
        connections: []
      }
    ];
  }
  register(type, processor) {
    this.processors.set(type, processor);
    return this;
  }
  async process(input) {
    if (this.isCircuitOpen) {
      const now = Date.now();
      if (now - this.lastErrorTime > this.circuitBreakerResetTime) {
        this.isCircuitOpen = false;
        this.errorCount = 0;
      } else {
        this.sendError(new Error("Circuit breaker is open"), input);
        return null;
      }
    }
    if (this.processing >= this.maxConcurrent) {
      await new Promise((resolve) => this.queue.push(resolve));
    }
    this.processing++;
    const start = performance.now();
    let processorName = "default";
    try {
      let selectedProcessor = this.defaultProcessor;
      for (const [type, processor] of this.processors) {
        if (input instanceof type) {
          selectedProcessor = processor;
          processorName = type.name;
          break;
        }
      }
      const result = await selectedProcessor(input);
      this.recordPerformance(processorName, performance.now() - start);
      this.lastResult = result;
      if (this.visual.outlets[0]) {
        this.visual.outlets[0].send(result);
      }
      this.errorCount = 0;
      return result;
    } catch (error) {
      this.handleError(error, input);
      return null;
    } finally {
      this.processing--;
      const nextInQueue = this.queue.shift();
      if (nextInQueue) nextInQueue();
    }
  }
  handleError(error, input) {
    this.errorCount++;
    this.lastErrorTime = Date.now();
    if (this.errorCount >= this.circuitBreakerThreshold) {
      this.isCircuitOpen = true;
      console.error(`Circuit breaker opened for node ${this.id} after ${this.errorCount} errors`);
    }
    this.sendError(error, input);
  }
  sendError(error, input) {
    const nodeError = {
      error,
      input,
      nodeId: this.id,
      timestamp: Date.now()
    };
    if (this.visual.outlets[1] && this.visual.outlets[1].connections.length > 0) {
      this.visual.outlets[1].send(nodeError);
    } else {
      console.error(`Unhandled error in node ${this.id}:`, error);
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
  setPosition(x, y) {
    this.visual.x = x;
    this.visual.y = y;
    return this;
  }
  setLabel(label) {
    this.visual.label = label;
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
};
var Connection = class {
  constructor(source, sourceOutlet, target, targetInlet, transformer) {
    this.source = source;
    this.sourceOutlet = sourceOutlet;
    this.target = target;
    this.targetInlet = targetInlet;
    this.transformer = transformer;
    this.source.visual.outlets[this.sourceOutlet].connections.push(this);
  }
  async transfer(data) {
    try {
      const transformed = this.transformer ? await this.transformer(data) : data;
      await this.target.process(transformed);
    } catch (error) {
      console.error(`Error in connection ${this.source.id} -> ${this.target.id}:`, error);
      if (this.target.visual.outlets[1]) {
        this.target.visual.outlets[1].send({
          error,
          input: data,
          nodeId: `connection-${this.source.id}-${this.target.id}`,
          timestamp: Date.now()
        });
      }
    }
  }
  disconnect() {
    const outlet = this.source.visual.outlets[this.sourceOutlet];
    const index = outlet.connections.indexOf(this);
    if (index > -1) {
      outlet.connections.splice(index, 1);
    }
  }
};
var Graph = class {
  nodes = /* @__PURE__ */ new Map();
  connections = /* @__PURE__ */ new Set();
  executionOrder = [];
  addNode(node) {
    this.nodes.set(node.id, node);
    this.updateExecutionOrder();
    return this;
  }
  removeNode(nodeId) {
    const node = this.nodes.get(nodeId);
    if (!node) return this;
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
    const connection = new Connection(source, sourceOutlet, target, targetInlet, transformer);
    this.connections.add(connection);
    this.updateExecutionOrder();
    return connection;
  }
  // Connect error outlets
  connectError(source, errorHandler) {
    return this.connect(source, errorHandler, void 0, 1, 0);
  }
  disconnect(connection) {
    connection.disconnect();
    this.connections.delete(connection);
    this.updateExecutionOrder();
    return this;
  }
  updateExecutionOrder() {
    const visited = /* @__PURE__ */ new Set();
    const order = [];
    const visit = (node) => {
      if (visited.has(node.id)) return;
      visited.add(node.id);
      for (const conn of this.connections) {
        if (conn.target.id === node.id && conn.sourceOutlet === 0) {
          visit(conn.source);
        }
      }
      order.push(node);
    };
    for (const node of this.nodes.values()) {
      visit(node);
    }
    this.executionOrder = order;
  }
  async execute(input, startNodeId) {
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Node ${startNodeId} not found`);
      return startNode.process(input);
    }
    let result = input;
    for (const node of this.executionOrder) {
      result = await node.process(result);
    }
    return result;
  }
  // Parallel execution for independent nodes
  async executeParallel(input) {
    const results = /* @__PURE__ */ new Map();
    const dependencies = this.calculateDependencies();
    const executed = /* @__PURE__ */ new Set();
    const canExecute = (nodeId) => {
      const deps = dependencies.get(nodeId) || /* @__PURE__ */ new Set();
      for (const dep of deps) {
        if (!executed.has(dep)) return false;
      }
      return true;
    };
    while (executed.size < this.nodes.size) {
      const readyNodes = Array.from(this.nodes.values()).filter(
        (node) => !executed.has(node.id) && canExecute(node.id)
      );
      if (readyNodes.length === 0) break;
      const promises = readyNodes.map(async (node) => {
        const result = await node.process(input);
        results.set(node.id, result);
        executed.add(node.id);
        return result;
      });
      await Promise.all(promises);
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
  toJSON() {
    return {
      nodes: Array.from(this.nodes.values()).map((node) => ({
        id: node.id,
        label: node.visual.label,
        x: node.visual.x,
        y: node.visual.y
      })),
      connections: Array.from(this.connections).map((conn) => ({
        source: conn.source.id,
        sourceOutlet: conn.sourceOutlet,
        target: conn.target.id,
        targetInlet: conn.targetInlet
      }))
    };
  }
};
var createDelayNode = (ms) => new AdaptiveNode(async (input) => {
  await new Promise((resolve) => setTimeout(resolve, ms));
  return input;
}).setLabel(`delay(${ms}ms)`);
var createThrottleNode = (ms) => {
  let lastEmit = 0;
  return new AdaptiveNode((input) => {
    const now = Date.now();
    if (now - lastEmit >= ms) {
      lastEmit = now;
      return input;
    }
    return null;
  }).setLabel(`throttle(${ms}ms)`);
};
var createDebounceNode = (ms) => {
  let timeoutId = null;
  let resolver = null;
  return new AdaptiveNode(async (input) => {
    if (timeoutId) clearTimeout(timeoutId);
    return new Promise((resolve) => {
      resolver = resolve;
      timeoutId = setTimeout(() => {
        resolve(input);
        timeoutId = null;
        resolver = null;
      }, ms);
    });
  }).setLabel(`debounce(${ms}ms)`);
};
var createErrorLoggerNode = () => new AdaptiveNode((error) => {
  console.error(
    `[${new Date(error.timestamp).toISOString()}] Error in node ${error.nodeId}:`,
    error.error.message,
    "\nInput:",
    error.input
  );
  return error;
}).setLabel("errorLogger");
var createErrorRecoveryNode = (defaultValue) => new AdaptiveNode((error) => {
  console.warn(`Recovering from error with default value:`, defaultValue);
  return defaultValue;
}).setLabel("errorRecovery");
var createRetryNode = (maxRetries = 3, delayMs = 1e3) => {
  const retryMap = /* @__PURE__ */ new Map();
  return new AdaptiveNode(async (error) => {
    const key = `${error.nodeId}-${JSON.stringify(error.input)}`;
    const retries = retryMap.get(key) || 0;
    if (retries < maxRetries) {
      retryMap.set(key, retries + 1);
      await new Promise((resolve) => setTimeout(resolve, delayMs * (retries + 1)));
      console.log(`Retrying (${retries + 1}/${maxRetries})...`);
      return null;
    } else {
      console.error(`Max retries (${maxRetries}) exceeded`);
      retryMap.delete(key);
      return null;
    }
  }).setLabel(`retry(${maxRetries})`);
};
var TestNode = class extends AdaptiveNode {
  receivedInputs = [];
  processedOutputs = [];
  errors = [];
  constructor(processor) {
    super(processor || ((input) => {
      this.receivedInputs.push(input);
      return input;
    }));
    const originalProcess = this.process.bind(this);
    this.process = async (input) => {
      const result = await originalProcess(input);
      if (result !== null) {
        this.processedOutputs.push(result);
      }
      return result;
    };
    this.visual.outlets[1].send = (error) => {
      this.errors.push(error);
    };
    this.setLabel("test");
  }
  reset() {
    this.receivedInputs = [];
    this.processedOutputs = [];
    this.errors = [];
  }
  assertReceived(expected) {
    if (JSON.stringify(this.receivedInputs) !== JSON.stringify(expected)) {
      throw new Error(`Expected inputs ${JSON.stringify(expected)}, got ${JSON.stringify(this.receivedInputs)}`);
    }
  }
  assertProcessed(expected) {
    if (JSON.stringify(this.processedOutputs) !== JSON.stringify(expected)) {
      throw new Error(`Expected outputs ${JSON.stringify(expected)}, got ${JSON.stringify(this.processedOutputs)}`);
    }
  }
  assertNoErrors() {
    if (this.errors.length > 0) {
      throw new Error(`Expected no errors, got ${this.errors.length}: ${this.errors.map((e) => e.error.message).join(", ")}`);
    }
  }
};
async function testGraph(graph, input, expectedOutput, startNodeId) {
  const output = await graph.execute(input, startNodeId);
  if (JSON.stringify(output) !== JSON.stringify(expectedOutput)) {
    throw new Error(`Expected output ${JSON.stringify(expectedOutput)}, got ${JSON.stringify(output)}`);
  }
}
var SubGraphNode = class extends AdaptiveNode {
  constructor(subGraph, inputNodeId, outputNodeId) {
    super(async (input) => {
      const result = await this.subGraph.execute(input, this.inputNodeId);
      if (this.outputNodeId) {
        const outputNode = this.subGraph.getNode(this.outputNodeId);
        if (outputNode) {
          return outputNode.getLastResult() || result;
        }
      }
      return result;
    });
    this.subGraph = subGraph;
    this.inputNodeId = inputNodeId;
    this.outputNodeId = outputNodeId;
    this.setLabel("subgraph");
  }
  getSubGraph() {
    return this.subGraph;
  }
};
var createAddNode = () => new AdaptiveNode(
  (inputs) => inputs.reduce((a, b) => a + b, 0)
).setLabel("+");
var createMultiplyNode = () => new AdaptiveNode(
  (inputs) => inputs.reduce((a, b) => a * b, 1)
).setLabel("*");
var createSubtractNode = () => new AdaptiveNode(
  ([a, b]) => a - b
).setLabel("-");
var createDivideNode = () => new AdaptiveNode(
  ([a, b]) => b !== 0 ? a / b : 0
).setLabel("/");
var createFloat32MultiplyNode = () => new AdaptiveNode(
  (input) => {
    const result = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      result[i] = input[i] * 0.5;
    }
    return result;
  }
).setLabel("f32*");
var createConditionalNode = () => new AdaptiveNode(
  ([condition, ifTrue, ifFalse]) => condition ? ifTrue : ifFalse
).setLabel("?");
var createAndNode = () => new AdaptiveNode(
  (inputs) => inputs.every(Boolean)
).setLabel("&&");
var createOrNode = () => new AdaptiveNode(
  (inputs) => inputs.some(Boolean)
).setLabel("||");
var createNotNode = () => new AdaptiveNode(
  (input) => !input
).setLabel("!");
var createGateNode = () => new AdaptiveNode(
  ([pass, data]) => pass ? data : null
).setLabel("gate");
var createMergeNode = () => new AdaptiveNode(
  (inputs) => inputs.filter((x) => x !== null && x !== void 0)
).setLabel("merge");
var createSplitNode = (count = 2) => {
  const node = new AdaptiveNode(
    (input) => Array(count).fill(input)
  ).setLabel("split");
  node.visual.outlets = [
    ...Array(count).fill(null).map(() => ({
      send: () => {
      },
      connections: []
    })),
    // Error outlet
    {
      send: () => {
      },
      connections: []
    }
  ];
  return node;
};
var createRouterNode = () => new AdaptiveNode(
  (input) => ({ route: "default", data: input })
).register(AudioBuffer, (audio) => ({ route: "audio", data: audio })).register(ArrayBuffer, (buffer) => ({ route: "binary", data: buffer })).register(Array, (array) => ({ route: "array", data: array })).register(Object, (obj) => {
  if ("sampleRate" in obj) return { route: "audio", data: obj };
  if ("buffer" in obj) return { route: "binary", data: obj };
  return { route: "object", data: obj };
}).setLabel("router");
function createLoadBalancerNode(nodes, strategy = "round-robin") {
  let index = 0;
  const nodeHealth = new Map(nodes.map((n) => [n.id, true]));
  const loadBalancer = new AdaptiveNode(async (input) => {
    const healthyNodes = nodes.filter((n) => nodeHealth.get(n.id));
    if (healthyNodes.length === 0) {
      throw new Error("No healthy nodes available");
    }
    let selectedNode;
    switch (strategy) {
      case "random":
        selectedNode = healthyNodes[Math.floor(Math.random() * healthyNodes.length)];
        break;
      case "least-loaded":
        selectedNode = healthyNodes[0];
        break;
      default:
        selectedNode = healthyNodes[index % healthyNodes.length];
        index++;
    }
    try {
      const result = await selectedNode.process(input);
      if (result === null) {
        throw new Error("Node returned null");
      }
      return result;
    } catch (error) {
      nodeHealth.set(selectedNode.id, false);
      if (healthyNodes.length > 1) {
        const retryNodes = nodes.filter((n) => n.id !== selectedNode.id && nodeHealth.get(n.id));
        if (retryNodes.length > 0) {
          const retryBalancer = createLoadBalancerNode(retryNodes, strategy);
          const retryResult = await retryBalancer.process(input);
          if (retryResult === null) {
            throw new Error("Retry returned null");
          }
          return retryResult;
        }
      }
      throw error;
    }
  });
  return loadBalancer.setLabel(`loadBalance(${strategy})`);
}
function createParallelNode(nodes) {
  return new AdaptiveNode(async (input) => {
    return Promise.all(nodes.map((node) => node.process(input)));
  }).setLabel("parallel");
}
function createCacheNode(processor, ttl = 1e3, maxSize = 100) {
  const cache = /* @__PURE__ */ new Map();
  return new AdaptiveNode(async (input) => {
    const key = JSON.stringify(input);
    const cached = cache.get(key);
    if (cached && Date.now() - cached.timestamp < ttl) {
      return cached.value;
    }
    const result = await processor(input);
    cache.set(key, { value: result, timestamp: Date.now() });
    if (cache.size > maxSize) {
      const firstKey = cache.keys().next().value;
      if (firstKey !== void 0) {
        cache.delete(firstKey);
      }
    }
    for (const [k, v] of cache.entries()) {
      if (Date.now() - v.timestamp > ttl) {
        cache.delete(k);
      }
    }
    return result;
  }).setLabel(`cache(${ttl}ms)`);
}
var OscillatorNode = class extends AdaptiveNode {
  phase = 0;
  constructor() {
    super((params) => {
      const samples = new Float32Array(params.length || 128);
      const phaseIncrement = 2 * Math.PI * params.frequency / params.sampleRate;
      switch (params.waveform || "sine") {
        case "sine":
          for (let i = 0; i < samples.length; i++) {
            samples[i] = Math.sin(this.phase) * params.amplitude;
            this.phase += phaseIncrement;
          }
          break;
        case "square":
          for (let i = 0; i < samples.length; i++) {
            samples[i] = (Math.sin(this.phase) > 0 ? 1 : -1) * params.amplitude;
            this.phase += phaseIncrement;
          }
          break;
        case "sawtooth":
          for (let i = 0; i < samples.length; i++) {
            samples[i] = (this.phase / Math.PI - 1) * params.amplitude;
            this.phase += phaseIncrement;
            if (this.phase >= 2 * Math.PI) this.phase -= 2 * Math.PI;
          }
          break;
        case "triangle":
          for (let i = 0; i < samples.length; i++) {
            samples[i] = (2 * Math.abs(2 * (this.phase / (2 * Math.PI) - Math.floor(this.phase / (2 * Math.PI) + 0.5))) - 1) * params.amplitude;
            this.phase += phaseIncrement;
          }
          break;
      }
      this.phase = this.phase % (2 * Math.PI);
      return samples;
    });
    this.setLabel("oscillator");
  }
};
function createProcessor(fn, label) {
  const node = new AdaptiveNode(fn);
  if (label) node.setLabel(label);
  return node;
}
function chain(first, second) {
  return new AdaptiveNode(async (input) => {
    const intermediate = await first.process(input);
    if (intermediate === null) {
      throw new Error("Intermediate result was null");
    }
    const result = await second.process(intermediate);
    if (result === null) {
      throw new Error("Second node result was null");
    }
    return result;
  }).setLabel(`${first.visual.label} \u2192 ${second.visual.label}`);
}

export {
  AdaptiveNode,
  Connection,
  Graph,
  createDelayNode,
  createThrottleNode,
  createDebounceNode,
  createErrorLoggerNode,
  createErrorRecoveryNode,
  createRetryNode,
  TestNode,
  testGraph,
  SubGraphNode,
  createAddNode,
  createMultiplyNode,
  createSubtractNode,
  createDivideNode,
  createFloat32MultiplyNode,
  createConditionalNode,
  createAndNode,
  createOrNode,
  createNotNode,
  createGateNode,
  createMergeNode,
  createSplitNode,
  createRouterNode,
  createLoadBalancerNode,
  createParallelNode,
  createCacheNode,
  OscillatorNode,
  createProcessor,
  chain
};
