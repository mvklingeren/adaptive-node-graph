// src/core.ts
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
      if (this.outlets[0] && result !== null) {
        await this.outlets[0].send(result, graph, hooks);
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
    this.errorCount = 0;
    return result;
  }
  handleError(error, input, graph, hooks) {
    this.errorCount++;
    this.lastErrorTime = Date.now();
    if (this.errorCount >= this.circuitBreakerThreshold) {
      this.isCircuitOpen = true;
      console.error(
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
      console.error(`Unhandled error in node ${this.id}:`, nodeError);
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
      console.error(`Error in ${connError.nodeId}:`, connError.error);
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
    const visited = /* @__PURE__ */ new Set();
    const order = [];
    const visit = (node) => {
      if (visited.has(node.id)) return;
      visited.add(node.id);
      for (const conn of this.connections) {
        if (conn.target.id === node.id && conn.sourceOutlet === 0) {
          const sourceNode = this.nodes.get(conn.source.id);
          if (sourceNode) visit(sourceNode);
        }
      }
      order.push(node);
    };
    for (const node of this.nodes.values()) {
      visit(node);
    }
    this.executionOrder = order;
  }
  async execute(input, startNodeId, hooks) {
    this.isStopped = false;
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Start node ${startNodeId} not found`);
      return startNode.process(input, this, hooks);
    }
    const sourceNodes = this.executionOrder.filter((node) => {
      const hasIncomingConnection = Array.from(this.connections).some(
        (conn) => conn.target.id === node.id && conn.sourceOutlet === 0
      );
      const hasInitialValue = node.initialValue !== null;
      return !hasIncomingConnection || hasInitialValue;
    });
    const entryNodes = sourceNodes.length > 0 ? sourceNodes : this.executionOrder.length > 0 ? [this.executionOrder[0]] : [];
    const promises = entryNodes.map((node) => node.process(input, this, hooks));
    await Promise.all(promises);
    const lastNode = this.executionOrder[this.executionOrder.length - 1];
    return lastNode?.getLastResult();
  }
  stop() {
    this.isStopped = true;
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
};
var createDelayNode = (ms) => new AdaptiveNode(async (input) => {
  await new Promise((resolve) => setTimeout(resolve, ms));
  return input;
}).setName(`delay(${ms}ms)`);
var createThrottleNode = (ms) => {
  let lastEmit = 0;
  return new AdaptiveNode((input) => {
    const now = Date.now();
    if (now - lastEmit >= ms) {
      lastEmit = now;
      return input;
    }
    return null;
  }).setName(`throttle(${ms}ms)`);
};
var createDebounceNode = (ms) => {
  let timeoutId = null;
  return new AdaptiveNode(async (input) => {
    if (timeoutId) clearTimeout(timeoutId);
    return new Promise((resolve) => {
      timeoutId = setTimeout(() => {
        resolve(input);
        timeoutId = null;
      }, ms);
    });
  }).setName(`debounce(${ms}ms)`);
};
var createErrorLoggerNode = () => new AdaptiveNode((error) => {
  console.error(
    `[${new Date(error.timestamp).toISOString()}] Error in node ${error.nodeId}:`,
    error.error.message,
    "\nInput:",
    error.input
  );
  return error;
}).setName("errorLogger");
var createErrorRecoveryNode = (defaultValue) => new AdaptiveNode((_error) => {
  console.warn(`Recovering from error. Returning default value.`);
  return defaultValue;
}).setName("errorRecovery");
var TestNode = class extends AdaptiveNode {
  receivedInputs = [];
  processedOutputs = [];
  errors = [];
  constructor(processor) {
    super(
      processor || ((input) => {
        this.receivedInputs.push(input);
        return input;
      })
    );
    const originalProcess = this.process.bind(this);
    this.process = async (input) => {
      const result = await originalProcess(input);
      if (result !== null) {
        this.processedOutputs.push(result);
      }
      return result;
    };
    this.outlets[1].send = (error) => {
      this.errors.push(error);
    };
    this.setName("test");
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
var SubGraphNode = class extends AdaptiveNode {
  constructor(subGraph, inputNodeId, outputNodeId) {
    super(async () => null);
    this.subGraph = subGraph;
    this.inputNodeId = inputNodeId;
    this.outputNodeId = outputNodeId;
    this.setName("subgraph");
  }
  async executeProcessor(input, graph, hooks) {
    const result = await this.subGraph.execute(input, this.inputNodeId, hooks);
    let finalResult;
    if (this.outputNodeId) {
      const outputNode = this.subGraph.getNode(this.outputNodeId);
      if (outputNode) {
        finalResult = outputNode.getLastResult() || result;
      } else {
        finalResult = result;
      }
    } else {
      finalResult = result;
    }
    this.lastResult = finalResult;
    return finalResult;
  }
  getSubGraph() {
    return this.subGraph;
  }
};
var createAddNode = () => new AdaptiveNode(
  (inputs) => inputs.reduce((a, b) => a + b, 0)
).setName("+");
var createMultiplyNode = () => new AdaptiveNode(
  (inputs) => inputs.reduce((a, b) => a * b, 1)
).setName("*");
var createSubtractNode = () => new AdaptiveNode(([a, b]) => a - b).setName("-");
var createDivideNode = () => new AdaptiveNode(
  ([a, b]) => b !== 0 ? a / b : 0
).setName("/");
var createFloat32MultiplyNode = () => new AdaptiveNode((input) => {
  const result = new Float32Array(input?.length);
  for (let i = 0; i < input?.length; i++) {
    result[i] = input[i] * 0.5;
  }
  return result;
}).setName("f32*");
var createConditionalNode = () => new AdaptiveNode(
  ([condition, ifTrue, ifFalse]) => condition ? ifTrue : ifFalse
).setName("?");
var createAndNode = () => new AdaptiveNode(
  (inputs) => inputs.every(Boolean)
).setName("&&");
var createOrNode = () => new AdaptiveNode(
  (inputs) => inputs.some(Boolean)
).setName("||");
var createNotNode = () => new AdaptiveNode((input) => !input).setName("!");
var createGateNode = () => new AdaptiveNode(
  ([pass, data]) => pass ? data : null
).setName("gate");
var createMergeNode = () => new AdaptiveNode(
  (inputs) => inputs.filter((x) => x !== null && x !== void 0)
).setName("merge");
var createSplitNode = (count = 2) => {
  const node = new AdaptiveNode(async (input) => {
    const promises = node.outlets.slice(0, count).map((outlet) => outlet.send(input));
    await Promise.all(promises);
    return input;
  }).setName("split");
  const dataOutlets = Array.from({ length: count }, () => ({
    send: async () => {
    },
    // This will be overwritten by the connection
    connections: []
  }));
  const errorOutlet = {
    send: () => {
    },
    connections: []
  };
  node.outlets = [...dataOutlets, errorOutlet];
  return node;
};
var createRouterNode = () => new AdaptiveNode((input) => ({
  route: "default",
  data: input
})).register(AudioBuffer, (audio) => ({ route: "audio", data: audio })).register(ArrayBuffer, (buffer) => ({ route: "binary", data: buffer })).register(Array, (array) => ({ route: "array", data: array })).register(Object, (obj) => {
  if ("sampleRate" in obj) return { route: "audio", data: obj };
  if ("buffer" in obj) return { route: "binary", data: obj };
  return { route: "object", data: obj };
}).setName("router");
function createLoadBalancerNode(nodes, strategy = "round-robin") {
  let index = 0;
  const nodeHealth = new Map(nodes.map((n) => [n.id, true]));
  const loadBalancer = new AdaptiveNode(async (input) => {
    let nodesToTry = nodes.filter((n) => nodeHealth.get(n.id));
    if (nodesToTry.length === 0) {
      throw new Error("No healthy nodes available");
    }
    while (nodesToTry.length > 0) {
      let selectedNode;
      let nodeIndexInTryList;
      switch (strategy) {
        case "random":
          nodeIndexInTryList = Math.floor(Math.random() * nodesToTry.length);
          selectedNode = nodesToTry[nodeIndexInTryList];
          break;
        case "least-loaded":
          nodeIndexInTryList = 0;
          selectedNode = nodesToTry[0];
          break;
        default:
          nodeIndexInTryList = index % nodesToTry.length;
          selectedNode = nodesToTry[nodeIndexInTryList];
          index++;
          break;
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
        nodeHealth.set(selectedNode.id, false);
        nodesToTry.splice(nodeIndexInTryList, 1);
      }
    }
    throw new Error("All available nodes failed to process the input.");
  });
  return loadBalancer.setName(`loadBalance(${strategy})`);
}
function createParallelNode(nodes) {
  return new AdaptiveNode(async (input) => {
    return Promise.all(nodes.map((node) => node.process(input)));
  }).setName("parallel");
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
    if (Math.random() < 0.1) {
      for (const [k, v] of cache.entries()) {
        if (Date.now() - v.timestamp > ttl) {
          cache.delete(k);
        }
      }
    }
    return result;
  }).setName(`cache(${ttl}ms)`);
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

export {
  AdaptiveNode,
  Connection,
  Graph,
  createDelayNode,
  createThrottleNode,
  createDebounceNode,
  createErrorLoggerNode,
  createErrorRecoveryNode,
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
  createProcessor
};
