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
var createDelayNode = (ms) => new AdaptiveNode(async (input) => {
  await new Promise((resolve) => setTimeout(resolve, ms));
  return input;
}).setName(`delay(${ms}ms)`);
var createThrottleNode = (ms) => {
  return new AdaptiveNode(
    /* @__PURE__ */ (() => {
      let lastEmit = 0;
      return (input) => {
        const now = Date.now();
        if (now - lastEmit >= ms) {
          lastEmit = now;
          return input;
        }
        return null;
      };
    })()
  ).setName(`throttle(${ms}ms)`);
};
var createErrorLoggerNode = (logger) => new AdaptiveNode((error) => {
  const log = logger || pino();
  log.error(
    {
      timestamp: new Date(error.timestamp).toISOString(),
      nodeId: error.nodeId,
      errorMessage: error.error.message,
      input: error.input
    },
    `Error in node ${error.nodeId}`
  );
  return error;
}).setName("errorLogger");
var createErrorRecoveryNode = (defaultValue, logger) => new AdaptiveNode((error) => {
  const log = logger || pino();
  log.warn(
    { nodeId: error.nodeId },
    `Recovering from error in node ${error.nodeId}. Returning default value.`
  );
  return defaultValue;
}).setName("errorRecovery");
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
function createCacheNode(processor, options = {}) {
  const { ttl = 1e3, maxSize = 100, getKey = JSON.stringify } = options;
  return new AdaptiveNode(
    /* @__PURE__ */ (() => {
      const cache = /* @__PURE__ */ new Map();
      const evictOldest = () => {
        let oldestKey;
        let oldestTimestamp = Infinity;
        for (const [key, value] of cache.entries()) {
          if (value.timestamp < oldestTimestamp) {
            oldestTimestamp = value.timestamp;
            oldestKey = key;
          }
        }
        if (oldestKey) {
          cache.delete(oldestKey);
        }
      };
      return async (input) => {
        const key = getKey(input);
        const cached = cache.get(key);
        const now = Date.now();
        if (cached && now - cached.timestamp < ttl) {
          cached.timestamp = now;
          return cached.value;
        }
        const result = await processor(input);
        if (cache.size >= maxSize) {
          evictOldest();
        }
        cache.set(key, { value: result, timestamp: now });
        return result;
      };
    })()
  ).setName(`cache(${ttl}ms)`);
}

// src/test-improvements.ts
console.log("=== Error Handling Demo ===");
async function errorHandlingDemo() {
  const graph = new Graph();
  const unreliableNode = new AdaptiveNode((input) => {
    if (input > 5) {
      throw new Error(`Value ${input} is too high!`);
    }
    return input * 2;
  }).setName("unreliable");
  const errorLogger = createErrorLoggerNode();
  const errorRecovery = createErrorRecoveryNode(0);
  graph.addNode(unreliableNode);
  graph.addNode(errorLogger);
  graph.addNode(errorRecovery);
  graph.connectError(unreliableNode, errorLogger);
  graph.connect(errorLogger, errorRecovery);
  console.log("Processing 3:", await unreliableNode.process(3));
  console.log("Processing 10:", await unreliableNode.process(10));
}
console.log("\n=== Type Safety Demo ===");
function typeSafetyDemo() {
  const numberNode = new AdaptiveNode((n) => n.toString());
  const stringNode = new AdaptiveNode((s) => s.length > 0);
  const graph = new Graph();
  graph.addNode(numberNode);
  graph.addNode(stringNode);
  graph.connect(numberNode, stringNode);
  console.log("Type-safe connections established");
}
console.log("\n=== Async Flow Control Demo ===");
async function flowControlDemo() {
  const slowNode = new AdaptiveNode(
    async (input) => {
      console.log(`Processing ${input}...`);
      await new Promise((resolve) => setTimeout(resolve, 100));
      return input * 2;
    },
    { maxConcurrent: 2 }
    // Only 2 concurrent operations
  ).setName("slowNode");
  const promises = [];
  for (let i = 0; i < 5; i++) {
    promises.push(slowNode.process(i));
  }
  console.log("Started 5 operations with max concurrency of 2");
  const results = await Promise.all(promises);
  console.log("Results:", results);
}
console.log("\n=== Time-based Operators Demo ===");
async function timeBasedDemo() {
  const graph = new Graph();
  const delayNode = createDelayNode(500);
  const throttleNode = createThrottleNode(1e3);
  console.log("Testing delay node...");
  const start = Date.now();
  await delayNode.process("delayed");
  console.log(`Delayed by ${Date.now() - start}ms`);
  console.log("\nTesting throttle node...");
  for (let i = 0; i < 5; i++) {
    const result = await throttleNode.process(i);
    console.log(`Throttle result ${i}:`, result);
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
}
console.log("\n=== Testing Utilities Demo ===");
async function testingDemo() {
  const testNode1 = new TestNode();
  const testNode2 = new TestNode((n) => n * 2);
  const graph = new Graph();
  graph.addNode(testNode1);
  graph.addNode(testNode2);
  graph.connect(testNode1, testNode2);
  await testNode1.process(5);
  await testNode1.process(10);
  try {
    testNode1.assertReceived([5, 10]);
    testNode2.assertProcessed([10, 20]);
    testNode1.assertNoErrors();
    console.log("All tests passed!");
  } catch (error) {
    console.error("Test failed:", error);
  }
}
console.log("\n=== Circuit Breaker Demo ===");
async function circuitBreakerDemo() {
  const failingNode = new AdaptiveNode(
    () => {
      throw new Error("Always fails");
    },
    {
      circuitBreakerThreshold: 3,
      circuitBreakerResetTime: 2e3
    }
  ).setName("failing");
  const errorLogger = createErrorLoggerNode();
  const graph = new Graph();
  graph.addNode(failingNode);
  graph.addNode(errorLogger);
  graph.connectError(failingNode, errorLogger);
  console.log("Attempting to trigger circuit breaker...");
  for (let i = 0; i < 5; i++) {
    await failingNode.process(i);
    console.log(`Attempt ${i + 1} completed`);
  }
  console.log("Circuit breaker should be open now");
  console.log("Waiting 2 seconds for circuit breaker reset...");
  await new Promise((resolve) => setTimeout(resolve, 2100));
  console.log("Trying again after reset...");
  await failingNode.process(99);
}
console.log("\n=== Load Balancer Demo ===");
async function loadBalancerDemo() {
  const workers = [
    new AdaptiveNode((n) => `Worker1: ${n}`).setName("worker1"),
    new AdaptiveNode((n) => `Worker2: ${n}`).setName("worker2"),
    new AdaptiveNode((n) => `Worker3: ${n}`).setName("worker3")
  ];
  const loadBalancer = createLoadBalancerNode(workers, "round-robin");
  console.log("Round-robin load balancing:");
  for (let i = 0; i < 6; i++) {
    const result = await loadBalancer.process(i);
    console.log(result);
  }
}
console.log("\n=== Cache Node Demo ===");
async function cacheDemo() {
  let computeCount = 0;
  const expensiveCompute = async (n) => {
    computeCount++;
    console.log(`Computing ${n}... (computation #${computeCount})`);
    await new Promise((resolve) => setTimeout(resolve, 100));
    return n * n;
  };
  const cachedNode = createCacheNode(expensiveCompute, 1e3, 10);
  console.log("First calls:");
  console.log(await cachedNode.process(5));
  console.log(await cachedNode.process(10));
  console.log("\nRepeated calls (cached):");
  console.log(await cachedNode.process(5));
  console.log(await cachedNode.process(10));
  console.log("\nWaiting for cache expiry...");
  await new Promise((resolve) => setTimeout(resolve, 1100));
  console.log("\nAfter cache expiry:");
  console.log(await cachedNode.process(5));
}
console.log("\n=== Sub-graph Demo ===");
async function subGraphDemo() {
  const subGraph = new Graph();
  const add1 = new AdaptiveNode((n) => n + 1).setName("add1");
  const add2 = new AdaptiveNode((n) => n + 2).setName("add2");
  subGraph.addNode(add1);
  subGraph.addNode(add2);
  subGraph.connect(add1, add2);
  const subGraphNode = new SubGraphNode(subGraph, add1.id, add2.id);
  const result = await subGraphNode.process(10);
  console.log("Sub-graph result:", result);
}
async function runAllDemos() {
  try {
    await errorHandlingDemo();
    typeSafetyDemo();
    await flowControlDemo();
    await timeBasedDemo();
    await testingDemo();
    await circuitBreakerDemo();
    await loadBalancerDemo();
    await cacheDemo();
    await subGraphDemo();
    console.log("\n=== All demos completed successfully! ===");
  } catch (error) {
    console.error("Demo error:", error);
  }
}
runAllDemos().catch(console.error);
export {
  runAllDemos
};
