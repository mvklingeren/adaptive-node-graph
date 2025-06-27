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
        if (this.outlets[0]) {
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
        logger.warn({ nodeId: node.id }, `Cycle detected in graph involving node ${node.id}`);
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
    this._isDirty = false;
  }
  async execute(input, startNodeId, hooks) {
    this.isStopped = false;
    const executionOrder = this.getExecutionOrder(hooks);
    const processNode = (node, nodeInput) => {
      return node.process(nodeInput, this, hooks);
    };
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Start node ${startNodeId} not found`);
      const nodeInput = input instanceof Map ? input.get(startNodeId) : input;
      await processNode(startNode, nodeInput);
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
    const promises = entryNodes.map((node) => {
      const nodeInput = input instanceof Map ? input.get(node.id) : input;
      return processNode(node, nodeInput);
    });
    await Promise.all(promises);
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
var createDebounceNode = (ms) => {
  let timeoutId = null;
  let lastResolve = null;
  const node = new AdaptiveNode(async (input) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    if (lastResolve) {
      lastResolve(null);
    }
    return new Promise((resolve) => {
      lastResolve = resolve;
      timeoutId = setTimeout(() => {
        resolve(input);
        timeoutId = null;
        lastResolve = null;
      }, ms);
    });
  }).setName(`debounce(${ms}ms)`);
  const originalDestroy = node.destroy.bind(node);
  node.destroy = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    originalDestroy();
  };
  return node;
};
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
  async executeProcessor(input, _graph, hooks) {
    const logger = hooks?.logger || this.logger;
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
      const result = await selectedNode.process(input, _graph, hooks);
      if (result === null) {
        throw new Error(`Node ${selectedNode.id} returned null`);
      }
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
tests["Error Handling and Recovery"] = async () => {
  const graph = new Graph();
  const unreliableNode = new AdaptiveNode((input) => {
    if (input > 5) throw new Error("Value too high");
    return input * 2;
  }).setName("unreliable");
  const errorCapture = new TestNode().setName("errorCapture");
  const recoveryNode = createErrorRecoveryNode(-1);
  const finalOutput = new TestNode().setName("finalOutput");
  graph.addNode(unreliableNode).addNode(errorCapture).addNode(recoveryNode).addNode(finalOutput);
  graph.connect(unreliableNode, finalOutput);
  graph.connectError(unreliableNode, errorCapture);
  graph.connectError(unreliableNode, recoveryNode);
  graph.connect(recoveryNode, finalOutput);
  await graph.execute(3, unreliableNode.id);
  finalOutput.assertReceived([6]);
  errorCapture.assertReceived([]);
  finalOutput.reset();
  await graph.execute(10, unreliableNode.id);
  assert.equal(errorCapture.receivedInputs.length, 1, "Error was not captured");
  finalOutput.assertReceived([-1]);
};
tests["Async Flow Control with maxConcurrent"] = async () => {
  let concurrentCount = 0;
  let maxConcurrent = 0;
  const slowNode = new AdaptiveNode(
    async (input) => {
      concurrentCount++;
      maxConcurrent = Math.max(maxConcurrent, concurrentCount);
      await new Promise((resolve) => setTimeout(resolve, 50));
      concurrentCount--;
      return input;
    },
    { maxConcurrent: 2 }
  ).setName("slowNode");
  const promises = Array.from({ length: 5 }, (_, i) => slowNode.process(i));
  await Promise.all(promises);
  assert.equal(maxConcurrent, 2, "Concurrency limit was not respected");
};
tests["Time-based Operators (Delay, Throttle, Debounce)"] = async () => {
  const delayNode = createDelayNode(100);
  const start = Date.now();
  await delayNode.process(null);
  const duration = Date.now() - start;
  assert(duration >= 100, `Delay was too short: ${duration}ms`);
  const throttleNode = createThrottleNode(100);
  const throttleOutput = new TestNode().setName("throttleOutput");
  throttleNode.outlets[0].connections.push({
    target: throttleOutput,
    transfer: async (data) => throttleOutput.process(data)
  });
  await throttleNode.process(1);
  await throttleNode.process(2);
  await new Promise((resolve) => setTimeout(resolve, 110));
  await throttleNode.process(3);
  throttleOutput.assertReceived([1, 3]);
  const debounceNode = createDebounceNode(100);
  const debounceOutput = new TestNode().setName("debounceOutput");
  debounceNode.outlets[0].connections.push({
    target: debounceOutput,
    transfer: async (data) => debounceOutput.process(data)
  });
  debounceNode.process(1);
  debounceNode.process(2);
  await new Promise((resolve) => setTimeout(resolve, 50));
  debounceNode.process(3);
  await new Promise((resolve) => setTimeout(resolve, 110));
  debounceOutput.assertReceived([3]);
};
tests["Circuit Breaker"] = async () => {
  const failingNode = new AdaptiveNode(
    () => {
      throw new Error("Failure");
    },
    { circuitBreakerThreshold: 2, circuitBreakerResetTime: 200 }
  ).setName("failing");
  const errorCapture = new TestNode().setName("errorCapture");
  failingNode.outlets[1].connections.push({
    target: errorCapture,
    transfer: async (data) => errorCapture.process(data)
  });
  await failingNode.process(1);
  await failingNode.process(2);
  assert.equal(errorCapture.receivedInputs.length, 2, "Errors not captured before break");
  await failingNode.process(3);
  assert.equal(errorCapture.receivedInputs.length, 3, "Open circuit error not captured");
  assert(errorCapture.receivedInputs[2].error.message.includes("Circuit breaker is open"));
  await new Promise((resolve) => setTimeout(resolve, 210));
  await failingNode.process(4);
  assert.equal(errorCapture.receivedInputs.length, 4, "Error not captured after reset");
  assert.equal(errorCapture.receivedInputs[3].error.message, "Failure");
};
tests["Load Balancer (Round Robin)"] = async () => {
  const worker1 = new TestNode().setName("worker1");
  const worker2 = new TestNode().setName("worker2");
  const loadBalancer = createLoadBalancerNode([worker1, worker2], { strategy: "round-robin" });
  await loadBalancer.process(1);
  await loadBalancer.process(2);
  await loadBalancer.process(3);
  worker1.assertReceived([1, 3]);
  worker2.assertReceived([2]);
};
tests["Cache Node"] = async () => {
  let computeCount = 0;
  const expensiveCompute = async (n) => {
    computeCount++;
    return n * n;
  };
  const cachedNode = createCacheNode(expensiveCompute, { ttl: 200 });
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 1);
  assert.equal(await cachedNode.process(10), 100);
  assert.equal(computeCount, 2);
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 2, "Cache was not used for first repeat");
  assert.equal(await cachedNode.process(10), 100);
  assert.equal(computeCount, 2, "Cache was not used for second repeat");
  await new Promise((resolve) => setTimeout(resolve, 210));
  assert.equal(await cachedNode.process(5), 25);
  assert.equal(computeCount, 3, "Cache did not expire");
};
tests["Sub-graph Execution"] = async () => {
  const subGraph = new Graph();
  const add1 = new AdaptiveNode((n) => n + 1).setName("add1");
  const add2 = new AdaptiveNode((n) => n + 2).setName("add2");
  subGraph.addNode(add1).addNode(add2).connect(add1, add2);
  const subGraphNode = new SubGraphNode(subGraph, add1.id, add2.id);
  const mainGraph = new Graph().addNode(subGraphNode);
  const output = new TestNode().setName("output");
  mainGraph.addNode(output).connect(subGraphNode, output);
  await mainGraph.execute(10, subGraphNode.id);
  output.assertReceived([13]);
};
runAllTests().catch((err) => {
  console.error("Unhandled error during test execution:", err);
  process.exit(1);
});
