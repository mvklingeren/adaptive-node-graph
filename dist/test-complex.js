// src/core.ts
import pino from "pino";
import crypto from "crypto";
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
        logger.error(
          { reason: result.reason },
          "An error occurred in an execution branch."
        );
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
var createDelayNode = (ms) => new AdaptiveNode(async (input) => {
  await new Promise((resolve) => setTimeout(resolve, ms));
  return input;
}).setName(`delay(${ms}ms)`);
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
var createAddNode = () => new AdaptiveNode(
  (inputs) => inputs.reduce((a, b) => a + b, 0)
).setName("+");
var createGateNode = () => new AdaptiveNode(
  ([pass, data]) => pass ? data : null
).setName("gate");
var MergeNode = class extends AdaptiveNode {
  receivedInputs = [];
  expectedInputs = 0;
  constructor(expectedInputs) {
    super(async (input) => {
      this.receivedInputs.push(input);
      if (this.receivedInputs.length >= this.expectedInputs) {
        const result = [...this.receivedInputs];
        this.receivedInputs = [];
        return result;
      }
      return [];
    });
    this.expectedInputs = expectedInputs;
    this.setName("merge");
  }
  addSource(source) {
  }
};
var createMergeNode = (expectedInputs = 2) => {
  return new MergeNode(expectedInputs);
};
var SplitNode = class extends AdaptiveNode {
  dataOutletCount;
  constructor(count = 2) {
    super((input) => input);
    this.setName("split");
    this.dataOutletCount = count;
    const dataOutlets = Array.from({ length: count }, () => {
      const outlet = {
        send: async (data, graph, hooks) => {
          const promises = outlet.connections.map(
            (conn) => conn.transfer(data, graph, hooks)
          );
          await Promise.all(promises);
        },
        connections: []
      };
      return outlet;
    });
    const errorOutlet = this.outlets[1];
    this.outlets = [...dataOutlets, errorOutlet];
  }
  async process(input, graph, hooks) {
    if (graph?.isStopped) {
      return null;
    }
    hooks?.onNodeStart?.(this.id);
    if (input === null && this.initialValue !== null) {
      input = this.initialValue;
    }
    try {
      const result = await this.executeProcessor(input, graph, hooks);
      const promises = this.outlets.slice(0, this.dataOutletCount).map((outlet) => outlet.send(result, graph, hooks));
      await Promise.all(promises);
      hooks?.onNodeComplete?.(this.id, result);
      return result;
    } catch (error) {
      this.handleError(error, input, graph, hooks);
      throw error;
    }
  }
};
var createSplitNode = (count = 2) => {
  return new SplitNode(count);
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
  async process(input, graph, hooks) {
    hooks?.onNodeStart?.(this.id);
    const logger = hooks?.logger || this.logger;
    if (input === null && this.initialValue !== null) {
      input = this.initialValue;
    }
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
function createProcessor(fn, name) {
  const node = new AdaptiveNode(fn);
  if (name) node.setName(name);
  return node;
}

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
