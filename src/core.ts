// core-improved.ts
// Enhanced Core library for the Adaptive Node with Routing System
// Implements: Error Handling, Type Safety, Async Flow Control, and more

import pino from "pino";

// ============================================================================
// Core Types and Interfaces
// ============================================================================

export interface Processor<TInput = any, TOutput = any> {
  (input: TInput): TOutput | Promise<TOutput>;
}

export interface Inlet<T = any> {
  accept(data: T): void;
}

export interface Outlet<T = any> {
  send(data: T, graph?: Graph, hooks?: ExecutionHooks): void;
  connections: Connection<T, any>[];
}

// Error handling types
export interface NodeError {
  error: Error;
  input: any;
  nodeId: string;
  timestamp: number;
}

export interface ExecutionHooks {
  onNodeStart?: (nodeId: string) => void;
  onNodeComplete?: (nodeId: string, result: any) => void;
  logger?: pino.Logger;
}

// ============================================================================
// Enhanced AdaptiveNode with Error Handling and Flow Control
// ============================================================================

export class AdaptiveNode<TInput = any, TOutput = any> {
  private processors = new Map<Function, Processor<any, TOutput>>();
  private performanceStats = new Map<string, number[]>();

  // Flow control
  private processing = 0;
  private maxConcurrent = 10;
  private queue: Array<() => void> = [];

  // Error handling
  private errorCount = 0;
  private lastErrorTime = 0;
  private circuitBreakerThreshold = 5;
  private circuitBreakerResetTime = 60000; // 1 minute
  private isCircuitOpen = false;

  // Store last result for sub-graphs
  protected lastResult: TOutput | null = null;
  public isEmitting = true;
  public onProcess: ((id: string, data: any) => void) | null = null;
  public initialValue: any = null;

  public readonly id: string;
  public name: string;
  public inlets: Inlet<TInput>[] = [];
  public outlets: Outlet<any>[] = [];

  // Type markers for compile-time type checking
  readonly inputType!: TInput;
  readonly outputType!: TOutput;

  constructor(
    private defaultProcessor: Processor<TInput, TOutput>,
    private options: {
      maxConcurrent?: number;
      circuitBreakerThreshold?: number;
      circuitBreakerResetTime?: number;
    } = {}
  ) {
    this.maxConcurrent = options.maxConcurrent || 10;
    this.circuitBreakerThreshold = options.circuitBreakerThreshold || 5;
    this.circuitBreakerResetTime = options.circuitBreakerResetTime || 60000;
    this.id = `node_${crypto.randomUUID()}`;
    this.name = this.constructor.name;
    this.setupInOut();
  }

  private setupInOut(): void {
    this.inlets = [{ accept: (data: TInput) => this.process(data) }];

    const dataOutletConnections: Connection<TOutput, any>[] = [];
    const errorOutletConnections: Connection<NodeError, any>[] = [];

    this.outlets = [
      {
        send: async (data: TOutput, graph?: Graph, hooks?: ExecutionHooks) => {
          const promises = dataOutletConnections.map((conn) =>
            conn.transfer(data, graph, hooks)
          );
          await Promise.all(promises);
        },
        connections: dataOutletConnections,
      },
      {
        send: async (error: NodeError, graph?: Graph, hooks?: ExecutionHooks) => {
          const promises = errorOutletConnections.map((conn) =>
            conn.transfer(error, graph, hooks)
          );
          await Promise.all(promises);
        },
        connections: errorOutletConnections,
      },
    ];
  }

  register<T extends TInput>(
    predicate: (input: any) => input is T,
    processor: Processor<T, TOutput>
  ): this {
    this.processors.set(predicate, processor);
    return this;
  }

  async process(
    input: TInput,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<TOutput | null> {
    if (graph?.isStopped) return null;

    hooks?.onNodeStart?.(this.id);

    if (this.isEmitting && this.onProcess) {
      this.onProcess(this.id, input);
    }

    if (input === null && this.initialValue !== null) {
      input = this.initialValue;
    }

    // Circuit breaker check
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

    // Flow control - wait if at capacity
    if (this.processing >= this.maxConcurrent) {
      await new Promise<void>((resolve) => this.queue.push(resolve));
    }

    this.processing++;

    let result: TOutput | null = null;
    try {
      result = await this.executeProcessor(input, graph, hooks);

      if (result !== null) {
        if (this.outlets[0]) {
          await this.outlets[0].send(result, graph, hooks);
        }
      }

      return result;
    } catch (error) {
      this.handleError(error as Error, input, graph, hooks);
      return null;
    } finally {
      this.processing--;
      if (this.queue.length > 0) {
        this.queue.shift()?.();
      }
      hooks?.onNodeComplete?.(this.id, result);
    }
  }

  protected async executeProcessor(
    input: TInput,
    _graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<TOutput> {
    const start = performance.now();
    let processorName = "default";

    // Type-based processor selection
    let selectedProcessor = this.defaultProcessor;
    processorName = "default";
    for (const [predicate, processor] of this.processors) {
      if ((predicate as (input: any) => boolean)(input)) {
        selectedProcessor = processor;
        processorName = `predicate:${predicate.toString()}`;
        break;
      }
    }

    const result = await selectedProcessor(input);
    this.recordPerformance(processorName, performance.now() - start);

    // Store last result
    this.lastResult = result;

    // Reset error count on success
    this.errorCount = 0;

    return result;
  }

  protected handleError(
    error: Error,
    input: TInput,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): void {
    this.errorCount++;
    this.lastErrorTime = Date.now();

    // Check circuit breaker
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

  protected sendError(
    error: Error,
    input: TInput,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): void {
    const nodeError: NodeError = {
      error,
      input,
      nodeId: this.id,
      timestamp: Date.now(),
    };

    if (this.outlets[1]?.connections.length > 0) {
      this.outlets[1].send(nodeError, graph, hooks);
    } else {
      const logger = hooks?.logger || pino();
      logger.error({ error: nodeError }, `Unhandled error in node ${this.id}`);
    }
  }

  protected recordPerformance(processorName: string, duration: number): void {
    if (!this.performanceStats.has(processorName)) {
      this.performanceStats.set(processorName, []);
    }
    const stats = this.performanceStats.get(processorName)!;
    stats.push(duration);

    // Keep only last 100 measurements
    if (stats.length > 100) {
      stats.shift();
    }
  }

  getPerformanceStats(): Map<
    string,
    { avg: number; min: number; max: number }
  > {
    const result = new Map();
    for (const [name, durations] of this.performanceStats) {
      const avg = durations.reduce((a, b) => a + b, 0) / durations.length;
      const min = Math.min(...durations);
      const max = Math.max(...durations);
      result.set(name, { avg, min, max });
    }
    return result;
  }

  getLastResult(): TOutput | null {
    return this.lastResult;
  }

  setName(name: string): this {
    this.name = name;
    return this;
  }

  setInitialValue(value: any): this {
    this.initialValue = value;
    return this;
  }

  setEmitting(
    isEmitting: boolean,
    onProcess?: (id: string, data: any) => void
  ): this {
    this.isEmitting = isEmitting;
    if (onProcess) {
      this.onProcess = onProcess;
    }
    return this;
  }

  // New methods for flow control
  setMaxConcurrent(max: number): this {
    this.maxConcurrent = max;
    return this;
  }

  resetCircuitBreaker(): this {
    this.isCircuitOpen = false;
    this.errorCount = 0;
    return this;
  }

  /**
   * Cleans up any resources used by the node, like timers or intervals.
   * Override in subclasses for custom cleanup logic.
   */
  destroy(): void {
    // Base implementation does nothing.
  }
}

// ============================================================================
// Type-Safe Connection Implementation
// ============================================================================

export class Connection<TData, TTransformed = TData> {
  constructor(
    public readonly source: AdaptiveNode<any, TData>,
    public readonly sourceOutlet: number,
    public readonly target: AdaptiveNode<TTransformed, any>,
    public readonly targetInlet: number,
    private transformer?: Processor<TData, TTransformed>
  ) {
    this.source.outlets[this.sourceOutlet].connections.push(this);
  }

  async transfer(
    data: TData,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<void> {
    try {
      const transformedData = this.transformer
        ? await this.transformer(data)
        : (data as unknown as TTransformed);
      await this.target.process(transformedData, graph, hooks);
    } catch (error) {
      const connError: NodeError = {
        error: error as Error,
        input: data,
        nodeId: `connection-${this.source.id}-${this.target.id}`,
        timestamp: Date.now(),
      };
      const logger = hooks?.logger || pino();
      logger.error(
        {
          sourceNode: this.source.id,
          targetNode: this.target.id,
          error: connError.error,
        },
        `Error in connection from ${this.source.id} to ${this.target.id}`
      );
      // Forward the error to the target's error outlet
      this.target.outlets[1]?.send(connError, graph, hooks);
    }
  }

  disconnect(): void {
    const outlet = this.source.outlets[this.sourceOutlet];
    if (outlet) {
      const index = outlet.connections.indexOf(this);
      if (index > -1) {
        outlet.connections.splice(index, 1);
      }
    }
  }
}

// ============================================================================
// Enhanced Graph Implementation
// ============================================================================

export class Graph {
  public nodes = new Map<string, AdaptiveNode>();
  public connections = new Set<Connection<any, any>>();
  private executionOrder: AdaptiveNode[] = [];
  private _isDirty = true;
  public isStopped = false;

  addNode(node: AdaptiveNode): this {
    this.nodes.set(node.id, node);
    this._isDirty = true;
    return this;
  }

  removeNode(nodeId: string): this {
    const node = this.nodes.get(nodeId);
    if (!node) return this;

    // Clean up node-specific resources
    node.destroy();

    // Remove all connections involving this node
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

  getNode(nodeId: string): AdaptiveNode | undefined {
    return this.nodes.get(nodeId);
  }

  // Type-safe connect method
  connect<A, B>(
    source: AdaptiveNode<any, A>,
    target: AdaptiveNode<A, B>,
    transformer?: Processor<A, A>,
    sourceOutlet: number = 0,
    targetInlet: number = 0
  ): Connection<A, A> {
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
  connectError(
    source: AdaptiveNode,
    errorHandler: AdaptiveNode<NodeError, any>
  ): Connection<NodeError, NodeError> {
    const connection = new Connection(
      source,
      1, // error outlet
      errorHandler,
      0, // default inlet
      undefined
    );
    this.connections.add(connection);
    this._isDirty = true;
    return connection;
  }

  disconnect(connection: Connection<any, any>): this {
    connection.disconnect();
    this.connections.delete(connection);
    this._isDirty = true;
    return this;
  }

  public getExecutionOrder(hooks?: ExecutionHooks): AdaptiveNode[] {
    if (this._isDirty) {
      this.updateExecutionOrder(hooks);
    }
    return this.executionOrder;
  }

  private updateExecutionOrder(hooks?: ExecutionHooks): void {
    const logger = hooks?.logger || pino();
    // More robust topological sort using DFS and cycle detection
    const order: AdaptiveNode[] = [];
    const visiting = new Set<string>(); // For detecting cycles (nodes currently in recursion stack)
    const visited = new Set<string>(); // For tracking all visited nodes

    const visit = (node: AdaptiveNode) => {
      if (visited.has(node.id)) {
        return;
      }
      if (visiting.has(node.id)) {
        logger.warn({ nodeId: node.id }, `Cycle detected in graph involving node ${node.id}`);
        return;
      }

      visiting.add(node.id);

      // Visit dependencies first
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

  async execute<T>(
    input: T | Map<string, any>,
    startNodeId?: string,
    hooks?: ExecutionHooks
  ): Promise<any> {
    this.isStopped = false;
    const executionOrder = this.getExecutionOrder(hooks);

    const processNode = (node: AdaptiveNode, nodeInput: any) => {
      return node.process(nodeInput, this, hooks);
    };

    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Start node ${startNodeId} not found`);
      const nodeInput = input instanceof Map ? input.get(startNodeId) : input;
      await processNode(startNode, nodeInput);
      // After execution, return the result from the last node in the execution order
      const lastNode = executionOrder[executionOrder.length - 1];
      return lastNode?.getLastResult();
    }

    // Find entry nodes: nodes with no incoming data connections OR nodes with an initial value.
    const entryNodes = executionOrder.filter((node) => {
      const hasIncomingConnection = Array.from(this.connections).some(
        (conn) => conn.target.id === node.id && conn.sourceOutlet === 0
      );
      const hasInitialValue = node.initialValue !== null;
      return !hasIncomingConnection || hasInitialValue;
    });

    if (entryNodes.length === 0 && this.nodes.size > 0) {
      // If no clear entry nodes are found (e.g., in a graph with cycles),
      // the behavior is to fall back to the first node in the topological sort.
      // Note: This may not be the desired entry point for all cyclic graphs.
      // For predictable execution in such cases, explicitly provide a `startNodeId`.
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
        `Multiple entry nodes found (${entryNodes
          .map((n) => n.id)
          .join(
            ", "
          )}). Please specify a startNodeId or provide a Map of inputs.`
      );
    }

    const promises = entryNodes.map((node) => {
      const nodeInput = input instanceof Map ? input.get(node.id) : input;
      return processNode(node, nodeInput);
    });
    await Promise.all(promises);

    // Return the result from the last node in the execution order
    const lastNode = executionOrder[executionOrder.length - 1];
    return lastNode?.getLastResult();
  }

  stop() {
    this.isStopped = true;
  }

  // Parallel execution for independent nodes
  async executeParallel<T>(
    initialInputs: Map<string, T>,
    hooks?: ExecutionHooks
  ): Promise<Map<string, any>> {
    this.isStopped = false;
    const results = new Map<string, any>();
    const dependencies = this.calculateDependencies();
    const inDegree = new Map<string, number>();
    const queue: AdaptiveNode[] = [];

    // Initialize in-degrees and find initial nodes
    for (const node of this.nodes.values()) {
      const deps = dependencies.get(node.id) || new Set();
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
            executedNodes: results.size,
          },
          "Parallel execution stalled. A cycle may be present or graph is disconnected."
        );
        break;
      }

      const node = queue.shift()!;
      processedCount++;

      // Aggregate inputs from dependencies
      const nodeDependencies = dependencies.get(node.id) || new Set();
      const aggregatedInputs = Array.from(nodeDependencies).map((depId) =>
        results.get(depId)
      );

      // Use initial input if provided, otherwise use aggregated inputs.
      // If only one dependency, pass its result directly, not in an array.
      const input =
        initialInputs.get(node.id) ??
        (aggregatedInputs.length === 1
          ? aggregatedInputs[0]
          : aggregatedInputs);

      const result = await node.process(input, this, hooks);
      results.set(node.id, result);

      // Decrement in-degree of downstream nodes
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
          executedNodes: results.size,
        },
        "Parallel execution did not execute all nodes."
      );
    }

    return results;
  }

  private calculateDependencies(): Map<string, Set<string>> {
    const deps = new Map<string, Set<string>>();

    for (const conn of this.connections) {
      if (conn.sourceOutlet === 0) {
        // Only data connections, not error connections
        if (!deps.has(conn.target.id)) {
          deps.set(conn.target.id, new Set());
        }
        deps.get(conn.target.id)!.add(conn.source.id);
      }
    }

    return deps;
  }
}

// ============================================================================
// Time-based Operators
// ============================================================================

export const createDelayNode = (ms: number) =>
  new AdaptiveNode(async (input: any) => {
    await new Promise((resolve) => setTimeout(resolve, ms));
    return input;
  }).setName(`delay(${ms}ms)`);

export const createThrottleNode = (ms: number) => {
  return new AdaptiveNode(
    (() => {
      let lastEmit = 0;
      return (input: any) => {
        const now = Date.now();
        if (now - lastEmit >= ms) {
          lastEmit = now;
          return input;
        }
        return null; // skip
      };
    })()
  ).setName(`throttle(${ms}ms)`);
};

export const createDebounceNode = (ms: number) => {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let lastResolve: ((value: any) => void) | null = null;

  const node = new AdaptiveNode(async (input: any) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    if (lastResolve) {
      lastResolve(null); // Resolve previous promise to prevent hanging
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

  // Override destroy to clean up timer
  const originalDestroy = node.destroy.bind(node);
  node.destroy = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    originalDestroy();
  };

  return node;
};

// ============================================================================
// Error Handling Nodes
// ============================================================================

export const createErrorLoggerNode = (logger?: pino.Logger) =>
  new AdaptiveNode<NodeError, NodeError>((error) => {
    const log = logger || pino();
    log.error(
      {
        timestamp: new Date(error.timestamp).toISOString(),
        nodeId: error.nodeId,
        errorMessage: error.error.message,
        input: error.input,
      },
      `Error in node ${error.nodeId}`
    );
    return error;
  }).setName("errorLogger");

export const createErrorRecoveryNode = <T>(
  defaultValue: T,
  logger?: pino.Logger
) =>
  new AdaptiveNode<NodeError, T>((error) => {
    const log = logger || pino();
    log.warn(
      { nodeId: error.nodeId },
      `Recovering from error in node ${error.nodeId}. Returning default value.`
    );
    return defaultValue;
  }).setName("errorRecovery");

// ============================================================================
// Testing Utilities
// ============================================================================

export class TestNode<T> extends AdaptiveNode<T, T> {
  public receivedInputs: T[] = [];
  public processedOutputs: (T | null)[] = [];
  public errors: NodeError[] = [];

  constructor(processor?: Processor<T, T>) {
    const capturingProcessor: Processor<T, T> = async (input) => {
      this.receivedInputs.push(input);
      const output = await (processor ? processor(input) : input);
      this.processedOutputs.push(output);
      return output;
    };

    super(capturingProcessor);
    this.setName("test");

    // Connect a dedicated error handler to capture errors
    const errorCollector = new AdaptiveNode<NodeError, void>((error) => {
      this.errors.push(error);
    }).setName("testErrorCollector");

    // This connection is internal to the TestNode's setup
    new Connection(this, 1, errorCollector, 0);
  }

  reset(): void {
    this.receivedInputs = [];
    this.processedOutputs = [];
    this.errors = [];
  }

  assertReceived(expected: T[]): void {
    if (JSON.stringify(this.receivedInputs) !== JSON.stringify(expected)) {
      throw new Error(
        `Expected inputs ${JSON.stringify(expected)}, got ${JSON.stringify(
          this.receivedInputs
        )}`
      );
    }
  }

  assertProcessed(expected: T[]): void {
    if (JSON.stringify(this.processedOutputs) !== JSON.stringify(expected)) {
      throw new Error(
        `Expected outputs ${JSON.stringify(expected)}, got ${JSON.stringify(
          this.processedOutputs
        )}`
      );
    }
  }

  assertNoErrors(): void {
    if (this.errors.length > 0) {
      throw new Error(
        `Expected no errors, got ${this.errors.length}: ${this.errors
          .map((e) => e.error.message)
          .join(", ")}`
      );
    }
  }
}

export async function testGraph(
  graph: Graph,
  input: any,
  expectedOutput: any,
  startNodeId?: string
): Promise<void> {
  const output = await graph.execute(input, startNodeId);
  if (JSON.stringify(output) !== JSON.stringify(expectedOutput)) {
    throw new Error(
      `Expected output ${JSON.stringify(expectedOutput)}, got ${JSON.stringify(
        output
      )}`
    );
  }
}

// ============================================================================
// Sub-graph Support
// ============================================================================

export class SubGraphNode<TInput = any, TOutput = any> extends AdaptiveNode<
  TInput,
  TOutput
> {
  constructor(
    private subGraph: Graph,
    private inputNodeId?: string,
    private outputNodeId?: string
  ) {
    super(async () => null as unknown as TOutput);
    this.setName("subgraph");
  }

  protected async executeProcessor(
    input: TInput,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<TOutput> {
    // Execute the sub-graph, passing the hooks down
    const result = await this.subGraph.execute(input, this.inputNodeId, hooks);

    let finalResult: TOutput;

    // If output node specified, get its result
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

    // Store the final result in the SubGraphNode itself
    this.lastResult = finalResult;
    return finalResult;
  }

  getSubGraph(): Graph {
    return this.subGraph;
  }
}

// ============================================================================
// Enhanced Core Node Library (keeping originals, adding new ones)
// ============================================================================

// Math Operators
export const createAddNode = () =>
  new AdaptiveNode<number[], number>((inputs) =>
    inputs.reduce((a, b) => a + b, 0)
  ).setName("+");

export const createMultiplyNode = () =>
  new AdaptiveNode<number[], number>((inputs) =>
    inputs.reduce((a, b) => a * b, 1)
  ).setName("*");

export const createSubtractNode = () =>
  new AdaptiveNode<[number, number], number>(([a, b]) => a - b).setName("-");

export const createDivideNode = () =>
  new AdaptiveNode<[number, number], number>(([a, b]) =>
    b !== 0 ? a / b : 0
  ).setName("/");

// Float32Array Operators
export const createFloat32MultiplyNode = () =>
  new AdaptiveNode<Float32Array, Float32Array>((input) => {
    const result = new Float32Array(input?.length);
    for (let i = 0; i < input?.length; i++) {
      result[i] = input[i] * 0.5;
    }
    return result;
  }).setName("f32*");

// Logic Operators
export const createConditionalNode = () =>
  new AdaptiveNode<[boolean, any, any], any>(([condition, ifTrue, ifFalse]) =>
    condition ? ifTrue : ifFalse
  ).setName("?");

export const createAndNode = () =>
  new AdaptiveNode<boolean[], boolean>((inputs) =>
    inputs.every(Boolean)
  ).setName("&&");

export const createOrNode = () =>
  new AdaptiveNode<boolean[], boolean>((inputs) =>
    inputs.some(Boolean)
  ).setName("||");

export const createNotNode = () =>
  new AdaptiveNode<boolean, boolean>((input) => !input).setName("!");

// Data Flow Operators
export const createGateNode = () =>
  new AdaptiveNode<[boolean, any], any | null>(([pass, data]) =>
    pass ? data : null
  ).setName("gate");

export class MergeNode<T> extends AdaptiveNode<T, T[]> {
  private receivedInputs: T[] = [];
  private expectedInputs: number = 0;

  constructor(expectedInputs: number) {
    super(async (input: T) => {
      this.receivedInputs.push(input);

      if (this.receivedInputs.length >= this.expectedInputs) {
        const result = [...this.receivedInputs];
        this.receivedInputs = []; // Reset for next execution
        return result;
      }

      return []; // Not all inputs received yet, return empty array
    });
    this.expectedInputs = expectedInputs;
    this.setName("merge");
  }

  addSource(source: AdaptiveNode<any, T>): void {
    // This method is a placeholder for a more robust implementation
    // that would dynamically adjust expectedInputs or use a different
    // mechanism to determine when to emit.
  }
}

export const createMergeNode = <T>(expectedInputs: number = 2) => {
  return new MergeNode<T>(expectedInputs);
};

export class SplitNode<T = any> extends AdaptiveNode<T, T> {
  private readonly dataOutletCount: number;

  constructor(count: number = 2) {
    super((input) => input); // The processor just passes the input through.
    this.setName("split");
    this.dataOutletCount = count;

    // Create specified number of data outlets
    const dataOutlets: Outlet<T>[] = Array.from({ length: count }, () => {
      const outlet: Outlet<T> = {
        send: async (data: T, graph?: Graph, hooks?: ExecutionHooks) => {
          const promises = outlet.connections.map((conn) =>
            conn.transfer(data, graph, hooks)
          );
          await Promise.all(promises);
        },
        connections: [],
      };
      return outlet;
    });

    // The error outlet is at the end.
    const errorOutlet: Outlet<NodeError> = this.outlets[1];
    this.outlets = [...dataOutlets, errorOutlet];
  }

  async process(
    input: T,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<T | null> {
    const result = await super.process(input, graph, hooks);

    if (result !== null) {
      const promises = this.outlets
        .slice(0, this.dataOutletCount)
        .map((outlet) => outlet.send(result, graph, hooks));
      await Promise.all(promises);
    }

    return result;
  }
}

export const createSplitNode = (count: number = 2) => {
  return new SplitNode(count);
};

// Smart Routing
export const createRouterNode = () =>
  new AdaptiveNode<any, { route: string; data: any }>((input) => ({
    route: "default",
    data: input,
  }))
    .register(
      (input): input is AudioBuffer => input instanceof AudioBuffer,
      (audio) => ({ route: "audio", data: audio })
    )
    .register(
      (input): input is ArrayBuffer => input instanceof ArrayBuffer,
      (buffer) => ({ route: "binary", data: buffer })
    )
    .register(
      (input): input is any[] => Array.isArray(input),
      (array) => ({ route: "array", data: array })
    )
    .register(
      (input): input is object => typeof input === "object" && input !== null,
      (obj) => {
        if ("sampleRate" in obj) return { route: "audio", data: obj };
        if ("buffer" in obj) return { route: "binary", data: obj };
        return { route: "object", data: obj };
      }
    )
    .setName("router");

// Load Balancer
class LoadBalancerNode<T, U> extends AdaptiveNode<T, U> {
  private index = 0;
  private nodeHealth: Map<string, { healthy: boolean; lastCheck: number }>;
  private healthCheckTimer: ReturnType<typeof setInterval>;

  constructor(
    private nodes: AdaptiveNode<T, U>[],
    private strategy: "round-robin" | "random" = "round-robin",
    healthCheckInterval: number = 30000,
    private logger: pino.Logger = pino()
  ) {
    super(async () => null as U); // The actual logic is in executeProcessor
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

  protected async executeProcessor(
    input: T,
    _graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<U> {
    const logger = hooks?.logger || this.logger;
    const nodesToTry = this.nodes.filter(
      (n) => this.nodeHealth.get(n.id)?.healthy
    );
    if (nodesToTry.length === 0) {
      throw new Error("No healthy nodes available in load balancer.");
    }

    let selectedNode: AdaptiveNode<T, U> | undefined;

    switch (this.strategy) {
      case "random":
        selectedNode =
          nodesToTry[Math.floor(Math.random() * nodesToTry.length)];
        break;
      default: // round-robin
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

  destroy(): void {
    clearInterval(this.healthCheckTimer);
    super.destroy();
  }
}

export function createLoadBalancerNode<T, U>(
  nodes: AdaptiveNode<T, U>[],
  options: {
    strategy?: "round-robin" | "random";
    healthCheckInterval?: number; // in ms
  } = {}
): AdaptiveNode<T, U> {
  const { strategy = "round-robin", healthCheckInterval = 30000 } = options;
  return new LoadBalancerNode(nodes, strategy, healthCheckInterval).setName(
    `loadBalance(${strategy})`
  );
}

// Parallel Processor
export function createParallelNode<T, U>(
  nodes: AdaptiveNode<T, U>[]
): AdaptiveNode<T, (U | null)[]> {
  return new AdaptiveNode<T, (U | null)[]>(async (input) => {
    return Promise.all(nodes.map((node) => node.process(input)));
  }).setName("parallel");
}

// Cache Node
export function createCacheNode<T, U>(
  processor: Processor<T, U>,
  options: {
    ttl?: number;
    maxSize?: number;
    getKey?: (input: T) => string;
  } = {}
): AdaptiveNode<T, U> {
  const { ttl = 1000, maxSize = 100, getKey = JSON.stringify } = options;

  return new AdaptiveNode<T, U>(
    (() => {
      const cache = new Map<string, { value: U; timestamp: number }>();

      const evictOldest = () => {
        let oldestKey: string | undefined;
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

      return async (input: T) => {
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

// ============================================================================
// Oscillator Node
// ============================================================================

export interface OscillatorParams {
  frequency: number;
  amplitude: number;
  sampleRate: number;
  length?: number;
  waveform?: "sine" | "square" | "sawtooth" | "triangle";
  phase?: number; // Allow passing in phase
}

export class OscillatorNode extends AdaptiveNode<
  OscillatorParams,
  { samples: Float32Array; nextPhase: number }
> {
  constructor() {
    super((params: OscillatorParams) => {
      const {
        frequency,
        amplitude,
        sampleRate,
        length = 128,
        waveform = "sine",
        phase = 0,
      } = params;

      const samples = new Float32Array(length);
      const phaseIncrement = (2 * Math.PI * frequency) / sampleRate;
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
            samples[i] =
              (2 *
                Math.abs(
                  2 *
                    (currentPhase / (2 * Math.PI) -
                      Math.floor(currentPhase / (2 * Math.PI) + 0.5))
                ) -
                1) *
              amplitude;
            break;
        }
        currentPhase += phaseIncrement;
      }

      const nextPhase = currentPhase % (2 * Math.PI);
      return { samples, nextPhase };
    });

    this.setName("oscillator");
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

export function createProcessor<TIn, TOut>(
  fn: Processor<TIn, TOut>,
  name?: string
): AdaptiveNode<TIn, TOut> {
  const node = new AdaptiveNode(fn);
  if (name) node.setName(name);
  return node;
}
