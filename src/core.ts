// core-improved.ts
// Enhanced Core library for the Adaptive Node with Routing System
// Implements: Error Handling, Type Safety, Async Flow Control, and more

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
        send: (error: NodeError, graph?: Graph, hooks?: ExecutionHooks) => {
          errorOutletConnections.forEach((conn) => {
            conn.transfer(error, graph, hooks);
          });
        },
        connections: errorOutletConnections,
      },
    ];
  }

  register<T extends TInput>(
    type: new (...args: any[]) => T,
    processor: Processor<T, TOutput>
  ): this {
    this.processors.set(type, processor);
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

      if (this.outlets[0] && result !== null) {
        await this.outlets[0].send(result, graph, hooks);
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
    _hooks?: ExecutionHooks
  ): Promise<TOutput> {
    const start = performance.now();
    let processorName = "default";

    // Type-based processor selection
    let selectedProcessor = this.defaultProcessor;
    for (const [type, processor] of this.processors) {
      if (input instanceof (type as any)) {
        selectedProcessor = processor;
        processorName = type.name;
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
      console.error(
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
      console.error(`Unhandled error in node ${this.id}:`, nodeError);
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
      console.error(`Error in ${connError.nodeId}:`, connError.error);
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
  public isStopped = false;

  addNode(node: AdaptiveNode): this {
    this.nodes.set(node.id, node);
    this.updateExecutionOrder();
    return this;
  }

  removeNode(nodeId: string): this {
    const node = this.nodes.get(nodeId);
    if (!node) return this;

    // Remove all connections involving this node
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
    this.updateExecutionOrder();
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
    this.updateExecutionOrder();
    return connection;
  }

  disconnect(connection: Connection<any, any>): this {
    connection.disconnect();
    this.connections.delete(connection);
    this.updateExecutionOrder();
    return this;
  }

  public getExecutionOrder(): AdaptiveNode[] {
    this.updateExecutionOrder();
    return this.executionOrder;
  }

  private updateExecutionOrder(): void {
    // Simple topological sort
    const visited = new Set<string>();
    const order: AdaptiveNode[] = [];

    const visit = (node: AdaptiveNode) => {
      if (visited.has(node.id)) return;
      visited.add(node.id);

      // Visit dependencies first
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

  async execute<T>(
    input: T,
    startNodeId?: string,
    hooks?: ExecutionHooks
  ): Promise<any> {
    this.isStopped = false;
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Start node ${startNodeId} not found`);
      return startNode.process(input, this, hooks);
    }

    // Find entry nodes: nodes with no incoming data connections OR nodes with an initial value.
    const sourceNodes = this.executionOrder.filter((node) => {
      const hasIncomingConnection = Array.from(this.connections).some(
        (conn) => conn.target.id === node.id && conn.sourceOutlet === 0
      );
      const hasInitialValue = node.initialValue !== null;
      return !hasIncomingConnection || hasInitialValue;
    });

    const entryNodes =
      sourceNodes.length > 0
        ? sourceNodes
        : this.executionOrder.length > 0
        ? [this.executionOrder[0]]
        : [];

    const promises = entryNodes.map((node) => node.process(input, this, hooks));
    await Promise.all(promises);

    // Return the result from the last node in the execution order
    const lastNode = this.executionOrder[this.executionOrder.length - 1];
    return lastNode?.getLastResult();
  }

  stop() {
    this.isStopped = true;
  }

  // Parallel execution for independent nodes
  async executeParallel<T>(input: T): Promise<Map<string, any>> {
    const results = new Map<string, any>();
    const dependencies = this.calculateDependencies();
    const executed = new Set<string>();

    const canExecute = (nodeId: string): boolean => {
      const deps = dependencies.get(nodeId) || new Set();
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
  let lastEmit = 0;
  return new AdaptiveNode((input: any) => {
    const now = Date.now();
    if (now - lastEmit >= ms) {
      lastEmit = now;
      return input;
    }
    return null; // skip
  }).setName(`throttle(${ms}ms)`);
};

export const createDebounceNode = (ms: number) => {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return new AdaptiveNode(async (input: any) => {
    if (timeoutId) clearTimeout(timeoutId);
    return new Promise((resolve) => {
      timeoutId = setTimeout(() => {
        resolve(input);
        timeoutId = null;
      }, ms);
    });
  }).setName(`debounce(${ms}ms)`);
};

// ============================================================================
// Error Handling Nodes
// ============================================================================

export const createErrorLoggerNode = () =>
  new AdaptiveNode<NodeError, NodeError>((error) => {
    console.error(
      `[${new Date(error.timestamp).toISOString()}] Error in node ${
        error.nodeId
      }:`,
      error.error.message,
      "\nInput:",
      error.input
    );
    return error;
  }).setName("errorLogger");

export const createErrorRecoveryNode = <T>(defaultValue: T) =>
  new AdaptiveNode<NodeError, T>((_error) => {
    console.warn(`Recovering from error. Returning default value.`);
    return defaultValue;
  }).setName("errorRecovery");

// ============================================================================
// Testing Utilities
// ============================================================================

export class TestNode<T> extends AdaptiveNode<T, T> {
  receivedInputs: T[] = [];
  processedOutputs: T[] = [];
  errors: NodeError[] = [];

  constructor(processor?: Processor<T, T>) {
    super(
      processor ||
        ((input) => {
          this.receivedInputs.push(input);
          return input;
        })
    );

    // Override process to track outputs
    const originalProcess = this.process.bind(this);
    this.process = async (input: T) => {
      const result = await originalProcess(input);
      if (result !== null) {
        this.processedOutputs.push(result);
      }
      return result;
    };

    this.outlets[1].send = (error: NodeError) => {
      this.errors.push(error);
    };
    this.setName("test");
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

export const createMergeNode = () =>
  new AdaptiveNode<any[], any[]>((inputs) =>
    inputs.filter((x) => x !== null && x !== undefined)
  ).setName("merge");

export const createSplitNode = (count: number = 2) => {
  const node = new AdaptiveNode<any, any>(async (input) => {
    // The processor sends the input to all data outlets.
    const promises = node.outlets
      .slice(0, count) // Only send to the data outlets
      .map((outlet) => outlet.send(input));
    await Promise.all(promises);
    // The split node itself doesn't "return" a single value in the traditional sense.
    // It forwards the input to multiple downstream paths.
    // We return the original input for potential chaining if needed.
    return input;
  }).setName("split");

  // Create specified number of data outlets + 1 error outlet
  const dataOutlets: Outlet<any>[] = Array.from({ length: count }, () => ({
    send: async () => {}, // This will be overwritten by the connection
    connections: [],
  }));

  const errorOutlet: Outlet<NodeError> = {
    send: () => {},
    connections: [],
  };

  node.outlets = [...dataOutlets, errorOutlet];

  return node;
};

// Smart Routing
export const createRouterNode = () =>
  new AdaptiveNode<any, { route: string; data: any }>((input) => ({
    route: "default",
    data: input,
  }))
    .register(AudioBuffer, (audio) => ({ route: "audio", data: audio }))
    .register(ArrayBuffer, (buffer) => ({ route: "binary", data: buffer }))
    .register(Array, (array) => ({ route: "array", data: array }))
    .register(Object, (obj) => {
      if ("sampleRate" in obj) return { route: "audio", data: obj };
      if ("buffer" in obj) return { route: "binary", data: obj };
      return { route: "object", data: obj };
    })
    .setName("router");

// Load Balancer
export function createLoadBalancerNode<T, U>(
  nodes: AdaptiveNode<T, U>[],
  strategy: "round-robin" | "random" | "least-loaded" = "round-robin"
): AdaptiveNode<T, U> {
  let index = 0;
  const nodeHealth = new Map(nodes.map((n) => [n.id, true]));

  const loadBalancer = new AdaptiveNode<T, U>(async (input) => {
    let nodesToTry = nodes.filter((n) => nodeHealth.get(n.id));
    if (nodesToTry.length === 0) {
      throw new Error("No healthy nodes available");
    }

    while (nodesToTry.length > 0) {
      let selectedNode: AdaptiveNode<T, U>;
      let nodeIndexInTryList: number;

      switch (strategy) {
        case "random":
          nodeIndexInTryList = Math.floor(Math.random() * nodesToTry.length);
          selectedNode = nodesToTry[nodeIndexInTryList];
          break;
        case "least-loaded":
          // Simplified: a real implementation would need performance metrics.
          nodeIndexInTryList = 0;
          selectedNode = nodesToTry[0];
          break;
        default: // round-robin
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
  ttl: number = 1000,
  maxSize: number = 100
): AdaptiveNode<T, U> {
  const cache = new Map<string, { value: U; timestamp: number }>();

  return new AdaptiveNode<T, U>(async (input: T) => {
    const key = JSON.stringify(input);
    const cached = cache.get(key);

    if (cached && Date.now() - cached.timestamp < ttl) {
      return cached.value;
    }

    const result = await processor(input);
    cache.set(key, { value: result, timestamp: Date.now() });

    if (cache.size > maxSize) {
      const firstKey = cache.keys().next().value;
      if (firstKey !== undefined) {
        cache.delete(firstKey);
      }
    }

    // Clean expired entries periodically
    if (Math.random() < 0.1) {
      // 10% chance to clean
      for (const [k, v] of cache.entries()) {
        if (Date.now() - v.timestamp > ttl) {
          cache.delete(k);
        }
      }
    }

    return result;
  }).setName(`cache(${ttl}ms)`);
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
