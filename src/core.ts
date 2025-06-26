"use client";
// @ts-nocheck

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

export interface NodeVisual {
  x: number;
  y: number;
  label: string;
  inlets: Inlet[];
  outlets: Outlet[];
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
  public code: string | null = null;
  public initialValue: any = null;
  public initialValueType:
    | "string"
    | "number"
    | "boolean"
    | "json"
    | "floatarray" = "string";

  public readonly id: string = Math.random().toString(36).substr(2, 9);
  public readonly visual: NodeVisual = {
    x: 0,
    y: 0,
    label: "",
    inlets: [],
    outlets: [],
  };

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
    this.setupInletsOutlets();
  }

  private setupInletsOutlets(): void {
    // Default: 1 inlet, 2 outlets (data + error)
    this.visual.inlets = [
      {
        accept: (data: TInput) => this.process(data),
      },
    ];

    const dataOutletConnections: Connection<TOutput, any>[] = [];
    const errorOutletConnections: Connection<NodeError, any>[] = [];

    this.visual.outlets = [
      // Data outlet
      {
        send: async (data: TOutput, graph?: Graph, hooks?: ExecutionHooks) => {
          console.log(`Node ${this.id} data outlet send called with hooks:`, hooks ? 'hooks present' : 'no hooks');
          console.log(`Node ${this.id} has ${dataOutletConnections.length} connections`);
          const promises = dataOutletConnections.map((conn) => {
            console.log(`Node ${this.id} sending data to connection ${conn.source.id} -> ${conn.target.id}`);
            return conn.transfer(data, graph, hooks);
          });
          await Promise.all(promises);
        },
        connections: dataOutletConnections,
      },
      // Error outlet
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
    if (graph?.isStopped) {
      return null;
    }

    console.log(`Node ${this.id} process called with hooks:`, hooks ? 'hooks present' : 'no hooks');
    
    // Make a local copy of hooks to ensure it's not lost in async operations
    const localHooks = hooks;
    
    localHooks?.onNodeStart?.(this.id);

    if (this.isEmitting && this.onProcess) {
      this.onProcess(this.id, input);
    }

    if (this.code && this.code.trim() !== "") {
      try {
        // eslint-disable-next-line no-eval
        const result = eval(this.code);
        return result;
      } catch (error) {
        this.handleError(error as Error, input, graph, hooks);
        return null;
      }
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
        this.sendError(new Error("Circuit breaker is open"), input, graph, hooks);
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

      // Send to data outlet
      if (this.visual.outlets[0] && result !== null) {
        await this.visual.outlets[0].send(result, graph, hooks);
      }

      return result;
    } catch (error) {
      this.handleError(error as Error, input, graph, hooks);
      return null;
    } finally {
      this.processing--;
      const nextInQueue = this.queue.shift();
      if (nextInQueue) nextInQueue();
      
      // Use the local copy of hooks to ensure it's not lost
      localHooks?.onNodeComplete?.(this.id, result);
    }
  }

  protected async executeProcessor(
    input: TInput,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<TOutput> {
    const start = performance.now();
    let processorName = "default";

    // Type-based processor selection
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

    // Store last result
    this.lastResult = result;

    // Reset error count on success
    this.errorCount = 0;

    return result;
  }

  protected handleError(error: Error, input: TInput, graph?: Graph, hooks?: ExecutionHooks): void {
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

  protected sendError(error: Error, input: TInput, graph?: Graph, hooks?: ExecutionHooks): void {
    const nodeError: NodeError = {
      error,
      input,
      nodeId: this.id,
      timestamp: Date.now(),
    };

    // Send to error outlet if connected
    if (
      this.visual.outlets[1] &&
      this.visual.outlets[1].connections.length > 0
    ) {
      this.visual.outlets[1].send(nodeError, graph, hooks);
    } else {
      // Log if no error handler connected
      console.error(`Unhandled error in node ${this.id}:`, error);
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

  setPosition(x: number, y: number): this {
    this.visual.x = x;
    this.visual.y = y;
    return this;
  }

  setLabel(label: string): this {
    this.visual.label = label;
    return this;
  }

  setCode(code: string): this {
    this.code = code;
    return this;
  }

  setInitialValue(value: any): this {
    this.initialValue = value;
    return this;
  }

  setInitialValueType(
    type: "string" | "number" | "boolean" | "json" | "floatarray"
  ): this {
    this.initialValueType = type;
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
    // Register this connection with the source outlet
    this.source.visual.outlets[this.sourceOutlet].connections.push(this);
  }

  async transfer(
    data: TData,
    graph?: Graph,
    hooks?: ExecutionHooks
  ): Promise<void> {
    try {
      console.log(`Connection ${this.source.id} -> ${this.target.id} transfer called with hooks:`, hooks ? 'hooks present' : 'no hooks');
      const transformed = this.transformer
        ? await this.transformer(data)
        : (data as unknown as TTransformed);

      await this.target.process(transformed, graph, hooks);
    } catch (error) {
      console.error(
        `Error in connection ${this.source.id} -> ${this.target.id}:`,
        error
      );
      // Send error to target's error handling
      if (this.target.visual.outlets[1]) {
        this.target.visual.outlets[1].send({
          error: error as Error,
          input: data,
          nodeId: `connection-${this.source.id}-${this.target.id}`,
          timestamp: Date.now(),
        });
      }
    }
  }

  disconnect(): void {
    const outlet = this.source.visual.outlets[this.sourceOutlet];
    const index = outlet.connections.indexOf(this);
    if (index > -1) {
      outlet.connections.splice(index, 1);
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

  async execute<T>(
    input: T,
    startNodeId?: string,
    hooks?: ExecutionHooks
  ): Promise<any> {
    this.isStopped = false;
    console.log(`Graph.execute called with hooks:`, hooks ? 'hooks present' : 'no hooks');
    
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Node ${startNodeId} not found`);
      console.log(`Executing start node ${startNodeId}`);
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

    console.log(`Found ${entryNodes.length} entry nodes:`, entryNodes.map(node => node.id));
    
    // Execute all entry nodes with the initial input
    const promises = entryNodes.map((node) => {
      console.log(`Executing entry node ${node.id}`);
      return node.process(input, this, hooks);
    });
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

  toJSON(): object {
    return {
      nodes: Array.from(this.nodes.values()).map((node) => ({
        id: node.id,
        label: node.visual.label,
        x: node.visual.x,
        y: node.visual.y,
      })),
      connections: Array.from(this.connections).map((conn) => ({
        source: conn.source.id,
        sourceOutlet: conn.sourceOutlet,
        target: conn.target.id,
        targetInlet: conn.targetInlet,
      })),
    };
  }
}

// ============================================================================
// Time-based Operators
// ============================================================================

export const createDelayNode = (ms: number) =>
  new AdaptiveNode(async (input: any) => {
    await new Promise((resolve) => setTimeout(resolve, ms));
    return input;
  }).setLabel(`delay(${ms}ms)`);

export const createThrottleNode = (ms: number) => {
  let lastEmit = 0;
  return new AdaptiveNode((input: any) => {
    const now = Date.now();
    if (now - lastEmit >= ms) {
      lastEmit = now;
      return input;
    }
    return null; // skip
  }).setLabel(`throttle(${ms}ms)`);
};

export const createDebounceNode = (ms: number) => {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let resolver: ((value: any) => void) | null = null;

  return new AdaptiveNode(async (input: any) => {
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
  }).setLabel("errorLogger");

export const createErrorRecoveryNode = <T>(defaultValue: T) =>
  new AdaptiveNode<NodeError, T>((error) => {
    console.warn(`Recovering from error with default value:`, defaultValue);
    return defaultValue;
  }).setLabel("errorRecovery");

export const createRetryNode = <T>(
  maxRetries: number = 3,
  delayMs: number = 1000
) => {
  const retryMap = new Map<string, number>();

  return new AdaptiveNode<NodeError, T | null>(async (error) => {
    const key = `${error.nodeId}-${JSON.stringify(error.input)}`;
    const retries = retryMap.get(key) || 0;

    if (retries < maxRetries) {
      retryMap.set(key, retries + 1);
      await new Promise((resolve) =>
        setTimeout(resolve, delayMs * (retries + 1))
      );

      // Attempt to reprocess in the original node
      // This would need access to the original node - simplified here
      console.log(`Retrying (${retries + 1}/${maxRetries})...`);
      return null;
    } else {
      console.error(`Max retries (${maxRetries}) exceeded`);
      retryMap.delete(key);
      return null;
    }
  }).setLabel(`retry(${maxRetries})`);
};

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

    // Connect error tracking
    this.visual.outlets[1].send = (error: NodeError) => {
      this.errors.push(error);
    };

    this.setLabel("test");
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
    // We pass a dummy processor to the super constructor, as we're overriding executeProcessor.
    super(async () => null as unknown as TOutput);
    this.setLabel("subgraph");
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

// Math Operators (original)
export const createAddNode = () =>
  new AdaptiveNode<number[], number>((inputs) =>
    inputs.reduce((a, b) => a + b, 0)
  ).setLabel("+");

export const createMultiplyNode = () =>
  new AdaptiveNode<number[], number>((inputs) =>
    inputs.reduce((a, b) => a * b, 1)
  ).setLabel("*");

export const createSubtractNode = () =>
  new AdaptiveNode<[number, number], number>(([a, b]) => a - b).setLabel("-");

export const createDivideNode = () =>
  new AdaptiveNode<[number, number], number>(([a, b]) =>
    b !== 0 ? a / b : 0
  ).setLabel("/");

// Float32Array Operators (original)
export const createFloat32MultiplyNode = () =>
  new AdaptiveNode<Float32Array, Float32Array>((input) => {
    const result = new Float32Array(input?.length);
    for (let i = 0; i < input?.length; i++) {
      result[i] = input[i] * 0.5;
    }
    return result;
  }).setLabel("f32*");

// Logic Operators (original)
export const createConditionalNode = () =>
  new AdaptiveNode<[boolean, any, any], any>(([condition, ifTrue, ifFalse]) =>
    condition ? ifTrue : ifFalse
  ).setLabel("?");

export const createAndNode = () =>
  new AdaptiveNode<boolean[], boolean>((inputs) =>
    inputs.every(Boolean)
  ).setLabel("&&");

export const createOrNode = () =>
  new AdaptiveNode<boolean[], boolean>((inputs) =>
    inputs.some(Boolean)
  ).setLabel("||");

export const createNotNode = () =>
  new AdaptiveNode<boolean, boolean>((input) => !input).setLabel("!");

// Data Flow Operators (enhanced)
export const createGateNode = () =>
  new AdaptiveNode<[boolean, any], any | null>(([pass, data]) =>
    pass ? data : null
  ).setLabel("gate");

export const createMergeNode = () =>
  new AdaptiveNode<any[], any[]>((inputs) =>
    inputs.filter((x) => x !== null && x !== undefined)
  ).setLabel("merge");

export const createSplitNode = (count: number = 2) => {
  const node = new AdaptiveNode<any, any[]>((input) =>
    Array(count).fill(input)
  ).setLabel("split");

  // Create multiple outlets (plus error outlet)
  node.visual.outlets = [
    ...Array(count)
      .fill(null)
      .map(() => ({
        send: () => {},
        connections: [],
      })),
    // Error outlet
    {
      send: () => {},
      connections: [],
    },
  ];

  return node;
};

// Smart Routing (original)
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
    .setLabel("router");

// Load Balancer (enhanced with health checks)
export function createLoadBalancerNode<T, U>(
  nodes: AdaptiveNode<T, U>[],
  strategy: "round-robin" | "random" | "least-loaded" = "round-robin"
): AdaptiveNode<T, U> {
  let index = 0;
  const nodeHealth = new Map(nodes.map((n) => [n.id, true]));

  const loadBalancer = new AdaptiveNode<T, U>(async (input) => {
    // Get a list of currently healthy nodes for this attempt
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
          // This is a simplified version. A real implementation would need metrics.
          nodeIndexInTryList = 0;
          selectedNode = nodesToTry[nodeIndexInTryList];
          break;
        default: // round-robin
          nodeIndexInTryList = index % nodesToTry.length;
          selectedNode = nodesToTry[nodeIndexInTryList];
          index++;
          break;
      }

      try {
        const result = await selectedNode.process(input);
        // A null result might be a valid outcome for some nodes,
        // but the original implementation treated it as an error.
        if (result === null) {
          throw new Error(`Node ${selectedNode.id} returned null`);
        }
        return result; // Success, exit the loop
      } catch (error) {
        console.warn(
          `Node ${selectedNode.id} failed in load balancer. Marking as unhealthy.`,
          error
        );
        nodeHealth.set(selectedNode.id, false);
        // Remove the failed node from our list for this attempt and retry
        nodesToTry.splice(nodeIndexInTryList, 1);
      }
    }

    // If we exit the loop, all healthy nodes failed for this input
    throw new Error("All available nodes failed to process the input.");
  });

  // Expose worker nodes for visualization
  (loadBalancer as any).workerNodes = nodes;

  return loadBalancer.setLabel(`loadBalance(${strategy})`);
}

// Parallel Processor (fixed type issue)
export function createParallelNode<T, U>(
  nodes: AdaptiveNode<T, U>[]
): AdaptiveNode<T, (U | null)[]> {
  return new AdaptiveNode<T, (U | null)[]>(async (input) => {
    return Promise.all(nodes.map((node) => node.process(input)));
  }).setLabel("parallel");
}

// Cache Node (fixed implementation)
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

    // Process with provided processor
    const result = await processor(input);

    // Add to cache
    cache.set(key, { value: result, timestamp: Date.now() });

    // Enforce size limit (LRU)
    if (cache.size > maxSize) {
      const firstKey = cache.keys().next().value;
      if (firstKey !== undefined) {
        cache.delete(firstKey);
      }
    }

    // Clean expired entries
    for (const [k, v] of cache.entries()) {
      if (Date.now() - v.timestamp > ttl) {
        cache.delete(k);
      }
    }

    return result;
  }).setLabel(`cache(${ttl}ms)`);
}

// ============================================================================
// Oscillator Node (original)
// ============================================================================

export interface OscillatorParams {
  frequency: number;
  amplitude: number;
  sampleRate: number;
  length?: number;
  waveform?: "sine" | "square" | "sawtooth" | "triangle";
}

export class OscillatorNode extends AdaptiveNode<
  OscillatorParams,
  Float32Array
> {
  private phase = 0;

  constructor() {
    super((params: OscillatorParams) => {
      const samples = new Float32Array(params?.length || 128);
      const phaseIncrement =
        (2 * Math.PI * params?.frequency) / params?.sampleRate;

      if (!params) {
        return samples;
      }

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
            samples[i] =
              (2 *
                Math.abs(
                  2 *
                    (this.phase / (2 * Math.PI) -
                      Math.floor(this.phase / (2 * Math.PI) + 0.5))
                ) -
                1) *
              params.amplitude;
            this.phase += phaseIncrement;
          }
          break;
      }

      this.phase = this.phase % (2 * Math.PI);
      return samples;
    });

    this.setLabel("oscillator");
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

export function createProcessor<TIn, TOut>(
  fn: Processor<TIn, TOut>,
  label?: string
): AdaptiveNode<TIn, TOut> {
  const node = new AdaptiveNode(fn);
  if (label) node.setLabel(label);
  return node;
}

export function chain<A, B, C>(
  first: AdaptiveNode<A, B>,
  second: AdaptiveNode<B, C>
): AdaptiveNode<A, C> {
  return new AdaptiveNode<A, C>(async (input: A) => {
    const intermediate = await first.process(input);
    if (intermediate === null) {
      throw new Error('Intermediate result was null');
    }
    const result = await second.process(intermediate);
    if (result === null) {
      throw new Error('Second node result was null');
    }
    return result;
  }).setLabel(`${first.visual.label} → ${second.visual.label}`);
}
