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
  send(data: T): void;
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
  private lastResult: TOutput | null = null;
  
  public readonly id: string = Math.random().toString(36).substr(2, 9);
  public readonly visual: NodeVisual = {
    x: 0,
    y: 0,
    label: '',
    inlets: [],
    outlets: []
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
    this.visual.inlets = [{
      accept: (data: TInput) => this.process(data)
    }];
    
    this.visual.outlets = [
      // Data outlet
      {
        send: (data: TOutput) => {
          this.visual.outlets[0].connections.forEach(conn => {
            conn.transfer(data);
          });
        },
        connections: []
      },
      // Error outlet
      {
        send: (error: NodeError) => {
          this.visual.outlets[1].connections.forEach(conn => {
            conn.transfer(error);
          });
        },
        connections: []
      }
    ];
  }
  
  register<T extends TInput>(
    type: new (...args: any[]) => T,
    processor: Processor<T, TOutput>
  ): this {
    this.processors.set(type, processor);
    return this;
  }
  
  async process(input: TInput): Promise<TOutput | null> {
    // Circuit breaker check
    if (this.isCircuitOpen) {
      const now = Date.now();
      if (now - this.lastErrorTime > this.circuitBreakerResetTime) {
        this.isCircuitOpen = false;
        this.errorCount = 0;
      } else {
        this.sendError(new Error('Circuit breaker is open'), input);
        return null;
      }
    }
    
    // Flow control - wait if at capacity
    if (this.processing >= this.maxConcurrent) {
      await new Promise<void>(resolve => this.queue.push(resolve));
    }
    
    this.processing++;
    const start = performance.now();
    let processorName = 'default';
    
    try {
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
      
      // Send to data outlet
      if (this.visual.outlets[0]) {
        this.visual.outlets[0].send(result);
      }
      
      // Reset error count on success
      this.errorCount = 0;
      
      return result;
    } catch (error) {
      this.handleError(error as Error, input);
      return null;
    } finally {
      this.processing--;
      const nextInQueue = this.queue.shift();
      if (nextInQueue) nextInQueue();
    }
  }
  
  private handleError(error: Error, input: TInput): void {
    this.errorCount++;
    this.lastErrorTime = Date.now();
    
    // Check circuit breaker
    if (this.errorCount >= this.circuitBreakerThreshold) {
      this.isCircuitOpen = true;
      console.error(`Circuit breaker opened for node ${this.id} after ${this.errorCount} errors`);
    }
    
    this.sendError(error, input);
  }
  
  private sendError(error: Error, input: TInput): void {
    const nodeError: NodeError = {
      error,
      input,
      nodeId: this.id,
      timestamp: Date.now()
    };
    
    // Send to error outlet if connected
    if (this.visual.outlets[1] && this.visual.outlets[1].connections.length > 0) {
      this.visual.outlets[1].send(nodeError);
    } else {
      // Log if no error handler connected
      console.error(`Unhandled error in node ${this.id}:`, error);
    }
  }
  
  private recordPerformance(processorName: string, duration: number): void {
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
  
  getPerformanceStats(): Map<string, { avg: number; min: number; max: number }> {
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
  
  async transfer(data: TData): Promise<void> {
    try {
      const transformed = this.transformer
        ? await this.transformer(data)
        : data as unknown as TTransformed;
      
      await this.target.process(transformed);
    } catch (error) {
      console.error(`Error in connection ${this.source.id} -> ${this.target.id}:`, error);
      // Send error to target's error handling
      if (this.target.visual.outlets[1]) {
        this.target.visual.outlets[1].send({
          error: error as Error,
          input: data,
          nodeId: `connection-${this.source.id}-${this.target.id}`,
          timestamp: Date.now()
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
  private nodes = new Map<string, AdaptiveNode>();
  private connections = new Set<Connection<any, any>>();
  private executionOrder: AdaptiveNode[] = [];
  
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
    const connection = new Connection(source, sourceOutlet, target, targetInlet, transformer);
    this.connections.add(connection);
    this.updateExecutionOrder();
    return connection;
  }
  
  // Connect error outlets
  connectError(
    source: AdaptiveNode,
    errorHandler: AdaptiveNode<NodeError, any>
  ): Connection<NodeError, NodeError> {
    return this.connect(source, errorHandler, undefined, 1, 0);
  }
  
  disconnect(connection: Connection<any, any>): this {
    connection.disconnect();
    this.connections.delete(connection);
    this.updateExecutionOrder();
    return this;
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
  
  async execute<T>(input: T, startNodeId?: string): Promise<any> {
    if (startNodeId) {
      const startNode = this.nodes.get(startNodeId);
      if (!startNode) throw new Error(`Node ${startNodeId} not found`);
      return startNode.process(input);
    }
    
    // Execute all nodes in order
    let result: any = input;
    for (const node of this.executionOrder) {
      result = await node.process(result);
    }
    return result;
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
        node => !executed.has(node.id) && canExecute(node.id)
      );
      
      if (readyNodes.length === 0) break;
      
      const promises = readyNodes.map(async node => {
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
      if (conn.sourceOutlet === 0) { // Only data connections, not error connections
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
      nodes: Array.from(this.nodes.values()).map(node => ({
        id: node.id,
        label: node.visual.label,
        x: node.visual.x,
        y: node.visual.y
      })),
      connections: Array.from(this.connections).map(conn => ({
        source: conn.source.id,
        sourceOutlet: conn.sourceOutlet,
        target: conn.target.id,
        targetInlet: conn.targetInlet
      }))
    };
  }
}

// ============================================================================
// Time-based Operators
// ============================================================================

export const createDelayNode = (ms: number) => 
  new AdaptiveNode(async (input: any) => {
    await new Promise(resolve => setTimeout(resolve, ms));
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
    console.error(`[${new Date(error.timestamp).toISOString()}] Error in node ${error.nodeId}:`, 
      error.error.message,
      '\nInput:', error.input
    );
    return error;
  }).setLabel('errorLogger');

export const createErrorRecoveryNode = <T>(defaultValue: T) =>
  new AdaptiveNode<NodeError, T>((error) => {
    console.warn(`Recovering from error with default value:`, defaultValue);
    return defaultValue;
  }).setLabel('errorRecovery');

export const createRetryNode = <T>(maxRetries: number = 3, delayMs: number = 1000) => {
  const retryMap = new Map<string, number>();
  
  return new AdaptiveNode<NodeError, T | null>(async (error) => {
    const key = `${error.nodeId}-${JSON.stringify(error.input)}`;
    const retries = retryMap.get(key) || 0;
    
    if (retries < maxRetries) {
      retryMap.set(key, retries + 1);
      await new Promise(resolve => setTimeout(resolve, delayMs * (retries + 1)));
      
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
    super(processor || ((input) => {
      this.receivedInputs.push(input);
      return input;
    }));
    
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
    
    this.setLabel('test');
  }
  
  reset(): void {
    this.receivedInputs = [];
    this.processedOutputs = [];
    this.errors = [];
  }
  
  assertReceived(expected: T[]): void {
    if (JSON.stringify(this.receivedInputs) !== JSON.stringify(expected)) {
      throw new Error(`Expected inputs ${JSON.stringify(expected)}, got ${JSON.stringify(this.receivedInputs)}`);
    }
  }
  
  assertProcessed(expected: T[]): void {
    if (JSON.stringify(this.processedOutputs) !== JSON.stringify(expected)) {
      throw new Error(`Expected outputs ${JSON.stringify(expected)}, got ${JSON.stringify(this.processedOutputs)}`);
    }
  }
  
  assertNoErrors(): void {
    if (this.errors.length > 0) {
      throw new Error(`Expected no errors, got ${this.errors.length}: ${this.errors.map(e => e.error.message).join(', ')}`);
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
    throw new Error(`Expected output ${JSON.stringify(expectedOutput)}, got ${JSON.stringify(output)}`);
  }
}

// ============================================================================
// Sub-graph Support
// ============================================================================

export class SubGraphNode<TInput = any, TOutput = any> extends AdaptiveNode<TInput, TOutput> {
  constructor(
    private subGraph: Graph,
    private inputNodeId?: string,
    private outputNodeId?: string
  ) {
    super(async (input: TInput) => {
      // Execute the sub-graph
      const result = await this.subGraph.execute(input, this.inputNodeId);
      
      // If output node specified, get its result
      if (this.outputNodeId) {
        const outputNode = this.subGraph.getNode(this.outputNodeId);
        if (outputNode) {
          return outputNode.getLastResult() || result;
        }
      }
      
      return result;
    });
    
    this.setLabel('subgraph');
  }
  
  getSubGraph(): Graph {
    return this.subGraph;
  }
}

// ============================================================================
// Enhanced Core Node Library (keeping originals, adding new ones)
// ============================================================================

// Math Operators (original)
export const createAddNode = () => new AdaptiveNode<number[], number>(
  (inputs) => inputs.reduce((a, b) => a + b, 0)
).setLabel('+');

export const createMultiplyNode = () => new AdaptiveNode<number[], number>(
  (inputs) => inputs.reduce((a, b) => a * b, 1)
).setLabel('*');

export const createSubtractNode = () => new AdaptiveNode<[number, number], number>(
  ([a, b]) => a - b
).setLabel('-');

export const createDivideNode = () => new AdaptiveNode<[number, number], number>(
  ([a, b]) => b !== 0 ? a / b : 0
).setLabel('/');

// Float32Array Operators (original)
export const createFloat32MultiplyNode = () => new AdaptiveNode<Float32Array, Float32Array>(
  (input) => {
    const result = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      result[i] = input[i] * 0.5;
    }
    return result;
  }
).setLabel('f32*');

// Logic Operators (original)
export const createConditionalNode = () => new AdaptiveNode<[boolean, any, any], any>(
  ([condition, ifTrue, ifFalse]) => condition ? ifTrue : ifFalse
).setLabel('?');

export const createAndNode = () => new AdaptiveNode<boolean[], boolean>(
  (inputs) => inputs.every(Boolean)
).setLabel('&&');

export const createOrNode = () => new AdaptiveNode<boolean[], boolean>(
  (inputs) => inputs.some(Boolean)
).setLabel('||');

export const createNotNode = () => new AdaptiveNode<boolean, boolean>(
  (input) => !input
).setLabel('!');

// Data Flow Operators (enhanced)
export const createGateNode = () => new AdaptiveNode<[boolean, any], any | null>(
  ([pass, data]) => pass ? data : null
).setLabel('gate');

export const createMergeNode = () => new AdaptiveNode<any[], any[]>(
  (inputs) => inputs.filter(x => x !== null && x !== undefined)
).setLabel('merge');

export const createSplitNode = (count: number = 2) => {
  const node = new AdaptiveNode<any, any[]>(
    (input) => Array(count).fill(input)
  ).setLabel('split');
  
  // Create multiple outlets (plus error outlet)
  node.visual.outlets = [
    ...Array(count).fill(null).map(() => ({
      send: () => {},
      connections: []
    })),
    // Error outlet
    {
      send: () => {},
      connections: []
    }
  ];
  
  return node;
};

// Smart Routing (original)
export const createRouterNode = () => new AdaptiveNode<any, { route: string; data: any }>(
  (input) => ({ route: 'default', data: input })
)
  .register(AudioBuffer, (audio) => ({ route: 'audio', data: audio }))
  .register(ArrayBuffer, (buffer) => ({ route: 'binary', data: buffer }))
  .register(Array, (array) => ({ route: 'array', data: array }))
  .register(Object, (obj) => {
    if ('sampleRate' in obj) return { route: 'audio', data: obj };
    if ('buffer' in obj) return { route: 'binary', data: obj };
    return { route: 'object', data: obj };
  })
  .setLabel('router');

// Load Balancer (enhanced with health checks)
export function createLoadBalancerNode<T, U>(
  nodes: AdaptiveNode<T, U>[],
  strategy: 'round-robin' | 'random' | 'least-loaded' = 'round-robin'
): AdaptiveNode<T, U> {
  let index = 0;
  const nodeHealth = new Map(nodes.map(n => [n.id, true]));
  
  const loadBalancer = new AdaptiveNode<T, U>(async (input) => {
    const healthyNodes = nodes.filter(n => nodeHealth.get(n.id));
    if (healthyNodes.length === 0) {
      throw new Error('No healthy nodes available');
    }
    
    let selectedNode: AdaptiveNode<T, U>;
    
    switch (strategy) {
      case 'random':
        selectedNode = healthyNodes[Math.floor(Math.random() * healthyNodes.length)];
        break;
      case 'least-loaded':
        // This would need access to node processing count
        selectedNode = healthyNodes[0]; // Simplified
        break;
      default: // round-robin
        selectedNode = healthyNodes[index % healthyNodes.length];
        index++;
    }
    
    try {
      const result = await selectedNode.process(input);
      if (result === null) {
        throw new Error('Node returned null');
      }
      return result;
    } catch (error) {
      nodeHealth.set(selectedNode.id, false);
      // Retry with another node
      if (healthyNodes.length > 1) {
        // Recursive retry - remove the failed node from consideration
        const retryNodes = nodes.filter(n => n.id !== selectedNode.id && nodeHealth.get(n.id));
        if (retryNodes.length > 0) {
          const retryBalancer = createLoadBalancerNode(retryNodes, strategy);
          const retryResult = await retryBalancer.process(input);
          if (retryResult === null) {
            throw new Error('Retry returned null');
          }
          return retryResult;
        }
      }
      throw error;
    }
  });
  
  return loadBalancer.setLabel(`loadBalance(${strategy})`);
}

// Parallel Processor (fixed type issue)
export function createParallelNode<T, U>(nodes: AdaptiveNode<T, U>[]): AdaptiveNode<T, (U | null)[]> {
  return new AdaptiveNode<T, (U | null)[]>(async (input) => {
    return Promise.all(nodes.map(node => node.process(input)));
  }).setLabel('parallel');
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
  waveform?: 'sine' | 'square' | 'sawtooth' | 'triangle';
}

export class OscillatorNode extends AdaptiveNode<OscillatorParams, Float32Array> {
  private phase = 0;
  
  constructor() {
    super((params: OscillatorParams) => {
      const samples = new Float32Array(params.length || 128);
      const phaseIncrement = (2 * Math.PI * params.frequency) / params.sampleRate;
      
      switch (params.waveform || 'sine') {
        case 'sine':
          for (let i = 0; i < samples.length; i++) {
            samples[i] = Math.sin(this.phase) * params.amplitude;
            this.phase += phaseIncrement;
          }
          break;
          
        case 'square':
          for (let i = 0; i < samples.length; i++) {
            samples[i] = (Math.sin(this.phase) > 0 ? 1 : -1) * params.amplitude;
            this.phase += phaseIncrement;
          }
          break;
          
        case 'sawtooth':
          for (let i = 0; i < samples.length; i++) {
            samples[i] = ((this.phase / Math.PI) - 1) * params.amplitude;
            this.phase += phaseIncrement;
            if (this.phase >= 2 * Math.PI) this.phase -= 2 * Math.PI;
          }
          break;
          
        case 'triangle':
          for (let i = 0; i < samples.length; i++) {
            samples[i] = (2 * Math.abs(2 * (this.phase / (2 * Math.PI) - Math.floor(this.phase / (2 * Math.PI) + 0.5))) - 1) * params.amplitude;
            this.phase += phaseIncrement;
          }
          break;
      }
      
      this.phase = this.phase % (2 * Math.PI);
      return samples;
    });
    
    this.setLabel('oscillator');
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
