// core.ts
// Core library for the Adaptive Node with Routing System

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
   
   // ============================================================================
   // Core AdaptiveNode Implementation
   // ============================================================================
   
   export class AdaptiveNode<TInput = any, TOutput = any> {
    private processors = new Map<Function, Processor<any, TOutput>>();
    private performanceStats = new Map<string, number[]>();
    
    public readonly id: string = Math.random().toString(36).substr(2, 9);
    public readonly visual: NodeVisual = {
      x: 0,
      y: 0,
      label: '',
      inlets: [],
      outlets: []
    };
    
    constructor(private defaultProcessor: Processor<TInput, TOutput>) {
      this.setupInletsOutlets();
    }
    
    private setupInletsOutlets(): void {
      // Default: 1 inlet, 1 outlet
      this.visual.inlets = [{
        accept: (data: TInput) => this.process(data)
      }];
      
      this.visual.outlets = [{
        send: (data: TOutput) => {
          this.visual.outlets[0].connections.forEach(conn => {
            conn.transfer(data);
          });
        },
        connections: []
      }];
    }
    
    register<T extends TInput>(
      type: new (...args: any[]) => T,
      processor: Processor<T, TOutput>
    ): this {
      this.processors.set(type, processor);
      return this;
    }
    
    async process(input: TInput): Promise<TOutput> {
      const start = performance.now();
      let result: TOutput;
      let processorName = 'default';
      
      // Type-based processor selection
      let selectedProcessor = this.defaultProcessor;
      for (const [type, processor] of this.processors) {
        if (input instanceof type) {
          selectedProcessor = processor;
          processorName = type.name;
          break;
        }
      }
      
      try {
        result = await selectedProcessor(input);
        this.recordPerformance(processorName, performance.now() - start);
        
        // Send to outlets
        if (this.visual.outlets[0]) {
          this.visual.outlets[0].send(result);
        }
        
        return result;
      } catch (error) {
        console.error(`Error in node ${this.id}:`, error);
        throw error;
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
    
    setPosition(x: number, y: number): this {
      this.visual.x = x;
      this.visual.y = y;
      return this;
    }
    
    setLabel(label: string): this {
      this.visual.label = label;
      return this;
    }
   }
   
   // ============================================================================
   // Connection Implementation
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
        throw error;
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
   // Graph Implementation
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
    
    connect<A, B>(
      source: AdaptiveNode<any, A>,
      target: AdaptiveNode<A, B>,
      transformer?: Processor<A, A>
    ): Connection<A, A> {
      const connection = new Connection(source, 0, target, 0, transformer);
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
    
    private updateExecutionOrder(): void {
      // Simple topological sort
      const visited = new Set<string>();
      const order: AdaptiveNode[] = [];
      
      const visit = (node: AdaptiveNode) => {
        if (visited.has(node.id)) return;
        visited.add(node.id);
        
        // Visit dependencies first
        for (const conn of this.connections) {
          if (conn.target.id === node.id) {
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
   // Core Node Library
   // ============================================================================
   
   // Math Operators
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
   
   // Float32Array Operators
   export const createFloat32MultiplyNode = () => new AdaptiveNode<Float32Array, Float32Array>(
    (input) => {
      const result = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        result[i] = input[i] * 0.5;
      }
      return result;
    }
   ).setLabel('f32*');
   
   // Logic Operators
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
   
   // Data Flow Operators
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
    
    // Create multiple outlets
    node.visual.outlets = Array(count).fill(null).map(() => ({
      send: () => {},
      connections: []
    }));
    
    return node;
   };
   
   // Smart Routing
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
   
   // Load Balancer
   export function createLoadBalancerNode<T, U>(nodes: AdaptiveNode<T, U>[]): AdaptiveNode<T, U> {
    let index = 0;
    return new AdaptiveNode<T, U>(async (input) => {
      const node = nodes[index % nodes.length];
      index++;
      return node.process(input);
    }).setLabel('loadBalance');
   }
   
   // Parallel Processor
   export function createParallelNode<T, U>(nodes: AdaptiveNode<T, U>[]): AdaptiveNode<T, U[]> {
    return new AdaptiveNode<T, U[]>(async (input) => {
      return Promise.all(nodes.map(node => node.process(input)));
    }).setLabel('parallel');
   }
   
   // Cache Node
   export function createCacheNode<T, U>(ttl: number = 1000): AdaptiveNode<T, U> {
    const cache = new Map<string, { value: U; timestamp: number }>();
    
    return new AdaptiveNode<T, U>(async function(this: AdaptiveNode<T, U>, input: T) {
      const key = JSON.stringify(input);
      const cached = cache.get(key);
      
      if (cached && Date.now() - cached.timestamp < ttl) {
        return cached.value;
      }
      
      // Process with registered processors
      const result = await this.process(input);
      cache.set(key, { value: result, timestamp: Date.now() });
      
      // Clean old entries
      for (const [k, v] of cache.entries()) {
        if (Date.now() - v.timestamp > ttl) {
          cache.delete(k);
        }
      }
      
      return result;
    }).setLabel('cache');
   }
   
   // ============================================================================
   // Example: Oscillator Node
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
      return second.process(intermediate);
    }).setLabel(`${first.visual.label} → ${second.visual.label}`);
   }
   
   // ============================================================================
   // Export all types
   // ============================================================================
   
//    export type {
//     Connection,
//     Graph,
//     Inlet,
//     Outlet,
//     NodeVisual
//    };
