// ============================================================================
// Universal CUDA Graph Engine: Core Components
// ============================================================================
// This file implements the core, generic data structures for building
// and representing computation graphs for the GPU.
// ============================================================================

import crypto from "crypto";
import { CudaTensor } from "../cuda-abstractions.js";

// ============================================================================
// Compiler Configuration
// ============================================================================

export interface CompilerConfig {
  useArrayInterface?: boolean;
  enableBoundsChecking?: boolean;
  defaultBlockSize?: number;
  memoryAlignment?: number;
  enableMemoryPooling?: boolean;
  verboseMemoryPlanning?: boolean;
  enableInPlaceOptimization?: boolean;
}

// ============================================================================
// CudaNode: A Generic GPU Operation
// ============================================================================

/**
 * Represents a single, generic operation within a CudaGraph.
 */
export class CudaNode {
  public readonly id: string;
  public name: string;
  public readonly inputs = new Map<
    string,
    { shape: number[]; dtype: string }
  >();
  public readonly outputs = new Map<
    string,
    { shape: number[]; dtype: string }
  >();
  public readonly parameters = new Map<string, CudaTensor>();
  public shapeResolver:
    | ((
        inputs: Map<string, { shape: number[] }>
      ) => Map<string, { shape: number[] }>)
    | null = null;

  constructor(
    public readonly deviceCode: string,
    public readonly functionName: string
  ) {
    this.id = `cudanode_${crypto.randomUUID()}`;
    this.name = functionName;
  }

  addInput(name: string, shape: number[], dtype: string): this {
    this.inputs.set(name, { shape, dtype });
    return this;
  }

  addOutput(name: string, shape: number[], dtype: string): this {
    this.outputs.set(name, { shape, dtype });
    return this;
  }

  addParameter(name: string, tensor: CudaTensor): this {
    this.parameters.set(name, tensor);
    return this;
  }

  setShapeResolver(
    resolver: (
      inputs: Map<string, { shape: number[] }>
    ) => Map<string, { shape: number[] }>
  ): this {
    this.shapeResolver = resolver;
    return this;
  }

  resolveShapes(): void {
    if (!this.shapeResolver) {
      return;
    }
    const currentInputShapes = new Map<string, { shape: number[] }>();
    for (const [name, spec] of this.inputs.entries()) {
      currentInputShapes.set(name, { shape: spec.shape });
    }
    const resolvedOutputShapes = this.shapeResolver(currentInputShapes);
    for (const [name, resolvedSpec] of resolvedOutputShapes.entries()) {
      const output = this.outputs.get(name);
      if (output) {
        this.outputs.set(name, { ...output, shape: resolvedSpec.shape });
      }
    }
  }

  updateInputShape(portName: string, newShape: number[]): void {
    const input = this.inputs.get(portName);
    if (!input) {
      throw new Error(
        `Input port '${portName}' not found on node ${this.name}.`
      );
    }
    this.inputs.set(portName, { ...input, shape: newShape });
  }

  getKernelCall(
    outputTensorNames: Map<string, string>,
    inputTensorNames: Map<string, string>,
    parameterResolver: (nodeId: string, paramName: string) => string
  ): string {
    const outputArgs = Array.from(this.outputs.keys()).map((name) =>
      outputTensorNames.get(name)
    ).filter(arg => arg !== undefined);
    
    const inputArgs = Array.from(this.inputs.keys()).map((name) =>
      inputTensorNames.get(name)
    ).filter(arg => arg !== undefined);
    
    const paramArgs = Array.from(this.parameters.keys()).map((name) =>
      parameterResolver(this.id, name)
    ).filter(arg => arg !== undefined);

    const allArgs = [...outputArgs, ...inputArgs, ...paramArgs].join(", ");
    return allArgs;
  }
}

// ============================================================================
// CudaGraph: A Directed Acyclic Graph of GPU Operations
// ============================================================================

export interface GraphConnection {
  fromNode: CudaNode;
  fromPort: string;
  toNode: CudaNode;
  toPort: string;
}

export class CudaGraph {
  public readonly id: string;
  public name: string;
  public nodes = new Map<string, CudaNode>();
  public connections = new Set<GraphConnection>();
  private _isDirty = true;
  private _executionOrder: CudaNode[] = [];

  constructor(name: string = "UntitledCudaGraph") {
    this.id = `cudagraph_${crypto.randomUUID()}`;
    this.name = name;
  }

  addNode(node: CudaNode): this {
    this.nodes.set(node.id, node);
    this._isDirty = true;
    return this;
  }

  connect(fromNode: CudaNode, fromPort: string, toNode: CudaNode, toPort: string): this {
    if (!this.nodes.has(fromNode.id) || !this.nodes.has(toNode.id)) {
      throw new Error("Both source and target nodes must be in the graph.");
    }
    if (!fromNode.outputs.has(fromPort)) {
      throw new Error(`Source node ${fromNode.name} does not have an output port named '${fromPort}'.`);
    }
    if (!toNode.inputs.has(toPort)) {
      throw new Error(`Target node ${toNode.name} does not have an input port named '${toPort}'.`);
    }
    
    this.connections.add({ fromNode, fromPort, toNode, toPort });
    this._isDirty = true;
    return this;
  }

  getGraphInputs(): Map<string, { node: CudaNode; port: string }> {
    const graphInputs = new Map<string, { node: CudaNode; port: string }>();
    const allNodeInputs = new Set<string>();

    for (const conn of this.connections) {
      allNodeInputs.add(`${conn.toNode.id}:${conn.toPort}`);
    }

    const unconnectedInputs: Array<{ node: CudaNode; port: string; key: string }> = [];
    
    for (const node of this.nodes.values()) {
      for (const [inputPort, spec] of node.inputs) {
        const key = `${node.id}:${inputPort}`;
        const isConnected = allNodeInputs.has(key);
        if (!isConnected) {
          unconnectedInputs.push({ node, port: inputPort, key });
        }
      }
    }
    
    const embeddingInput = unconnectedInputs.find(input => input.node.name === 'embedding_forward');
    if (embeddingInput) {
      graphInputs.set(embeddingInput.port, { node: embeddingInput.node, port: embeddingInput.port });
    } else if (unconnectedInputs.length > 0) {
      const firstInput = unconnectedInputs[0];
      graphInputs.set(firstInput.port, { node: firstInput.node, port: firstInput.port });
    }
    
    return graphInputs;
  }

  getExecutionOrder(): CudaNode[] {
    if (this._isDirty) {
      this.updateExecutionOrder();
    }
    return this._executionOrder;
  }

  private updateExecutionOrder(): void {
    const sortedOrder: CudaNode[] = [];
    const inDegree = new Map<string, number>();
    const adj = new Map<string, CudaNode[]>();

    for (const node of this.nodes.values()) {
        inDegree.set(node.id, 0);
        adj.set(node.id, []);
    }

    for (const conn of this.connections) {
        adj.get(conn.fromNode.id)!.push(conn.toNode);
        inDegree.set(conn.toNode.id, (inDegree.get(conn.toNode.id) || 0) + 1);
    }

    const queue: CudaNode[] = [];
    for (const node of this.nodes.values()) {
        if (inDegree.get(node.id) === 0) {
            queue.push(node);
        }
    }

    while (queue.length > 0) {
        const u = queue.shift()!;
        sortedOrder.push(u);

        for (const v of adj.get(u.id)!) {
            inDegree.set(v.id, (inDegree.get(v.id) || 0) - 1);
            if (inDegree.get(v.id) === 0) {
                queue.push(v);
            }
        }
    }

    if (sortedOrder.length !== this.nodes.size) {
        const detectedCycleNodes = Array.from(this.nodes.values())
            .filter(node => !sortedOrder.includes(node))
            .map(node => `${node.name} (ID: ${node.id})`)
            .join(', ');
        throw new Error(`Cycle detected in the graph. Nodes involved might be: ${detectedCycleNodes}`);
    }

    this._executionOrder = sortedOrder;
    this._isDirty = false;
  }
}
