// ============================================================================
// Universal CUDA Graph Engine
// ============================================================================
// This file implements the core, generic components for building and
// executing computation graphs on the GPU. It is designed to be
// domain-agnostic and can be used for any task that can be accelerated
// with CUDA, not just machine learning.
// ============================================================================

import crypto from "crypto";
import { CudaRuntime, CudaTensor, CudaKernel } from "./cuda-abstractions.js";

// ============================================================================
// CudaNode: A Generic GPU Operation
// ============================================================================

/**
 * Represents a single, generic operation within a CudaGraph.
 * Each CudaNode holds a piece of CUDA C++ device code that will be
 * compiled into a larger kernel. It now features explicit, named inputs
 * and outputs to enable complex, non-linear graph structures.
 */
export class CudaNode {
  public readonly id: string;
  public name: string;
  // Explicitly named inputs and outputs. The string key is the port name.
  public readonly inputs = new Map<string, { shape: number[]; dtype: string }>();
  public readonly outputs = new Map<string, { shape: number[]; dtype: string }>();
  // Holds named parameter tensors (e.g., weights, biases) for this node.
  public readonly parameters = new Map<string, CudaTensor>();
  // Optional function to dynamically resolve output shapes from input shapes.
  public shapeResolver: ((inputs: Map<string, { shape: number[] }>) => Map<string, { shape: number[] }>) | null = null;

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

  setShapeResolver(resolver: (inputs: Map<string, { shape: number[] }>) => Map<string, { shape: number[] }>): this {
    this.shapeResolver = resolver;
    return this;
  }

  resolveShapes(): void {
    if (!this.shapeResolver) {
      return; // No dynamic shape logic to run
    }

    // Prepare the input for the resolver function
    const currentInputShapes = new Map<string, { shape: number[] }>();
    for (const [name, spec] of this.inputs.entries()) {
      // If an input is dynamic, we still pass it to the resolver,
      // which might be able to handle it or might need to wait.
      currentInputShapes.set(name, { shape: spec.shape });
    }

    const resolvedOutputShapes = this.shapeResolver(currentInputShapes);

    // Update the node's output shapes with the resolved ones.
    // If the resolver returns an empty map, it means it couldn't resolve shapes yet.
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
      throw new Error(`Input port '${portName}' not found on node ${this.name}.`);
    }
    // Potentially add more validation here if needed
    this.inputs.set(portName, { ...input, shape: newShape });
  }

  getKernelCall(
    outputTensorNames: Map<string, string>,
    inputTensorNames: Map<string, string>
  ): string {
    const outputArgs = Array.from(this.outputs.keys()).map(name => outputTensorNames.get(name));
    const inputArgs = Array.from(this.inputs.keys()).map(name => inputTensorNames.get(name));
    const paramArgs = Array.from(this.parameters.keys());

    const allArgs = [...outputArgs, ...inputArgs, ...paramArgs].join(", ");
    return `${this.functionName}(${allArgs});`;
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
    
    // For now, we'll prioritize the embedding layer as the main graph input
    // In a more sophisticated system, we might need better logic to determine
    // which unconnected inputs are actual graph inputs vs internal disconnected nodes
    const embeddingInput = unconnectedInputs.find(input => input.node.name === 'embedding_forward');
    if (embeddingInput) {
      graphInputs.set(embeddingInput.port, { node: embeddingInput.node, port: embeddingInput.port });
    } else if (unconnectedInputs.length > 0) {
      // Fallback: use the first unconnected input
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

// ============================================================================
// CudaGraphCompiler: Compiles a CudaGraph into a set of kernels and a
// host-side execution function.
// ============================================================================

export class CudaGraphCompiler {
  constructor(private runtime: CudaRuntime) {}

  async compile(graph: CudaGraph, inputShapes: Map<string, number[]>): Promise<{kernel: CudaKernel, parameters: CudaTensor[], kernelCode: string, workspaceSize: number}> {
    const allParams = new Map<string, CudaTensor>();
    for (const node of graph.nodes.values()) {
        for (const [name, tensor] of node.parameters) {
            if (!Array.from(allParams.values()).some(t => t.id === tensor.id)) {
                allParams.set(name, tensor);
            }
        }
    }
    
    const orderedParams = Array.from(allParams.values());
    const paramNames = Array.from(allParams.keys());

    this._setAndPropagateShapes(graph, inputShapes);

    const { kernelCode, workspaceSize } = this.generateKernelCode(graph, paramNames);
    const kernel = await this.runtime.compile(kernelCode); 
    
    return { kernel, parameters: orderedParams, kernelCode, workspaceSize };
  }

  private _setAndPropagateShapes(graph: CudaGraph, inputShapes: Map<string, number[]>): void {
    const graphInputs = graph.getGraphInputs();

    // First, find a concrete batch size if one is provided.
    let batchSize = -1;
    for (const shape of inputShapes.values()) {
        if (shape.length > 0 && shape[0] !== -1) {
            batchSize = shape[0];
            break;
        }
    }

    // If no input has a concrete batch size, default to 1 for compilation.
    if (batchSize === -1) {
        batchSize = 1;
    }

    // Update all graph inputs with the resolved batch size.
    for (const [inputName, shape] of inputShapes.entries()) {
        const finalShape = shape.map(dim => dim === -1 ? batchSize : dim);
        const inputInfo = graphInputs.get(inputName);
        if (inputInfo) {
            inputInfo.node.updateInputShape(inputInfo.port, finalShape);
        } else {
            console.warn(`Graph input port named '${inputName}' not found.`);
        }
    }

    const executionOrder = graph.getExecutionOrder();

    // Multi-pass shape resolution: some nodes might need multiple passes
    // to resolve their shapes as upstream shapes become available.
    let maxPasses = 3;
    for (let pass = 0; pass < maxPasses; pass++) {
        let anyShapeChanged = false;

        for (const node of executionOrder) {
            // Store the current output shapes to detect changes
            const oldOutputShapes = new Map();
            for (const [port, spec] of node.outputs) {
                oldOutputShapes.set(port, [...spec.shape]);
            }

            // Resolve the node's own output shapes based on its now-finalized inputs.
            node.resolveShapes();

            // Check if any output shapes changed
            for (const [port, spec] of node.outputs) {
                const oldShape = oldOutputShapes.get(port);
                if (!oldShape || JSON.stringify(oldShape) !== JSON.stringify(spec.shape)) {
                    anyShapeChanged = true;
                }
            }

            // Propagate the resolved output shapes to all downstream nodes.
            for (const conn of graph.connections) {
                if (conn.fromNode.id === node.id) {
                    const sourceSpec = node.outputs.get(conn.fromPort);
                    if (sourceSpec) {
                        conn.toNode.updateInputShape(conn.toPort, sourceSpec.shape);
                    }
                }
            }
        }

        // If no shapes changed in this pass, we're done
        if (!anyShapeChanged) {
            break;
        }
    }

    // Final validation: check that all shapes are resolved
    for (const node of executionOrder) {
        for (const [outputPort, spec] of node.outputs) {
            if (spec.shape.includes(-1)) {
                const inputShapes = JSON.stringify(Object.fromEntries(node.inputs), null, 2);
                throw new Error(
                    `Compiler error: Node '${node.name}' failed to resolve its output shape for port '${outputPort}'. Current input shapes for this node:\n${inputShapes}`
                );
            }
        }
    }
}

  generateKernelCode(graph: CudaGraph, paramNames: string[]): { kernelCode: string, workspaceSize: number } {
    const executionOrder = graph.getExecutionOrder();
    
    const { intermediateTensors, workspaceSize, tensorRegistry } = this.planMemory(graph, executionOrder);

    // Add intermediate tensors to the registry BEFORE processing nodes
    const variableDeclarations: string[] = [];
    let intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
        const varName = `intermediate_${intermediateIdx}`;
        const shapeVar = `${varName}_shape`;
        const tensorVar = `${varName}_tensor`;
        variableDeclarations.push(`  const int ${shapeVar}[] = {${tensorInfo.spec.shape.join(', ')}};`);
        tensorRegistry.set(key, { varName: tensorVar, spec: tensorInfo.spec });
        intermediateIdx++;
    }

    const uniqueKernels = new Map<string, { code: string, node: CudaNode }>();
    for (const node of executionOrder) {
        if (!uniqueKernels.has(node.functionName)) {
            uniqueKernels.set(node.functionName, { code: node.deviceCode, node });
        }
    }
    const kernelDefinitions = Array.from(uniqueKernels.values()).map(k => k.code).join('\n\n');

    const executionCalls: string[] = [];
    for (const node of executionOrder) {
        const outputTensors = new Map<string, string>();
        for (const [outputPort] of node.outputs) {
            const key = `${node.id}:${outputPort}`;
            const tensorInfo = tensorRegistry.get(key);
            if (!tensorInfo) {
                throw new Error(`Compiler Error: Could not find output tensor for ${node.name}:${outputPort} (key: ${key})`);
            }
            outputTensors.set(outputPort, tensorInfo.varName);
        }

        const inputTensors = new Map<string, string>();
        for (const [inputPort] of node.inputs) {
            const conn = Array.from(graph.connections).find(c => c.toNode === node && c.toPort === inputPort);
            const sourceKey = conn ? `${conn.fromNode.id}:${conn.fromPort}` : `${node.id}:${inputPort}`;
            const tensorInfo = tensorRegistry.get(sourceKey);
            if (!tensorInfo) {
                throw new Error(`Compiler Error: Could not find source tensor for ${node.name}:${inputPort} (key: ${sourceKey})`);
            }
            inputTensors.set(inputPort, tensorInfo.varName);
        }

        const kernelArgs = [
            ...Array.from(node.outputs.keys()).map(name => outputTensors.get(name)),
            ...Array.from(node.inputs.keys()).map(name => inputTensors.get(name)),
            ...Array.from(node.parameters.keys())
        ].join(', ');

        const gridDim = "dim3(1, 1, 1)";
        const blockDim = "dim3(256, 1, 1)";
        
        executionCalls.push(`  ${node.functionName}<<<${gridDim}, ${blockDim}>>>(${kernelArgs});`);
    }

    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);
    
    const getTensorParams = (name: string) => [`float* ${name}_data`, `const int* ${name}_shape`, `int ${name}_dims`];
    const inputParams = Array.from(graphInputs.keys()).flatMap(getTensorParams);
    const outputParams = Array.from(graphOutputs.keys()).flatMap(getTensorParams);
    const parameterParams = paramNames.flatMap(getTensorParams);
    const hostFunctionSignatureParams = [...inputParams, ...outputParams, ...parameterParams, 'char* workspace'];

    const tensorStruct = `
template<typename T>
struct Tensor {
  T* data;
  const int* shape;
  int dims;

  __device__ inline T& operator()(int i) { return data[i]; }
  __device__ inline T& operator()(int i, int j) { return data[i * shape[1] + j]; }
  __device__ inline T& operator()(int i, int j, int k) { return data[(i * shape[1] + j) * shape[2] + k]; }
};
`;
    
    const tensorInstantiations: string[] = [];
    
    // Create tensor instantiations for intermediate tensors
    intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
        const varName = `intermediate_${intermediateIdx}`;
        const shapeVar = `${varName}_shape`;
        const tensorVar = `${varName}_tensor`;
        tensorInstantiations.push(`  Tensor<float> ${tensorVar} = {(float*)(workspace + ${tensorInfo.offset}), ${shapeVar}, ${tensorInfo.spec.shape.length}};`);
        intermediateIdx++;
    }

    const allGraphTensors = [...graphInputs.keys(), ...graphOutputs.keys(), ...paramNames];
    for (const name of allGraphTensors) {
        tensorInstantiations.push(`  Tensor<float> ${name} = {${name}_data, ${name}_shape, ${name}_dims};`);
    }

    const kernelCode = `
#include <cuda_runtime.h>
#include <math.h>

${tensorStruct}

// ======================================================
// Kernel Definitions
// ======================================================
${kernelDefinitions}

// ======================================================
// Main Host-Side Execution Function
// ======================================================
extern "C" void executeGraph(
  ${hostFunctionSignatureParams.join(",\n  ")}
) {
  // --- Variable Declarations ---
${variableDeclarations.join("\n")}

  // --- Tensor Struct Instantiation ---
${tensorInstantiations.join("\n")}

  // --- Kernel Launch Sequence ---
${executionCalls.join("\n")}
  // --- End Execution Flow ---
}
    `;

    return { kernelCode, workspaceSize };
  }

  private getGraphOutputs(graph: CudaGraph): Map<string, { node: CudaNode; port: string }> {
      const graphOutputs = new Map<string, { node: CudaNode; port: string }>();
      const allNodeOutputs = new Set<string>();
      for (const conn of graph.connections) {
          allNodeOutputs.add(`${conn.fromNode.id}:${conn.fromPort}`);
      }
      for (const node of graph.nodes.values()) {
          for (const [outputPort, spec] of node.outputs) {
              if (!allNodeOutputs.has(`${node.id}:${outputPort}`)) {
                  graphOutputs.set(outputPort, { node, port: outputPort });
              }
          }
      }
      return graphOutputs;
  }

  private planMemory(graph: CudaGraph, executionOrder: CudaNode[]): {
      intermediateTensors: Map<string, { spec: { shape: number[], dtype: string }, size: number, liveStart: number, liveEnd: number, offset: number }>,
      workspaceSize: number,
      tensorRegistry: Map<string, { varName: string, spec: { shape: number[], dtype: string } }>
  } {
    const tensorRegistry = new Map<string, { varName: string, spec: { shape: number[], dtype: string } }>();
    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);

    for (const [inputPort, { node, port }] of graphInputs.entries()) {
      tensorRegistry.set(`${node.id}:${port}`, { varName: inputPort, spec: node.inputs.get(port)! });
    }
    for (const [outputPort, { node, port }] of graphOutputs.entries()) {
      tensorRegistry.set(`${node.id}:${outputPort}`, { varName: outputPort, spec: node.outputs.get(port)! });
    }

    const intermediateTensors = new Map<string, { spec: { shape: number[], dtype: string }, size: number, liveStart: number, liveEnd: number, offset: number }>();
    executionOrder.forEach((node, i) => {
        for (const [outputPort, spec] of node.outputs) {
            const key = `${node.id}:${outputPort}`;
            if (spec.shape.includes(-1)) {
                const inputShapes = JSON.stringify(Array.from(node.inputs.entries()));
                throw new Error(`Compiler error: Unresolved dynamic shape ${JSON.stringify(spec.shape)} for output '${outputPort}' of node '${node.name}'. Node inputs: ${inputShapes}`);
            }
            if (!tensorRegistry.has(key)) {
                const size = spec.shape.reduce((a, b) => a * b, 1) * 4; // size in bytes
                intermediateTensors.set(key, { spec, size, liveStart: i, liveEnd: i, offset: -1 });
            }
        }
        for (const [inputPort, spec] of node.inputs) {
            if (spec.shape.includes(-1)) {
                 throw new Error(`Compiler error: Unresolved dynamic shape ${JSON.stringify(spec.shape)} for input '${inputPort}' of node '${node.name}'.`);
            }
            const conn = Array.from(graph.connections).find(c => c.toNode === node && c.toPort === inputPort);
            if (conn) {
                const sourceKey = `${conn.fromNode.id}:${conn.fromPort}`;
                if (intermediateTensors.has(sourceKey)) {
                    intermediateTensors.get(sourceKey)!.liveEnd = i;
                }
            }
        }
    });

    let workspaceSize = 0;
    const blocks: {start: number, end: number, size: number}[] = [];
    const sortedTensors = Array.from(intermediateTensors.entries()).sort((a, b) => a[1].liveStart - b[1].liveStart);

    for (const [key, tensor] of sortedTensors) {
        let placed = false;
        for (const block of blocks) {
            if (tensor.liveStart >= block.end && tensor.size <= block.size) {
                tensor.offset = block.start;
                block.end = tensor.liveEnd;
                placed = true;
                break;
            }
        }
        if (!placed) {
            tensor.offset = workspaceSize;
            workspaceSize += tensor.size;
            blocks.push({start: tensor.offset, end: tensor.liveEnd, size: tensor.size});
        }
    }
    return { intermediateTensors, workspaceSize, tensorRegistry };
  }
}
