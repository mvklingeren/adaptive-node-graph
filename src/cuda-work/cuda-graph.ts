// ============================================================================
// Universal CUDA Graph Engine
// ============================================================================
// This file implements the core, generic components for building and
// executing computation graphs on the GPU. It is designed to be
// domain-agnostic and can be used for any task that can be accelerated
// with CUDA, not just machine learning.
// ============================================================================

import crypto from "crypto";
import { CudaRuntime, CudaTensor, CudaKernel } from "./cuda-abstractions";

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
      // Ensure no dynamic shapes exist in the inputs before resolving.
      if (spec.shape.includes(-1)) {
        // This can be an error or a signal that more connections are needed first.
        // For now, we'll skip if not all inputs are concrete.
        return;
      }
      currentInputShapes.set(name, { shape: spec.shape });
    }

    const resolvedOutputShapes = this.shapeResolver(currentInputShapes);

    // Update the node's output shapes with the resolved ones.
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

    // --- Shape Propagation and Validation ---
    // First, ensure the source node's shapes are fully resolved.
    fromNode.resolveShapes();

    const sourceSpec = fromNode.outputs.get(fromPort)!;
    const targetSpec = toNode.inputs.get(toPort)!;

    // If the source shape is still dynamic, we cannot proceed.
    if (sourceSpec.shape.includes(-1)) {
      throw new Error(`Cannot connect from unresolved dynamic shape on ${fromNode.name}:${fromPort}.`);
    }

    // Propagate the now-concrete shape from the source to the target.
    toNode.updateInputShape(toPort, sourceSpec.shape);

    // Now that the target's input shape is updated, try to resolve its outputs.
    toNode.resolveShapes();

    // The original validation logic is now simplified because we always
    // propagate from source to target. We can add a final check for safety.
    const finalTargetSpec = toNode.inputs.get(toPort)!;
    if (JSON.stringify(sourceSpec.shape) !== JSON.stringify(finalTargetSpec.shape)) {
      // This should ideally not be reached if updateInputShape works correctly.
      throw new Error(
        `Shape mismatch after propagation when connecting ${fromNode.name}:${fromPort} (${JSON.stringify(sourceSpec.shape)}) to ${toNode.name}:${toPort} (${JSON.stringify(finalTargetSpec.shape)}).`
      );
    }
    
    this.connections.add({ fromNode, fromPort, toNode, toPort });
    this._isDirty = true;
    return this;
  }

  getExecutionOrder(): CudaNode[] {
    if (this._isDirty) {
      this.updateExecutionOrder();
    }
    return this._executionOrder;
  }

  private updateExecutionOrder(): void {
    const order: CudaNode[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();
    const nodeDependencies = new Map<string, Set<CudaNode>>();

    for (const node of this.nodes.values()) {
        nodeDependencies.set(node.id, new Set());
    }

    for (const conn of this.connections) {
        nodeDependencies.get(conn.toNode.id)!.add(conn.fromNode);
    }

    const visit = (node: CudaNode) => {
      if (visited.has(node.id)) return;
      if (visiting.has(node.id)) {
        throw new Error(`Cycle detected in graph involving node ${node.name} (${node.id})`);
      }

      visiting.add(node.id);

      const dependencies = nodeDependencies.get(node.id) || new Set();
      for (const dep of dependencies) {
        visit(dep);
      }

      visiting.delete(node.id);
      visited.add(node.id);
      order.push(node);
    };

    for (const node of this.nodes.values()) {
      if ((nodeDependencies.get(node.id) || new Set()).size === 0) {
        visit(node);
      }
    }
    
    for (const node of this.nodes.values()) {
        if (!visited.has(node.id)) {
            visit(node);
        }
    }

    this._executionOrder = order;
    this._isDirty = false;
  }
}

// ============================================================================
// CudaGraphCompiler: Fuses a CudaGraph into a Single Kernel
// ============================================================================

export class CudaGraphCompiler {
  constructor(private runtime: CudaRuntime) {}

  async compile(graph: CudaGraph): Promise<{kernel: CudaKernel, parameters: CudaTensor[], kernelCode: string, workspaceSize: number}> {
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

    const { kernelCode, workspaceSize } = this.generateKernelCode(graph, paramNames);
    const kernel = await this.runtime.compile(kernelCode);
    
    return { kernel, parameters: orderedParams, kernelCode, workspaceSize };
  }

  generateKernelCode(graph: CudaGraph, paramNames: string[]): { kernelCode: string, workspaceSize: number } {
    const executionOrder = graph.getExecutionOrder();
    const uniqueDeviceCodes = new Set(executionOrder.map(node => node.deviceCode));
    const nodeDeviceCodes = Array.from(uniqueDeviceCodes).join("\n\n");

    const tensorRegistry = new Map<string, { varName: string, spec: { shape: number[], dtype: string } }>();
    const variableDeclarations: string[] = [];
    
    // --- 1. Identify Graph Inputs and Outputs & Populate Registry ---
    const graphInputs = new Map<string, { node: CudaNode, port: string }>();
    const graphOutputs = new Map<string, { node: CudaNode, port: string }>();
    const allNodeInputs = new Set<string>();
    const allNodeOutputs = new Set<string>();

    for (const conn of graph.connections) {
      allNodeInputs.add(`${conn.toNode.id}:${conn.toPort}`);
      allNodeOutputs.add(`${conn.fromNode.id}:${conn.fromPort}`);
    }

    for (const node of graph.nodes.values()) {
      for (const [inputPort, spec] of node.inputs) {
        if (!allNodeInputs.has(`${node.id}:${inputPort}`)) {
          graphInputs.set(inputPort, { node, port: inputPort });
          tensorRegistry.set(`${node.id}:${inputPort}`, { varName: inputPort, spec });
        }
      }
      for (const [outputPort, spec] of node.outputs) {
        if (!allNodeOutputs.has(`${node.id}:${outputPort}`)) {
          graphOutputs.set(outputPort, { node, port: outputPort });
          tensorRegistry.set(`${node.id}:${outputPort}`, { varName: outputPort, spec });
        }
      }
    }

    // --- 2. Liveness Analysis & Memory Planning ---
    const intermediateTensors = new Map<string, { spec: { shape: number[], dtype: string }, size: number, liveStart: number, liveEnd: number, offset: number }>();
    executionOrder.forEach((node, i) => {
        for (const [outputPort, spec] of node.outputs) {
            const key = `${node.id}:${outputPort}`;
            if (spec.shape.includes(-1)) {
                throw new Error(`Compiler error: Unresolved dynamic shape ${JSON.stringify(spec.shape)} for output '${outputPort}' of node '${node.name}'. Ensure all graph inputs are connected and shapes are propagated.`);
            }
            if (!tensorRegistry.has(key)) { // It's an intermediate tensor
                const size = spec.shape.reduce((a, b) => a * b, 1) * 4; // size in bytes
                intermediateTensors.set(key, { spec, size, liveStart: i, liveEnd: i, offset: -1 }); // liveEnd starts at i
            }
        }
        for (const [inputPort, spec] of node.inputs) {
            if (spec.shape.includes(-1)) {
                 throw new Error(`Compiler error: Unresolved dynamic shape ${JSON.stringify(spec.shape)} for input '${inputPort}' of node '${node.name}'. Ensure all graph inputs are connected and shapes are propagated.`);
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

    // --- 3. Build Final Kernel Code ---
    const tensorStruct = `
template<typename T>
struct Tensor {
  T* data;
  const int* shape;
  int dims;

  __device__ inline T& operator()(int i) {
    return data[i];
  }

  __device__ inline T& operator()(int i, int j) {
    return data[i * shape[1] + j];
  }

  __device__ inline T& operator()(int i, int j, int k) {
    return data[(i * shape[1] + j) * shape[2] + k];
  }
};
`;
    const tensorInstantiations: string[] = [];
    
    // Register and create instantiations for intermediate tensors
    let intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
        const varName = `intermediate_${intermediateIdx}`;
        const shapeVar = `${varName}_shape`;
        const tensorVar = `${varName}_tensor`;
        variableDeclarations.push(`  const int ${shapeVar}[] = {${tensorInfo.spec.shape.join(', ')}};`);
        tensorInstantiations.push(`  Tensor<float> ${tensorVar} = {(float*)(workspace + ${tensorInfo.offset}), ${shapeVar}, ${tensorInfo.spec.shape.length}};`);
        tensorRegistry.set(key, { varName: tensorVar, spec: tensorInfo.spec });
        intermediateIdx++;
    }

    // Create instantiations for graph I/O and parameters
    const allGraphTensors = [...graphInputs.keys(), ...graphOutputs.keys(), ...paramNames];
    for (const name of allGraphTensors) {
        tensorInstantiations.push(`  Tensor<float> ${name} = {${name}_data, ${name}_shape, ${name}_dims};`);
    }

    // --- 4. Generate Execution Flow ---
    const executionFlow: string[] = [];
    for (const node of executionOrder) {
      const inputTensors = new Map<string, string>();
      const outputTensors = new Map<string, string>();

      for (const [inputPort] of node.inputs) {
        const conn = Array.from(graph.connections).find(c => c.toNode === node && c.toPort === inputPort);
        const sourceKey = conn ? `${conn.fromNode.id}:${conn.fromPort}` : `${node.id}:${inputPort}`;
        const tensorInfo = tensorRegistry.get(sourceKey);
        if (!tensorInfo) throw new Error(`Compiler error: Could not find source tensor for ${node.name}:${inputPort}`);
        inputTensors.set(inputPort, tensorInfo.varName);
      }

      for (const [outputPort] of node.outputs) {
        const key = `${node.id}:${outputPort}`;
        const tensorInfo = tensorRegistry.get(key);
        if (!tensorInfo) throw new Error(`Compiler error: Could not find output tensor for ${node.name}:${outputPort}`);
        outputTensors.set(outputPort, tensorInfo.varName);
      }
      
      const call = node.getKernelCall(outputTensors, inputTensors);
      executionFlow.push(`  ${call}`);
    }

    // --- 5. Assemble Final Kernel ---
    const getTensorParams = (name: string) => [`float* ${name}_data`, `const int* ${name}_shape`, `int ${name}_dims`];
    const inputParams = Array.from(graphInputs.keys()).flatMap(getTensorParams);
    const outputParams = Array.from(graphOutputs.keys()).flatMap(getTensorParams);
    const parameterParams = paramNames.flatMap(getTensorParams);
    const kernelSignatureParams = [...inputParams, ...outputParams, ...parameterParams, 'char* workspace'];

    const kernelCode = `
#include <cuda_runtime.h>
#include <math.h>

${tensorStruct}

// ======================================================
// Node Device Functions
// ======================================================
${nodeDeviceCodes}

// ======================================================
// Main Fused Graph Kernel
// ======================================================
extern "C" __global__ void executeGraph(
  ${kernelSignatureParams.join(",\n  ")}
) {
  // TODO: Implement tensor-based indexing
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // --- Variable Declarations ---
${variableDeclarations.join("\n")}

  // --- Tensor Struct Instantiation ---
${tensorInstantiations.join("\n")}

  // --- Generated Execution Flow ---
  ${executionFlow.join("\n  ")}
  // --- End Execution Flow ---
}
    `;

    return { kernelCode, workspaceSize };
  }
}
