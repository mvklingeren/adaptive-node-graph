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
 * compiled into a larger kernel.
 */
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

  /**
   * @param name - A descriptive name for the node.
   * @param deviceCode - The CUDA C++ device function code for this operation.
   * @param functionName - The name of the device function defined in the code.
   */
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

  /**
   * Generates the C++ code to call this node's device function.
   * @param outputVarNames - A map from this node's output port names to the
   *                         actual C++ variable names they will be assigned to.
   * @param inputVarNames - A map from this node's input port names to the
   *                        C++ variable names of their data source.
   * @returns The C++ code string for the function call.
   */
  getKernelCall(
    outputVarNames: Map<string, string>,
    inputVarNames: Map<string, string>
  ): string {
    const outputArgs = Array.from(this.outputs.keys()).map(name => outputVarNames.get(name));
    const inputArgs = Array.from(this.inputs.keys()).map(name => inputVarNames.get(name));
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
    
    // Ensure all nodes were visited (handles disconnected graphs)
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

  /**
   * Compiles an entire CudaGraph into a single, executable CUDA kernel.
   * @param graph - The CudaGraph to compile.
   * @param entryVarName - The name of the initial input variable for the kernel.
   * @returns A promise that resolves to an object containing the compiled kernel,
   *          the list of parameter tensors it expects, and the generated kernel source code.
   */
  async compile(graph: CudaGraph): Promise<{kernel: CudaKernel, parameters: CudaTensor[], kernelCode: string}> {
    // Collect all unique parameters from the graph's nodes.
    const allParams = new Map<string, CudaTensor>();
    for (const node of graph.nodes.values()) {
        for (const [name, tensor] of node.parameters) {
            // Ensure each tensor is included only once, using its ID for uniqueness.
            if (!Array.from(allParams.values()).some(t => t.id === tensor.id)) {
                allParams.set(name, tensor);
            }
        }
    }
    
    const orderedParams = Array.from(allParams.values());
    const paramNames = Array.from(allParams.keys());

    const kernelCode = this.generateKernelCode(graph, paramNames);
    const kernel = await this.runtime.compile(kernelCode);
    
    return { kernel, parameters: orderedParams, kernelCode };
  }

  /**
   * Generates the complete CUDA C++ source code for the graph's kernel.
   * This version uses the explicit graph connections to wire up variables.
   * @param graph - The CudaGraph to generate code for.
   * @param paramNames - A list of names for the kernel's parameter tensors.
   * @returns The full CUDA C++ source code as a string.
   */
  generateKernelCode(graph: CudaGraph, paramNames: string[]): string {
    const executionOrder = graph.getExecutionOrder();
    const uniqueDeviceCodes = new Set(executionOrder.map(node => node.deviceCode));
    const nodeDeviceCodes = Array.from(uniqueDeviceCodes).join("\n\n");

    const varNameMap = new Map<string, string>(); // Maps `nodeId:portName` to a C++ variable name
    const executionFlow: string[] = [];
    const variableDeclarations: string[] = [];

    // Identify graph inputs and outputs
    const graphInputs = new Map<string, {node: CudaNode, port: string}>();
    const graphOutputs = new Map<string, {node: CudaNode, port: string}>();
    const allNodeInputs = new Set<string>(); // "nodeId:port"
    const allNodeOutputs = new Set<string>(); // "nodeId:port"

    for (const conn of graph.connections) {
        allNodeInputs.add(`${conn.toNode.id}:${conn.toPort}`);
        allNodeOutputs.add(`${conn.fromNode.id}:${conn.fromPort}`);
    }

    for (const node of graph.nodes.values()) {
        for (const inputPort of node.inputs.keys()) {
            if (!allNodeInputs.has(`${node.id}:${inputPort}`)) {
                graphInputs.set(inputPort, { node, port: inputPort });
            }
        }
        for (const outputPort of node.outputs.keys()) {
            if (!allNodeOutputs.has(`${node.id}:${outputPort}`)) {
                graphOutputs.set(outputPort, { node, port: outputPort });
            }
        }
    }

    // Generate kernel signature
    const inputParams = Array.from(graphInputs.keys()).map(name => `const float* ${name}`);
    const outputParams = Array.from(graphOutputs.keys()).map(name => `float* ${name}`);
    const parameterParams = paramNames.map(name => `const float* ${name}`);
    const kernelSignatureParams = [...inputParams, ...outputParams, ...parameterParams, "int n"];

    // Main loop for code generation
    for (const node of executionOrder) {
        const inputVarNames = new Map<string, string>();
        const outputVarNames = new Map<string, string>();

        // Map inputs of the current node
        for (const inputPort of node.inputs.keys()) {
            const connection = Array.from(graph.connections).find(c => c.toNode === node && c.toPort === inputPort);
            if (connection) {
                const sourceVar = varNameMap.get(`${connection.fromNode.id}:${connection.fromPort}`);
                if (!sourceVar) throw new Error("Topological sort failed or logic error.");
                inputVarNames.set(inputPort, sourceVar);
            } else {
                // It's a graph-level input
                inputVarNames.set(inputPort, `${inputPort}[idx]`);
            }
        }

        // Create and map outputs of the current node
        for (const [outputPort, spec] of node.outputs) {
            const varName = `${node.name}_${outputPort}_${node.id.substring(0, 4)}`;
            // For now, we assume intermediate values are scalar floats.
            // This is where a memory planner would allocate from a workspace.
            variableDeclarations.push(`float ${varName};`);
            outputVarNames.set(outputPort, `&${varName}`); // Pass address to device func
            varNameMap.set(`${node.id}:${outputPort}`, varName);
        }

        // The generated device functions are now expected to take pointers for outputs
        const call = node.getKernelCall(outputVarNames, inputVarNames);
        executionFlow.push(`  // Node: ${node.name}`);
        executionFlow.push(`  ${call}`);
    }

    // Assign final results to graph outputs
    for (const [name, {node, port}] of graphOutputs) {
        const finalVar = varNameMap.get(`${node.id}:${port}`);
        if (finalVar) {
            executionFlow.push(`  ${name}[idx] = ${finalVar};`);
        }
    }

    return `
#include <cuda_runtime.h>
#include <math.h>

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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // --- Variable Declarations ---
  ${variableDeclarations.join("\n  ")}

  // --- Generated Execution Flow ---
${executionFlow.join("\n")}
  // --- End Execution Flow ---
}
    `;
  }
}
