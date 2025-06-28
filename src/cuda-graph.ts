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
export class CudaNode {
  public readonly id: string;
  public name: string;
  public readonly dependencies = new Set<CudaNode>();
  public readonly dependents = new Set<CudaNode>();
  // Holds named parameter tensors (e.g., weights, biases) for this node.
  public readonly parameters = new Map<string, CudaTensor>();

  /**
   * @param name - A descriptive name for the node.
   * @param deviceCode - The CUDA C++ device function code for this operation.
   *                     This code should be a self-contained function.
   * @param functionName - The name of the device function defined in the code.
   * @param outputVarName - The variable name used for the output of this node's operation.
   * @param inputVarNames - The variable names for inputs from other nodes.
   */
  constructor(
    public readonly deviceCode: string,
    public readonly functionName: string,
    public readonly outputVarName: string,
    public readonly inputVarNames: string[]
  ) {
    this.id = `cudanode_${crypto.randomUUID()}`;
    this.name = functionName;
  }

  addParameter(name: string, tensor: CudaTensor): this {
    this.parameters.set(name, tensor);
    return this;
  }

  /**
   * Generates the C++ code to call this node's device function.
   * Assumes the device function's signature lists dynamic inputs first,
   * then parameter inputs in the order they were added.
   * @param inputVarMapping - A map from the node's input variable names to the
   *                          actual variable names in the generated kernel.
   * @returns The C++ code string for the function call.
   */
  getKernelCall(inputVarMapping: Map<string, string>): string {
    const inputArgs = this.inputVarNames.map(name => inputVarMapping.get(name));
    const paramArgs = Array.from(this.parameters.keys());
    const allArgs = [...inputArgs, ...paramArgs].join(", ");
    return `float ${this.outputVarName} = ${this.functionName}(${allArgs});`;
  }
}

// ============================================================================
// CudaGraph: A Directed Acyclic Graph of GPU Operations
// ============================================================================

export class CudaGraph {
  public readonly id: string;
  public name: string;
  public nodes = new Map<string, CudaNode>();
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

  connect(source: CudaNode, target: CudaNode): this {
    if (!this.nodes.has(source.id) || !this.nodes.has(target.id)) {
      throw new Error("Both source and target nodes must be in the graph.");
    }
    source.dependents.add(target);
    target.dependencies.add(source);
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

    const visit = (node: CudaNode) => {
      if (visited.has(node.id)) return;
      if (visiting.has(node.id)) {
        throw new Error(`Cycle detected in graph involving node ${node.name} (${node.id})`);
      }

      visiting.add(node.id);

      for (const dep of node.dependencies) {
        visit(dep);
      }

      visiting.delete(node.id);
      visited.add(node.id);
      order.push(node);
    };

    for (const node of this.nodes.values()) {
      if (node.dependencies.size === 0) {
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
  async compile(graph: CudaGraph, entryVarName: string = 'value'): Promise<{kernel: CudaKernel, parameters: CudaTensor[], kernelCode: string}> {
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

    const kernelCode = this.generateKernelCode(graph, entryVarName, paramNames);
    const kernel = await this.runtime.compile(kernelCode);
    
    return { kernel, parameters: orderedParams, kernelCode };
  }

  /**
   * Generates the complete CUDA C++ source code for the graph's kernel.
   * @param graph - The CudaGraph to generate code for.
   * @param entryVarName - The name of the initial input variable.
   * @param paramNames - A list of names for the kernel's parameter tensors.
   * @returns The full CUDA C++ source code as a string.
   */
  generateKernelCode(graph: CudaGraph, entryVarName: string, paramNames: string[]): string {
    const executionOrder = graph.getExecutionOrder();
    // Use a Set to automatically handle deduplication of device code.
    const uniqueDeviceCodes = new Set(executionOrder.map(node => node.deviceCode));
    const nodeDeviceCodes = Array.from(uniqueDeviceCodes).join("\n\n");
    
    const executionFlow: string[] = [];

    for (const node of executionOrder) {
        const inputVarMapping = new Map<string, string>();
        
        if (node.dependencies.size === 0) {
            // Entry node: its first input is the main kernel input variable.
            if (node.inputVarNames.length > 0) {
                inputVarMapping.set(node.inputVarNames[0], entryVarName);
            }
        } else {
            // Node with dependencies: map dependency outputs to this node's inputs by order.
            // This is a simplification. A better graph API would map connections explicitly.
            const sortedDeps = Array.from(node.dependencies).sort((a, b) => a.id.localeCompare(b.id));

            for (let i = 0; i < node.inputVarNames.length; i++) {
                const inputName = node.inputVarNames[i];
                const dep = sortedDeps[i];
                if (dep) {
                    // Map the i-th input name to the i-th dependency's output variable name.
                    inputVarMapping.set(inputName, dep.outputVarName);
                } else {
                    // This case should ideally throw an error, but for now we log.
                    console.error(`[CudaGraphCompiler] Error: Not enough dependencies for node ${node.name} to satisfy input '${inputName}'`);
                }
            }
        }

        executionFlow.push(node.getKernelCall(inputVarMapping));
    }

    const lastNode = executionOrder[executionOrder.length - 1];
    const finalOutputVar = lastNode ? lastNode.outputVarName : entryVarName;
    
    const paramDeclarations = paramNames.map(name => `const float* ${name}`).join(", ");

    const kernelSignatureParams = [
      "const float* inputs",
      "float* outputs",
      "int n",
      ...paramNames.map(name => `const float* ${name}`)
    ];

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

  float ${entryVarName} = inputs[idx];

  // --- Generated Execution Flow ---
  ${executionFlow.join("\n  ")}
  // --- End Execution Flow ---

  outputs[idx] = ${finalOutputVar};
}
    `;
  }
}
