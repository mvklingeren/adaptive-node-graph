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
  constructor(
    private runtime: CudaRuntime,
    private defaultBlockSize: number = 256
  ) {}

  async compile(graph: CudaGraph, inputShapes: Map<string, number[]>): Promise<{kernel: CudaKernel, parameters: CudaTensor[], kernelCode: string, workspaceSize: number}> {
    // Create unique parameter names for each tensor to avoid conflicts
    const allParams = new Map<string, CudaTensor>();
    const paramNameMapping = new Map<string, string>(); // Maps node parameter names to unique global names
    let paramCounter = 0;
    
    for (const node of graph.nodes.values()) {
        if (!node || !node.id) {
            console.warn('Skipping invalid node in parameter mapping');
            continue;
        }
        
        for (const [localParamName, tensor] of node.parameters) {
            if (!tensor || !tensor.id) {
                console.warn(`Skipping invalid tensor '${localParamName}' in node '${node.name}'`);
                continue;
            }
            
            // Check if we've already seen this tensor (by ID)
            let globalParamName = null;
            for (const [existingName, existingTensor] of allParams.entries()) {
                if (existingTensor && existingTensor.id === tensor.id) {
                    globalParamName = existingName;
                    break;
                }
            }
            
            // If this is a new tensor, create a unique global name
            if (!globalParamName) {
                globalParamName = `param_${paramCounter++}_${localParamName}`;
                allParams.set(globalParamName, tensor);
            }
            
            // Map the node's local parameter name to the global name
            const nodeParamKey = `${node.id}:${localParamName}`;
            paramNameMapping.set(nodeParamKey, globalParamName);
        }
    }
    
    const orderedParams = Array.from(allParams.values());
    const paramNames = Array.from(allParams.keys());

    this._setAndPropagateShapes(graph, inputShapes);

    const { kernelCode, workspaceSize, filename } = this.generateKernelCode(graph, paramNames, paramNameMapping);
    const kernel = await this.runtime.compile(kernelCode, filename); 
    
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

  generateKernelCode(graph: CudaGraph, paramNames: string[], paramNameMapping: Map<string, string>): { kernelCode: string, workspaceSize: number, filename: string } {
    const executionOrder = graph.getExecutionOrder();
    const filename = `generated-${graph.name}-kernel.cu`;
    
    const { intermediateTensors, workspaceSize, tensorRegistry, memoryPools } = this.planMemory(graph, executionOrder);

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
        // Skip nodes with empty device code (like input nodes)
        if (node.deviceCode.trim() && !uniqueKernels.has(node.functionName)) {
            uniqueKernels.set(node.functionName, { code: node.deviceCode, node });
        }
    }
    const kernelDefinitions = Array.from(uniqueKernels.values()).map(k => k.code).join('\n\n');

    const executionCalls: string[] = [];
    for (const node of executionOrder) {
        // Skip nodes with empty device code (like input nodes)
        if (!node.deviceCode.trim()) {
            continue;
        }
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

        // Map node parameter names to unique global parameter names
        const paramArgs = Array.from(node.parameters.keys()).map(localParamName => {
            const nodeParamKey = `${node.id}:${localParamName}`;
            const globalParamName = paramNameMapping.get(nodeParamKey);
            if (!globalParamName) {
                throw new Error(`Compiler Error: Could not find global parameter name for ${node.name}:${localParamName}`);
            }
            return globalParamName;
        });

        // Standard kernel argument ordering: outputs, inputs, parameters
        const kernelArgs = [
            ...Array.from(node.outputs.keys()).map(name => outputTensors.get(name)),
            ...Array.from(node.inputs.keys()).map(name => inputTensors.get(name)),
            ...paramArgs
        ].join(', ');

        // Calculate optimal grid and block dimensions based on tensor shapes and kernel type
        const { gridDim, blockDim, sharedMemSize } = this.calculateOptimalGridBlock(node, outputTensors, inputTensors);
        
        executionCalls.push(`  ${node.functionName}<<<${gridDim}, ${blockDim}, ${sharedMemSize}>>>(${kernelArgs});`);
        executionCalls.push(`  CUDA_CHECK(cudaGetLastError());`);
    }

    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);
    
    // Generate proper parameter types based on tensor data types
    const getTensorParams = (name: string, dtype: string = "float32") => {
        const dataType = dtype === "int32" ? "int" : "float";
        return [`${dataType}* ${name}_data`, `const int* ${name}_shape`, `int ${name}_dims`];
    };
    
    const inputParams = Array.from(graphInputs.entries()).flatMap(([name, { node, port }]) => {
        const inputSpec = node.inputs.get(port)!;
        return getTensorParams(name, inputSpec.dtype);
    });
    const outputParams = Array.from(graphOutputs.keys()).flatMap(name => getTensorParams(name, "float32"));
    const parameterParams = paramNames.flatMap(name => getTensorParams(name, "float32"));
    const hostFunctionSignatureParams = [...inputParams, ...outputParams, ...parameterParams, 'char* workspace', 'size_t workspace_size'];

    const tensorStruct = `
// Debug mode bounds checking (can be disabled for release builds)
#ifndef NDEBUG
#define TENSOR_BOUNDS_CHECK 1
#define TENSOR_BOUNDS_CHECK_VERBOSE 1  // Enable verbose error messages in debug mode
#else
#define TENSOR_BOUNDS_CHECK 0
#define TENSOR_BOUNDS_CHECK_VERBOSE 0
#endif

template<typename T>
struct Tensor {
  T* data;
  const int* shape;
  int dims;

  __device__ inline T& operator()(int i) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 1 || i < 0 || i >= shape[0]) {
      #if TENSOR_BOUNDS_CHECK_VERBOSE
      printf("FATAL: Tensor bounds error: 1D access [%d] out of bounds [0, %d) for %dD tensor\\n", i, shape[0], dims);
      #endif
      #if defined(__CUDA_ARCH__)
      asm("trap;");  // Trigger a GPU trap/exception
      #endif
    }
    #endif
    return data[i]; 
  }
  
  __device__ inline T& operator()(int i, int j) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 2 || i < 0 || i >= shape[0] || j < 0 || j >= shape[1]) {
      #if TENSOR_BOUNDS_CHECK_VERBOSE
      printf("FATAL: Tensor bounds error: 2D access [%d,%d] out of bounds [0,%d)x[0,%d) for %dD tensor\\n", 
             i, j, shape[0], shape[1], dims);
      #endif
      #if defined(__CUDA_ARCH__)
      asm("trap;");  // Trigger a GPU trap/exception
      #endif
    }
    #endif
    return data[i * shape[1] + j]; 
  }
  
  __device__ inline T& operator()(int i, int j, int k) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 3 || i < 0 || i >= shape[0] || j < 0 || j >= shape[1] || k < 0 || k >= shape[2]) {
      #if TENSOR_BOUNDS_CHECK_VERBOSE
      printf("FATAL: Tensor bounds error: 3D access [%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d) for %dD tensor\\n", 
             i, j, k, shape[0], shape[1], shape[2], dims);
      #endif
      #if defined(__CUDA_ARCH__)
      asm("trap;");  // Trigger a GPU trap/exception
      #endif
    }
    #endif
    return data[(i * shape[1] + j) * shape[2] + k]; 
  }
  
  __device__ inline T& operator()(int i, int j, int k, int l) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 4 || i < 0 || i >= shape[0] || j < 0 || j >= shape[1] || 
        k < 0 || k >= shape[2] || l < 0 || l >= shape[3]) {
      #if TENSOR_BOUNDS_CHECK_VERBOSE
      printf("FATAL: Tensor bounds error: 4D access [%d,%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d)x[0,%d) for %dD tensor\\n", 
             i, j, k, l, shape[0], shape[1], shape[2], shape[3], dims);
      #endif
      #if defined(__CUDA_ARCH__)
      asm("trap;");  // Trigger a GPU trap/exception
      #endif
    }
    #endif
    return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l]; 
  }
  
  // Helper method to get total number of elements
  __device__ inline int total_elements() const {
    int total = 1;
    for (int i = 0; i < dims; i++) {
      total *= shape[i];
    }
    return total;
  }
};
`;
    
    const tensorInstantiations: string[] = [];
    
    // Create tensor instantiations for intermediate tensors using memory pools
    intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
        const varName = `intermediate_${intermediateIdx}`;
        const shapeVar = `${varName}_shape`;
        const tensorVar = `${varName}_tensor`;
        
        if (tensorInfo.poolId?.startsWith('inplace_')) {
            // For in-place operations, use the same memory offset as the source tensor
            const sourceKey = tensorInfo.poolId.replace('inplace_', '');
            const sourceTensor = intermediateTensors.get(sourceKey);
            if (sourceTensor && sourceTensor.offset >= 0) {
                // Use the same offset as the source tensor but with our own shape
                tensorInstantiations.push(`  float* ${varName}_data = (float*)(workspace + ${sourceTensor.offset}); // In-place reuse of ${sourceKey}`);
                tensorInstantiations.push(`  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`);
            } else {
                // Fallback to offset-based allocation if source not found
                tensorInstantiations.push(`  float* ${varName}_data = (float*)(workspace + ${tensorInfo.offset});`);
                tensorInstantiations.push(`  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`);
            }
        } else {
            // Use offset-based allocation from memory pools with bounds checking
            tensorInstantiations.push(`  // Pool: ${tensorInfo.poolId || 'default'}, Offset: ${tensorInfo.offset}, Size: ${tensorInfo.size} bytes`);
            tensorInstantiations.push(`  float* ${varName}_data = (float*)(workspace + ${tensorInfo.offset});`);
            tensorInstantiations.push(`  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`);
        }
        intermediateIdx++;
    }

    // Create tensor instantiations for graph inputs (with proper type handling)
    for (const [inputName, { node, port }] of graphInputs.entries()) {
        const inputSpec = node.inputs.get(port)!;
        const tensorType = inputSpec.dtype === "int32" ? "int" : "float";
        const dataType = inputSpec.dtype === "int32" ? "int" : "float";
        tensorInstantiations.push(`  Tensor<${tensorType}> ${inputName} = {(${dataType}*)${inputName}_data, ${inputName}_shape, ${inputName}_dims};`);
    }
    
    // Create tensor instantiations for graph outputs (always float for now)
    for (const outputName of graphOutputs.keys()) {
        tensorInstantiations.push(`  Tensor<float> ${outputName} = {${outputName}_data, ${outputName}_shape, ${outputName}_dims};`);
    }
    
    // Create tensor instantiations for parameters (always float for now)
    for (const paramName of paramNames) {
        tensorInstantiations.push(`  Tensor<float> ${paramName} = {${paramName}_data, ${paramName}_shape, ${paramName}_dims};`);
    }

    const kernelCode = `
#include <cuda_runtime.h>
#include <cfloat>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

${tensorStruct}

// ======================================================
// Dynamic Memory Allocator
// ======================================================
class WorkspaceAllocator {
private:
    char* base_ptr;
    size_t current_offset;
    size_t total_size;
    
    // Helper to align to 256-byte boundaries for optimal GPU performance
    inline size_t align_to_256_bytes(size_t size) const {
        return (size + 255) & ~255;
    }
    
public:
    WorkspaceAllocator(char* workspace, size_t workspace_size) 
        : base_ptr(workspace), current_offset(0), total_size(workspace_size) {}
    
    void* allocate(size_t size) {
        size_t aligned_size = align_to_256_bytes(size);
        
        if (current_offset + aligned_size > total_size) {
            fprintf(stderr, "ERROR: Workspace allocator out of memory. Requested: %zu bytes (aligned: %zu), Available: %zu bytes\\n", 
                    size, aligned_size, total_size - current_offset);
            return nullptr;
        }
        
        void* ptr = base_ptr + current_offset;
        current_offset += aligned_size;
        return ptr;
    }
    
    size_t get_used_size() const {
        return current_offset;
    }
    
    void reset() {
        current_offset = 0;
    }
};

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
  // --- Input Validation ---
  if (!workspace) {
    fprintf(stderr, "Error: Null workspace pointer passed to executeGraph\\n");
    return;
  }
  
  if (workspace_size < ${workspaceSize}) {
    fprintf(stderr, "Error: Insufficient workspace size. Required: ${workspaceSize} bytes, Provided: %zu bytes\\n", workspace_size);
    return;
  }
  
  // --- Initialize Dynamic Memory Allocator ---
  WorkspaceAllocator allocator(workspace, workspace_size);

  // --- Variable Declarations ---
${variableDeclarations.join("\n")}

  // --- Tensor Struct Instantiation ---
${tensorInstantiations.join("\n")}

  // --- Kernel Launch Sequence ---
${executionCalls.join("\n")}
  
  // --- Synchronization for completion verification ---
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // --- Report memory usage ---
  size_t used_memory = allocator.get_used_size();
  if (used_memory > 0) {
    // Uncomment for debugging memory usage
    // printf("Workspace memory used: %zu bytes (%.2f MB)\\n", used_memory, used_memory / (1024.0 * 1024.0));
  }
  
  // --- End Execution Flow ---
}
    `;

    return { kernelCode, workspaceSize, filename };
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
      intermediateTensors: Map<string, { spec: { shape: number[], dtype: string }, size: number, liveStart: number, liveEnd: number, offset: number, poolId?: string }>,
      workspaceSize: number,
      tensorRegistry: Map<string, { varName: string, spec: { shape: number[], dtype: string } }>,
      memoryPools: Map<string, { size: number, tensors: string[], offset: number }>
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

    const intermediateTensors = new Map<string, { spec: { shape: number[], dtype: string }, size: number, liveStart: number, liveEnd: number, offset: number, poolId?: string }>();
    
    // First pass: collect all intermediate tensors and their lifetimes
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
    });

    // Second pass: enhanced tensor lifetime analysis with transitive dependencies
    this.computeEnhancedLifetimes(graph, executionOrder, intermediateTensors);

    // Third pass: detect in-place operation opportunities
    this.detectInPlaceOperations(graph, executionOrder, intermediateTensors);

    // Fourth pass: create memory pools and allocate with advanced reuse
    const { memoryPools, workspaceSize } = this.allocateWithMemoryPools(intermediateTensors);
    
    const totalTensorSize = Array.from(intermediateTensors.values())
        .reduce((sum, tensor) => sum + this.alignTo256Bytes(tensor.size), 0);
    const efficiency = workspaceSize > 0 ? ((totalTensorSize / workspaceSize) * 100).toFixed(1) : "N/A";
    const savings = totalTensorSize - workspaceSize;
    
    console.log(`[Enhanced Memory Planning] Total workspace size: ${workspaceSize} bytes (${(workspaceSize / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`[Enhanced Memory Planning] Total tensor size (if not reused): ${totalTensorSize} bytes (${(totalTensorSize / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`[Enhanced Memory Planning] Memory saved: ${savings} bytes (${(savings / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`[Enhanced Memory Planning] Memory efficiency: ${efficiency}%`);
    console.log(`[Enhanced Memory Planning] Memory pools created: ${memoryPools.size}`);

    return { intermediateTensors, workspaceSize, tensorRegistry, memoryPools };
  }

  private computeEnhancedLifetimes(
    graph: CudaGraph, 
    executionOrder: CudaNode[], 
    intermediateTensors: Map<string, { spec: { shape: number[], dtype: string }, size: number, liveStart: number, liveEnd: number, offset: number, poolId?: string }>
  ): void {
    // Build dependency graph for transitive lifetime analysis
    const dependencyGraph = new Map<string, Set<string>>();
    const reverseDependencyGraph = new Map<string, Set<string>>();
    
    for (const conn of graph.connections) {
      const sourceKey = `${conn.fromNode.id}:${conn.fromPort}`;
      const targetKey = `${conn.toNode.id}:${conn.toPort}`;
      
      if (!dependencyGraph.has(sourceKey)) dependencyGraph.set(sourceKey, new Set());
      if (!reverseDependencyGraph.has(targetKey)) reverseDependencyGraph.set(targetKey, new Set());
      
      dependencyGraph.get(sourceKey)!.add(targetKey);
      reverseDependencyGraph.get(targetKey)!.add(sourceKey);
    }

    // Compute last usage considering all downstream dependencies
    executionOrder.forEach((node, i) => {
        for (const [inputPort, spec] of node.inputs) {
            const conn = Array.from(graph.connections).find(c => c.toNode === node && c.toPort === inputPort);
            if (conn) {
                const sourceKey = `${conn.fromNode.id}:${conn.fromPort}`;
                if (intermediateTensors.has(sourceKey)) {
                    // Update direct usage
                    intermediateTensors.get(sourceKey)!.liveEnd = Math.max(intermediateTensors.get(sourceKey)!.liveEnd, i);
                    
                    // Also consider transitive dependencies - if this tensor feeds into long-lived computations
                    this.propagateLifetimeExtension(sourceKey, i, dependencyGraph, intermediateTensors, executionOrder);
                }
            }
        }
    });
  }

  private propagateLifetimeExtension(
    tensorKey: string,
    currentEnd: number,
    dependencyGraph: Map<string, Set<string>>,
    intermediateTensors: Map<string, any>,
    executionOrder: CudaNode[]
  ): void {
    const visited = new Set<string>();
    const queue = [{ key: tensorKey, end: currentEnd }];
    
    while (queue.length > 0) {
      const { key, end } = queue.shift()!;
      if (visited.has(key)) continue;
      visited.add(key);
      
      const dependencies = dependencyGraph.get(key);
      if (dependencies) {
        for (const depKey of dependencies) {
          if (intermediateTensors.has(depKey)) {
            const tensor = intermediateTensors.get(depKey)!;
            const extendedEnd = Math.min(end + 2, executionOrder.length - 1); // Small extension for safety
            if (tensor.liveEnd < extendedEnd) {
              tensor.liveEnd = extendedEnd;
              queue.push({ key: depKey, end: extendedEnd });
            }
          }
        }
      }
    }
  }

  private detectInPlaceOperations(
    graph: CudaGraph,
    executionOrder: CudaNode[],
    intermediateTensors: Map<string, any>
  ): void {
    // Detect operations that can reuse input buffers
    const inPlaceCapableOps = new Set(['relu_forward', 'add_forward', 'scale_forward']);
    
    for (const node of executionOrder) {
      if (inPlaceCapableOps.has(node.functionName)) {
        // For element-wise operations, try to reuse the largest input buffer
        const inputKeys = Array.from(node.inputs.keys()).map(port => {
          const conn = Array.from(graph.connections).find(c => c.toNode === node && c.toPort === port);
          return conn ? `${conn.fromNode.id}:${conn.fromPort}` : null;
        }).filter(k => k !== null) as string[];
        
        const outputKeys = Array.from(node.outputs.keys()).map(port => `${node.id}:${port}`);
        
        if (inputKeys.length > 0 && outputKeys.length > 0) {
          // Find the largest input that can be reused
          let bestInputKey = inputKeys[0];
          let bestInputSize = 0;
          
          for (const inputKey of inputKeys) {
            const inputTensor = intermediateTensors.get(inputKey);
            if (inputTensor && inputTensor.size > bestInputSize) {
              bestInputSize = inputTensor.size;
              bestInputKey = inputKey;
            }
          }
          
          const outputKey = outputKeys[0];
          const outputTensor = intermediateTensors.get(outputKey);
          const inputTensor = intermediateTensors.get(bestInputKey);
          
          // If output size matches input size, mark for in-place operation
          if (inputTensor && outputTensor && inputTensor.size === outputTensor.size) {
            outputTensor.poolId = `inplace_${bestInputKey}`;
            console.log(`[In-Place Detection] ${node.functionName} can reuse buffer from ${bestInputKey}`);
          }
        }
      }
    }
  }

  private allocateWithMemoryPools(
    intermediateTensors: Map<string, { spec: { shape: number[], dtype: string }, size: number, liveStart: number, liveEnd: number, offset: number, poolId?: string }>
  ): { memoryPools: Map<string, { size: number, tensors: string[], offset: number }>, workspaceSize: number } {
    const memoryPools = new Map<string, { size: number, tensors: string[], offset: number }>();
    
    // Group tensors by size categories for pooling
    const sizeCategories = new Map<string, Array<[string, any]>>();
    
    for (const [key, tensor] of intermediateTensors.entries()) {
      // Skip tensors marked for in-place operations
      if (tensor.poolId?.startsWith('inplace_')) {
        continue;
      }
      
      const category = this.categorizeTensorSize(tensor.size);
      if (!sizeCategories.has(category)) {
        sizeCategories.set(category, []);
      }
      sizeCategories.get(category)!.push([key, tensor]);
    }
    
    let workspaceOffset = 0;
    
    // Process each size category
    for (const [category, tensors] of sizeCategories.entries()) {
      // Sort tensors by lifetime for optimal allocation
      tensors.sort((a, b) => a[1].liveStart - b[1].liveStart);
      
      // Create memory pools for this category
      const pools = this.createPoolsForCategory(category, tensors);
      
      // Allocate pools in workspace
      for (const [poolId, pool] of pools.entries()) {
        pool.offset = workspaceOffset;
        workspaceOffset += pool.size;
        memoryPools.set(poolId, pool);
        
        // Assign tensors to their pool offsets
        for (const tensorKey of pool.tensors) {
          const tensor = intermediateTensors.get(tensorKey);
          if (tensor) {
            tensor.offset = pool.offset;
            tensor.poolId = poolId;
          }
        }
      }
    }
    
    // Handle in-place operations
    for (const [key, tensor] of intermediateTensors.entries()) {
      if (tensor.poolId?.startsWith('inplace_')) {
        const sourceKey = tensor.poolId.replace('inplace_', '');
        const sourceTensor = intermediateTensors.get(sourceKey);
        if (sourceTensor) {
          tensor.offset = sourceTensor.offset;
          console.log(`[In-Place Allocation] ${key} reuses buffer from ${sourceKey} at offset ${tensor.offset}`);
        }
      }
    }
    
    return { memoryPools, workspaceSize: workspaceOffset };
  }

  private categorizeTensorSize(size: number): string {
    const MB = 1024 * 1024;
    if (size < MB) return 'small';           // < 1MB
    if (size < 10 * MB) return 'medium';     // 1-10MB  
    if (size < 50 * MB) return 'large';      // 10-50MB
    return 'xlarge';                         // > 50MB
  }

  private createPoolsForCategory(
    category: string, 
    tensors: Array<[string, any]>
  ): Map<string, { size: number, tensors: string[], offset: number }> {
    const pools = new Map<string, { size: number, tensors: string[], offset: number }>();
    
    // Advanced allocation algorithm based on interval scheduling
    const intervals = tensors.map(([key, tensor]) => ({
      key,
      start: tensor.liveStart,
      end: tensor.liveEnd,
      size: this.alignTo256Bytes(tensor.size)
    }));
    
    // Sort by start time
    intervals.sort((a, b) => a.start - b.start);
    
    let poolCounter = 0;
    
    for (const interval of intervals) {
      let assignedPool: string | null = null;
      let bestPoolSize = Infinity;
      
      // Find the best-fitting pool that's available
      for (const [poolId, pool] of pools.entries()) {
        if (poolId.startsWith(category)) {
          // Check if this pool is available (all tensors in pool end before this tensor starts)
          const poolAvailable = pool.tensors.every(tensorKey => {
            const tensor = tensors.find(([k]) => k === tensorKey)?.[1];
            return tensor && tensor.liveEnd < interval.start;
          });
          
          if (poolAvailable && pool.size >= interval.size && pool.size < bestPoolSize) {
            assignedPool = poolId;
            bestPoolSize = pool.size;
          }
        }
      }
      
      if (assignedPool) {
        // Reuse existing pool
        pools.get(assignedPool)!.tensors.push(interval.key);
      } else {
        // Create new pool
        const poolId = `${category}_pool_${poolCounter++}`;
        pools.set(poolId, {
          size: interval.size,
          tensors: [interval.key],
          offset: -1 // Will be set later
        });
      }
    }
    
    console.log(`[Pool Creation] Category ${category}: Created ${pools.size} pools for ${tensors.length} tensors`);
    
    return pools;
  }

  private alignTo256Bytes(size: number): number {
    return Math.ceil(size / 256) * 256;
  }

  private calculateOptimalGridBlock(
    node: CudaNode, 
    outputTensors: Map<string, string>, 
    inputTensors: Map<string, string>
  ): { gridDim: string; blockDim: string; sharedMemSize: number } {
    // Get the primary output tensor shape for grid calculation
    const primaryOutput = Array.from(node.outputs.values())[0];
    if (!primaryOutput) {
      return { gridDim: "dim3(1, 1, 1)", blockDim: "dim3(256, 1, 1)", sharedMemSize: 0 };
    }

    const shape = primaryOutput.shape;
    
    // Calculate optimal configurations based on kernel type and tensor dimensions
    switch (node.functionName) {
      case 'embedding_forward':
        // Grid: (seq_len, batch_size), Block: (min(embed_dim, 1024))
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const blockSize = Math.min(embedDim, 1024);
          // Ensure block size is multiple of 32 (warp size)
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          return {
            gridDim: `dim3(${seqLen}, ${batchSize})`,
            blockDim: `dim3(${Math.min(alignedBlockSize, 1024)}, 1, 1)`,
            sharedMemSize: 0
          };
        }
        break;

      case 'positional_encoding_forward':
        // Grid: (div_ceil(embed_dim, block_size), seq_len, batch_size)
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const blockSize = this.defaultBlockSize;
          const gridX = Math.ceil(embedDim / blockSize);
          return {
            gridDim: `dim3(${gridX}, ${seqLen}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: 0
          };
        }
        break;

      case 'dense_forward_2d':
        // Grid: (div_ceil(output_features, block_size), batch_size)
        if (shape.length === 2) {
          const batchSize = shape[0];
          const outputFeatures = shape[1];
          const blockSize = this.defaultBlockSize;
          const gridX = Math.ceil(outputFeatures / blockSize);
          return {
            gridDim: `dim3(${Math.max(gridX, 1)}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: 0
          };
        }
        break;

      case 'dense_forward_3d':
        // Grid: (div_ceil(output_features, block_size), seq_len, batch_size)
        if (shape.length === 3) {
            const batchSize = shape[0];
            const seqLen = shape[1];
            const outputFeatures = shape[2];
            const blockSize = this.defaultBlockSize;
            const gridX = Math.ceil(outputFeatures / blockSize);
            return {
                gridDim: `dim3(${gridX}, ${seqLen}, ${batchSize})`,
                blockDim: `dim3(${blockSize}, 1, 1)`,
                sharedMemSize: 0
            };
        }
        break;

      case 'split_heads_forward':
        // Grid: (num_heads, seq_len, batch_size), Block: (head_dim)
        if (shape.length >= 4) {
          const batchSize = shape[0];
          const numHeads = shape[1];
          const seqLen = shape[2];
          const headDim = shape[3];
          const blockSize = Math.min(headDim, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          return {
            gridDim: `dim3(${numHeads}, ${seqLen}, ${batchSize})`,
            blockDim: `dim3(${Math.min(alignedBlockSize, 1024)}, 1, 1)`,
            sharedMemSize: 0
          };
        }
        break;

      case 'batched_matmul_transpose_b':
      case 'batched_matmul':
        // Tiled matrix multiplication: Grid covers output matrix with tiles
        if (shape.length >= 4) {
          const batchSize = shape[0];
          const numHeads = shape[1];
          const outputHeight = shape[2]; // M dimension (rows)
          const outputWidth = shape[3];  // N dimension (cols)
          
          // Choose optimal tile size based on matrix dimensions
          let TILE_SIZE = 16; // Default tile size for most cases
          
          // Use larger tiles for bigger matrices to reduce grid overhead
          if (outputHeight >= 512 && outputWidth >= 512) {
            TILE_SIZE = 32;
          } else if (outputHeight <= 64 && outputWidth <= 64) {
            TILE_SIZE = 8; // Smaller tiles for small matrices
          }
          
          // Calculate number of tiles needed to cover the output matrix
          const tilesX = Math.ceil(outputWidth / TILE_SIZE);   // Tiles along width (N)
          const tilesY = Math.ceil(outputHeight / TILE_SIZE);  // Tiles along height (M)
          const totalBatches = batchSize * numHeads;           // Total batched operations
          
          // Shared memory: two input tiles (A and B) for tiled multiplication
          const sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * 4; // 2 tiles * sizeof(float)
          
          return {
            gridDim: `dim3(${tilesX}, ${tilesY}, ${totalBatches})`,
            blockDim: `dim3(${TILE_SIZE}, ${TILE_SIZE}, 1)`,
            sharedMemSize: sharedMemSize
          };
        }
        break;

      case 'softmax_forward':
        if (shape.length === 2) {
          // 2D case: [batch, features]
          const batchSize = shape[0];
          const features = shape[1];
          const blockSize = Math.min(features, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          const finalBlockSize = Math.min(alignedBlockSize, 1024);
          return {
            gridDim: `dim3(${batchSize}, 1, 1)`,
            blockDim: `dim3(${finalBlockSize}, 1, 1)`,
            sharedMemSize: finalBlockSize * 4 // blockDim.x * sizeof(float)
          };
        } else if (shape.length === 4) {
          // 4D case: [batch, heads, seq, seq]
          const batchSize = shape[0];
          const numHeads = shape[1];
          const seqLen = shape[2];
          const blockSize = Math.min(seqLen, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          const finalBlockSize = Math.min(alignedBlockSize, 1024);
          return {
            gridDim: `dim3(${seqLen}, ${numHeads}, ${batchSize})`,
            blockDim: `dim3(${finalBlockSize}, 1, 1)`,
            sharedMemSize: finalBlockSize * 4 // blockDim.x * sizeof(float)
          };
        }
        break;

      case 'fused_scale_softmax_forward':
        // Fused scale + softmax operation with proper shared memory allocation
        if (shape.length === 4) {
          // 4D case: [batch, heads, seq, seq]
          const batchSize = shape[0];
          const numHeads = shape[1];
          const seqLen = shape[2];
          const blockSize = Math.min(seqLen, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          const finalBlockSize = Math.min(alignedBlockSize, 1024);
          return {
            gridDim: `dim3(${seqLen}, ${numHeads}, ${batchSize})`,
            blockDim: `dim3(${finalBlockSize}, 1, 1)`,
            sharedMemSize: finalBlockSize * 4 // blockDim.x * sizeof(float) for reduction
          };
        } else if (shape.length === 2) {
          // 2D case: [batch, features]
          const batchSize = shape[0];
          const features = shape[1];
          const blockSize = Math.min(features, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          const finalBlockSize = Math.min(alignedBlockSize, 1024);
          return {
            gridDim: `dim3(${batchSize}, 1, 1)`,
            blockDim: `dim3(${finalBlockSize}, 1, 1)`,
            sharedMemSize: finalBlockSize * 4 // blockDim.x * sizeof(float)
          };
        }
        break;

      case 'layer_norm_forward':
        // Grid: (seq_len, batch_size), Block: (min(feature_count, 1024))
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const featureCount = shape[2];
          const blockSize = Math.min(featureCount, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          const finalBlockSize = Math.min(alignedBlockSize, 1024);
          return {
            gridDim: `dim3(${seqLen}, ${batchSize})`,
            blockDim: `dim3(${finalBlockSize}, 1, 1)`,
            sharedMemSize: finalBlockSize * 4 // blockDim.x * sizeof(float)
          };
        }
        break;

      case 'scale_forward':
      case 'add_forward':
      case 'relu_forward':
        // Element-wise operations: calculate total elements and use 1D grid
        const totalElements = shape.reduce((a, b) => a * b, 1);
        const blockSize = this.defaultBlockSize;
        const gridSize = Math.ceil(totalElements / blockSize);
        return {
          gridDim: `dim3(${gridSize}, 1, 1)`,
          blockDim: `dim3(${blockSize}, 1, 1)`,
          sharedMemSize: 0
        };

      case 'concat_heads_forward':
        // Grid: (seq_len, batch_size), Block: (min(embed_dim, 1024))
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const blockSize = Math.min(embedDim, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          return {
            gridDim: `dim3(${seqLen}, ${batchSize})`,
            blockDim: `dim3(${Math.min(alignedBlockSize, 1024)}, 1, 1)`,
            sharedMemSize: 0
          };
        }
        break;

      default:
        // Default case: use tensor-aware sizing
        if (shape.length >= 2) {
          const batchSize = shape[0];
          const features = shape[shape.length - 1];
          const blockSize = Math.min(features, this.defaultBlockSize);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          const gridSize = Math.ceil(features / alignedBlockSize);
          return {
            gridDim: `dim3(${gridSize}, ${batchSize})`,
            blockDim: `dim3(${Math.min(alignedBlockSize, 1024)}, 1, 1)`,
            sharedMemSize: 0
          };
        }
    }

    // Fallback: improved default based on tensor size
    const totalElements = shape.reduce((a, b) => a * b, 1);
    const blockSize = this.defaultBlockSize;
    const gridSize = Math.ceil(totalElements / blockSize);
    return {
      gridDim: `dim3(${Math.min(gridSize, 65535)}, 1, 1)`,
      blockDim: `dim3(${blockSize}, 1, 1)`,
      sharedMemSize: 0
    };
  }

}
