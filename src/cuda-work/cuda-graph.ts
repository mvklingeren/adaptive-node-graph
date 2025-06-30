// ============================================================================
// Universal CUDA Graph Engine with Improved Parameter Interface
// ============================================================================
// This file implements the core, generic components for building and
// executing computation graphs on the GPU. It is designed to be
// domain-agnostic and can be used for any task that can be accelerated
// with CUDA, not just machine learning.
// ============================================================================

import crypto from "crypto";
import { CudaRuntime, CudaTensor, CudaKernel } from "./cuda-abstractions.js";

// ============================================================================
// Compiler Configuration
// ============================================================================

export interface CompilerConfig {
  useArrayInterface?: boolean;  // Use the improved array-based parameter interface
  enableBoundsChecking?: boolean;  // Enable tensor bounds checking in debug mode
  defaultBlockSize?: number;  // Default CUDA block size
  memoryAlignment?: number;  // Memory alignment in bytes (default 256)
  enableMemoryPooling?: boolean;  // Enable memory pool optimization
  verboseMemoryPlanning?: boolean;  // Print detailed memory planning info
}

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
  public readonly inputs = new Map<
    string,
    { shape: number[]; dtype: string }
  >();
  public readonly outputs = new Map<
    string,
    { shape: number[]; dtype: string }
  >();
  // Holds named parameter tensors (e.g., weights, biases) for this node.
  public readonly parameters = new Map<string, CudaTensor>();
  // Optional function to dynamically resolve output shapes from input shapes.
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
      throw new Error(
        `Input port '${portName}' not found on node ${this.name}.`
      );
    }
    // Potentially add more validation here if needed
    this.inputs.set(portName, { ...input, shape: newShape });
  }

  getKernelCall(
    outputTensorNames: Map<string, string>,
    inputTensorNames: Map<string, string>,
    parameterResolver: (nodeId: string, paramName: string) => string
  ): string {
    const outputArgs = Array.from(this.outputs.keys()).map((name) =>
      outputTensorNames.get(name)
    );
    const inputArgs = Array.from(this.inputs.keys()).map((name) =>
      inputTensorNames.get(name)
    );
    const paramArgs = Array.from(this.parameters.keys()).map((name) =>
      parameterResolver(this.id, name)
    );

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
// host-side execution function with improved parameter interface.
// ============================================================================

export class CudaGraphCompiler {
  private config: CompilerConfig;

  constructor(private runtime: CudaRuntime, config?: CompilerConfig) {
    this.config = {
      useArrayInterface: true,
      enableBoundsChecking: true,
      defaultBlockSize: 256,
      memoryAlignment: 256,
      enableMemoryPooling: true,
      verboseMemoryPlanning: false,
      ...config,
    };
  }

  async compile(
    graph: CudaGraph,
    inputShapes: Map<string, number[]>
  ): Promise<{
    kernel: CudaKernel;
    parameters: CudaTensor[];
    kernelCode: string;
    workspaceSize: number;
  }> {
    // Create unique parameter names for each tensor to avoid conflicts
    const allParams = new Map<string, CudaTensor>();
    const paramNameMapping = new Map<string, string>(); // Maps node parameter names to unique global names
    let paramCounter = 0;

    for (const node of graph.nodes.values()) {
      if (!node || !node.id) {
        console.warn("Skipping invalid node in parameter mapping");
        continue;
      }

      for (const [localParamName, tensor] of node.parameters) {
        if (!tensor || !tensor.id) {
          console.warn(
            `Skipping invalid tensor '${localParamName}' in node '${node.name}'`
          );
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

    const { kernelCode, workspaceSize, filename } = this.config
      .useArrayInterface
      ? this.generateImprovedKernelCode(graph, paramNames, paramNameMapping)
      : this.generateLegacyKernelCode(graph, paramNames, paramNameMapping);

    const kernel = await this.runtime.compile(kernelCode, filename);

    return { kernel, parameters: orderedParams, kernelCode, workspaceSize };
  }

  private generateImprovedKernelCode(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): { kernelCode: string; workspaceSize: number; filename: string } {
    const executionOrder = graph.getExecutionOrder();
    const filename = `generated-${graph.name}-kernel.cu`;

    const { intermediateTensors, workspaceSize, tensorRegistry, memoryPools } =
      this.planMemory(graph, executionOrder);

    // Add intermediate tensors to the registry BEFORE processing nodes
    const variableDeclarations: string[] = [];
    let intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
      const varName = `intermediate_${intermediateIdx}`;
      const shapeVar = `${varName}_shape`;
      const tensorVar = `${varName}_tensor`;
      variableDeclarations.push(
        `  const int ${shapeVar}[] = {${tensorInfo.spec.shape.join(", ")}};`
      );
      tensorRegistry.set(key, { varName: tensorVar, spec: tensorInfo.spec });
      intermediateIdx++;
    }

    const uniqueKernels = new Map<string, { code: string; node: CudaNode }>();
    for (const node of executionOrder) {
      // Skip nodes with empty device code (like input nodes)
      if (node.deviceCode.trim() && !uniqueKernels.has(node.functionName)) {
        uniqueKernels.set(node.functionName, { code: node.deviceCode, node });
      }
    }
    const kernelDefinitions = Array.from(uniqueKernels.values())
      .map((k) => k.code)
      .join("\n\n");

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
          throw new Error(
            `Compiler Error: Could not find output tensor for ${node.name}:${outputPort} (key: ${key})`
          );
        }
        outputTensors.set(outputPort, tensorInfo.varName);
      }

      const inputTensors = new Map<string, string>();
      for (const [inputPort] of node.inputs) {
        const conn = Array.from(graph.connections).find(
          (c) => c.toNode === node && c.toPort === inputPort
        );
        const sourceKey = conn
          ? `${conn.fromNode.id}:${conn.fromPort}`
          : `${node.id}:${inputPort}`;
        const tensorInfo = tensorRegistry.get(sourceKey);
        if (!tensorInfo) {
          throw new Error(
            `Compiler Error: Could not find source tensor for ${node.name}:${inputPort} (key: ${sourceKey})`
          );
        }
        inputTensors.set(inputPort, tensorInfo.varName);
      }

      // Get kernel call with array-based parameter access
      const kernelCall = node.getKernelCall(
        outputTensors,
        inputTensors,
        (nodeId, paramName) => {
          const nodeParamKey = `${nodeId}:${paramName}`;
          const globalParamName = paramNameMapping.get(nodeParamKey);
          if (!globalParamName) {
            throw new Error(
              `Compiler Error: Could not find global parameter name for ${node.name}:${paramName}`
            );
          }
          const paramIndex = paramNames.indexOf(globalParamName);
          return `param_tensors[${paramIndex}]`;
        }
      );

      // Calculate optimal grid and block dimensions based on tensor shapes and kernel type
      const { gridDim, blockDim, sharedMemSize } =
        this.calculateOptimalGridBlock(node, outputTensors, inputTensors);

      executionCalls.push(
        `  ${node.functionName}<<<${gridDim}, ${blockDim}, ${sharedMemSize}>>>(${kernelCall});`
      );
      executionCalls.push(`  CUDA_CHECK(cudaGetLastError());`);
    }

    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);

    // Generate tensor instantiations with memory pooling
    const tensorInstantiations: string[] = [];
    intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
      const varName = `intermediate_${intermediateIdx}`;
      const shapeVar = `${varName}_shape`;
      const tensorVar = `${varName}_tensor`;

      if (tensorInfo.poolId?.startsWith("inplace_")) {
        // For in-place operations, use the same memory offset as the source tensor
        const sourceKey = tensorInfo.poolId.replace("inplace_", "");
        const sourceTensor = intermediateTensors.get(sourceKey);
        if (sourceTensor && sourceTensor.offset >= 0) {
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(workspace + ${sourceTensor.offset}); // In-place reuse of ${sourceKey}`
          );
          tensorInstantiations.push(
            `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
          );
        } else {
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(workspace + ${tensorInfo.offset});`
          );
          tensorInstantiations.push(
            `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
          );
        }
      } else {
        tensorInstantiations.push(
          `  // Pool: ${tensorInfo.poolId || "default"}, Offset: ${
            tensorInfo.offset
          }, Size: ${tensorInfo.size} bytes`
        );
        tensorInstantiations.push(
          `  float* ${varName}_data = (float*)(workspace + ${tensorInfo.offset});`
        );
        tensorInstantiations.push(
          `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
        );
      }
      intermediateIdx++;
    }

    // Create input/output tensor instantiations
    for (const [inputName, { node, port }] of graphInputs.entries()) {
      const inputSpec = node.inputs.get(port)!;
      const tensorType = inputSpec.dtype === "int32" ? "int" : "float";
      tensorInstantiations.push(
        `  Tensor<${tensorType}> ${inputName} = {(${tensorType}*)input_data, input_shape, input_dims};`
      );
    }

    for (const outputName of graphOutputs.keys()) {
      tensorInstantiations.push(
        `  Tensor<float> ${outputName} = {(float*)output_data, output_shape, output_dims};`
      );
    }

    const tensorStruct = this.getTensorStructDefinition();

    const kernelCode = `
#include <cuda_runtime.h>
#include <cfloat>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, \\
                cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
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
    
    // Helper to align to ${
      this.config.memoryAlignment
    }-byte boundaries for optimal GPU performance
    inline size_t align_to_boundary(size_t size) const {
        const size_t alignment = ${this.config.memoryAlignment};
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
public:
    WorkspaceAllocator(char* workspace, size_t workspace_size) 
        : base_ptr(workspace), current_offset(0), total_size(workspace_size) {}
    
    void* allocate(size_t size) {
        size_t aligned_size = align_to_boundary(size);
        
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
// Main Host-Side Execution Function (Array-Based Interface)
// ======================================================
extern "C" void executeGraph(
    // Input tensor
    void* input_data,
    const int* input_shape,
    int input_dims,
    const char* input_dtype,
    
    // Output tensor
    void* output_data,
    const int* output_shape,
    int output_dims,
    const char* output_dtype,
    
    // Parameters bundle
    void** param_data,
    const int** param_shapes,
    const int* param_dims,
    const char** param_dtypes,
    int param_count,
    
    // Workspace
    char* workspace,
    size_t workspace_size
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
    
    if (param_count != ${paramNames.length}) {
        fprintf(stderr, "Error: Expected ${
          paramNames.length
        } parameters, got %d\\n", param_count);
        return;
    }
    
    // --- Initialize Dynamic Memory Allocator ---
    WorkspaceAllocator allocator(workspace, workspace_size);

    // --- Variable Declarations ---
${variableDeclarations.join("\n")}

    // --- Create Parameter Tensors ---
    Tensor<float> param_tensors[${paramNames.length}];
    for (int i = 0; i < ${paramNames.length}; i++) {
        param_tensors[i] = {
            (float*)param_data[i],
            param_shapes[i],
            param_dims[i]
        };
    }

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
}

// ======================================================
// Simplified C-style Interface (for easier FFI)
// ======================================================
extern "C" void executeGraphSimple(
    void* input_data,
    int input_batch_size,
    int input_seq_len,
    
    void* output_data,
    int output_batch_size,
    int output_vocab_size,
    
    void** param_data,
    
    char* workspace,
    size_t workspace_size
) {
    // Fixed shapes for this specific model
    const int input_shape[] = {input_batch_size, input_seq_len};
    const int output_shape[] = {output_batch_size, output_vocab_size};
    
    // Parameter shapes are hardcoded for this specific model configuration
    ${this.generateParameterShapeDeclarations(graph, paramNames)}
    
    const int* param_shapes[${paramNames.length}] = {
        ${paramNames.map((_, i) => `param_shape_${i}`).join(",\\n        ")}
    };
    
    const int param_dims[${paramNames.length}] = {
        ${this.generateParameterDims(graph, paramNames).join(", ")}
    };
    
    const char* param_dtypes[${paramNames.length}] = {
        ${paramNames.map(() => '"float32"').join(", ")}
    };
    
    executeGraph(
        input_data, input_shape, 2, "int32",
        output_data, output_shape, 2, "float32",
        param_data, param_shapes, param_dims, param_dtypes, ${
          paramNames.length
        },
        workspace, workspace_size
    );
}
`;

    return { kernelCode, workspaceSize, filename };
  }

  private generateLegacyKernelCode(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): { kernelCode: string; workspaceSize: number; filename: string } {
    // This is the original kernel generation code for backward compatibility
    // Implementation would be the same as the original generateKernelCode method
    throw new Error("Legacy kernel generation not implemented in this example");
  }

  private generateParameterShapeDeclarations(
    graph: CudaGraph,
    paramNames: string[]
  ): string {
    const declarations: string[] = [];

    paramNames.forEach((paramName, i) => {
      // Extract tensor info from the parameter name or graph
      const tensor = this.findTensorByGlobalName(graph, paramName);
      if (tensor && tensor.shape) {
        declarations.push(
          `    const int param_shape_${i}[] = {${tensor.shape.join(
            ", "
          )}}; // ${paramName}`
        );
      } else {
        // Fallback based on naming convention
        if (paramName.includes("embeddings")) {
          declarations.push(
            `    const int param_shape_${i}[] = {65, 384}; // ${paramName}`
          );
        } else if (paramName.includes("frequencies")) {
          declarations.push(
            `    const int param_shape_${i}[] = {192}; // ${paramName}`
          );
        } else if (paramName.includes("weights")) {
          declarations.push(
            `    const int param_shape_${i}[] = {384, 384}; // ${paramName}`
          );
        } else if (paramName.includes("bias")) {
          declarations.push(
            `    const int param_shape_${i}[] = {384}; // ${paramName}`
          );
        } else {
          declarations.push(
            `    const int param_shape_${i}[] = {1}; // ${paramName}`
          );
        }
      }
    });

    return declarations.join("\\n");
  }

  private generateParameterDims(
    graph: CudaGraph,
    paramNames: string[]
  ): number[] {
    return paramNames.map((paramName) => {
      const tensor = this.findTensorByGlobalName(graph, paramName);
      return tensor ? tensor.shape.length : 1;
    });
  }

  private findTensorByGlobalName(
    graph: CudaGraph,
    globalName: string
  ): CudaTensor | null {
    // Helper to find tensor information from global parameter name
    for (const node of graph.nodes.values()) {
      for (const [localName, tensor] of node.parameters) {
        const nodeParamKey = `${node.id}:${localName}`;
        // Check if this matches our global name pattern
        if (globalName.includes(localName)) {
          return tensor;
        }
      }
    }
    return null;
  }

  private getTensorStructDefinition(): string {
    const boundsCheck = this.config.enableBoundsChecking;
    return `
// Debug mode bounds checking (can be disabled for release builds)
#ifndef NDEBUG
#define TENSOR_BOUNDS_CHECK ${boundsCheck ? 1 : 0}
#define TENSOR_BOUNDS_CHECK_VERBOSE ${boundsCheck ? 1 : 0}
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
      asm("trap;");
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
      asm("trap;");
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
      asm("trap;");
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
  }

  private _setAndPropagateShapes(
    graph: CudaGraph,
    inputShapes: Map<string, number[]>
  ): void {
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
      const finalShape = shape.map((dim) => (dim === -1 ? batchSize : dim));
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
          if (
            !oldShape ||
            JSON.stringify(oldShape) !== JSON.stringify(spec.shape)
          ) {
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
          const inputShapes = JSON.stringify(
            Object.fromEntries(node.inputs),
            null,
            2
          );
          throw new Error(
            `Compiler error: Node '${node.name}' failed to resolve its output shape for port '${outputPort}'. Current input shapes for this node:\n${inputShapes}`
          );
        }
      }
    }
  }

  private planMemory(
    graph: CudaGraph,
    executionOrder: CudaNode[]
  ): {
    intermediateTensors: Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >;
    workspaceSize: number;
    tensorRegistry: Map<
      string,
      { varName: string; spec: { shape: number[]; dtype: string } }
    >;
    memoryPools: Map<
      string,
      { size: number; tensors: string[]; offset: number }
    >;
  } {
    const tensorRegistry = new Map<
      string,
      { varName: string; spec: { shape: number[]; dtype: string } }
    >();
    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);

    for (const [inputPort, { node, port }] of graphInputs.entries()) {
      tensorRegistry.set(`${node.id}:${port}`, {
        varName: inputPort,
        spec: node.inputs.get(port)!,
      });
    }
    for (const [outputPort, { node, port }] of graphOutputs.entries()) {
      tensorRegistry.set(`${node.id}:${outputPort}`, {
        varName: outputPort,
        spec: node.outputs.get(port)!,
      });
    }

    const intermediateTensors = new Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >();

    // First pass: collect all intermediate tensors and their lifetimes
    executionOrder.forEach((node, i) => {
      for (const [outputPort, spec] of node.outputs) {
        const key = `${node.id}:${outputPort}`;
        if (spec.shape.includes(-1)) {
          const inputShapes = JSON.stringify(Array.from(node.inputs.entries()));
          throw new Error(
            `Compiler error: Unresolved dynamic shape ${JSON.stringify(
              spec.shape
            )} for output '${outputPort}' of node '${
              node.name
            }'. Node inputs: ${inputShapes}`
          );
        }
        if (!tensorRegistry.has(key)) {
          const size = spec.shape.reduce((a, b) => a * b, 1) * 4; // size in bytes
          intermediateTensors.set(key, {
            spec,
            size,
            liveStart: i,
            liveEnd: i,
            offset: -1,
          });
        }
      }
    });

    // Second pass: enhanced tensor lifetime analysis
    this.computeEnhancedLifetimes(graph, executionOrder, intermediateTensors);

    // Third pass: detect in-place operation opportunities (if enabled)
    if (this.config.enableMemoryPooling) {
      this.detectInPlaceOperations(graph, executionOrder, intermediateTensors);
    }

    // Fourth pass: create memory pools and allocate
    const { memoryPools, workspaceSize } =
      this.allocateWithMemoryPools(intermediateTensors);

    if (this.config.verboseMemoryPlanning) {
      const totalTensorSize = Array.from(intermediateTensors.values()).reduce(
        (sum, tensor) => sum + this.alignTo256Bytes(tensor.size),
        0
      );
      const efficiency =
        workspaceSize > 0
          ? ((totalTensorSize / workspaceSize) * 100).toFixed(1)
          : "N/A";
      const savings = totalTensorSize - workspaceSize;

      console.log(
        `[Enhanced Memory Planning] Total workspace size: ${workspaceSize} bytes (${(
          workspaceSize /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `[Enhanced Memory Planning] Total tensor size (if not reused): ${totalTensorSize} bytes (${(
          totalTensorSize /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `[Enhanced Memory Planning] Memory saved: ${savings} bytes (${(
          savings /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `[Enhanced Memory Planning] Memory efficiency: ${efficiency}%`
      );
      console.log(
        `[Enhanced Memory Planning] Memory pools created: ${memoryPools.size}`
      );
    }

    return { intermediateTensors, workspaceSize, tensorRegistry, memoryPools };
  }

  private computeEnhancedLifetimes(
    graph: CudaGraph,
    executionOrder: CudaNode[],
    intermediateTensors: Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >
  ): void {
    // Build node index map for fast lookup
    const nodeIndexMap = new Map<string, number>();
    executionOrder.forEach((node, i) => {
      nodeIndexMap.set(node.id, i);
    });

    // First, set initial lifetimes based on production
    for (const [tensorKey, tensorInfo] of intermediateTensors.entries()) {
      // The tensor is "born" when it's produced
      const [nodeId, port] = tensorKey.split(":");
      const producerIndex = nodeIndexMap.get(nodeId);
      if (producerIndex !== undefined) {
        tensorInfo.liveStart = producerIndex;
        tensorInfo.liveEnd = producerIndex; // Initially assume it dies immediately
      }
    }

    // Now extend lifetimes based on usage
    for (const [tensorKey, tensorInfo] of intermediateTensors.entries()) {
      // Find all nodes that use this tensor as input
      for (const conn of graph.connections) {
        const sourceKey = `${conn.fromNode.id}:${conn.fromPort}`;
        if (sourceKey === tensorKey) {
          const targetNodeIndex = nodeIndexMap.get(conn.toNode.id);
          if (targetNodeIndex !== undefined) {
            // Extend the lifetime to include this usage
            tensorInfo.liveEnd = Math.max(tensorInfo.liveEnd, targetNodeIndex);
          }
        }
      }

      // Check if this tensor is a graph output (needs to live until the end)
      const graphOutputs = this.getGraphOutputs(graph);
      for (const [outputName, outputInfo] of graphOutputs) {
        if (`${outputInfo.node.id}:${outputInfo.port}` === tensorKey) {
          tensorInfo.liveEnd = executionOrder.length - 1;
          break;
        }
      }
    }
  }

  private detectInPlaceOperations(
    graph: CudaGraph,
    executionOrder: CudaNode[],
    intermediateTensors: Map<string, any>
  ): void {
    // DISABLED: In-place operations can cause memory corruption if not handled carefully
    // Keeping the function for future implementation with proper safety checks
    return;
  }

  private allocateWithMemoryPools(
    intermediateTensors: Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >
  ): {
    memoryPools: Map<
      string,
      { size: number; tensors: string[]; offset: number }
    >;
    workspaceSize: number;
  } {
    const memoryPools = new Map<
      string,
      { size: number; tensors: string[]; offset: number }
    >();

    // Sort tensors by liveStart, then by size (for better packing)
    const sortedTensors = Array.from(intermediateTensors.entries())
      .filter(([_, tensor]) => !tensor.poolId?.startsWith("inplace_"))
      .sort((a, b) => {
        if (a[1].liveStart !== b[1].liveStart) {
          return a[1].liveStart - b[1].liveStart;
        }
        return b[1].size - a[1].size; // Larger sizes first within same liveStart
      });

    // Track memory usage over time
    const memoryTimeline: Map<number, number> = new Map();
    let maxMemoryUsage = 0;

    for (const [tensorKey, tensor] of sortedTensors) {
      const alignedSize = this.alignTo256Bytes(tensor.size);

      // Find the earliest offset where this tensor can fit
      let bestOffset = 0;
      let currentOffset = 0;
      const freeRanges: Array<{ start: number; end: number }> = [
        { start: 0, end: Infinity },
      ];

      // Check all existing allocations that overlap with this tensor's lifetime
      for (const [otherKey, otherTensor] of intermediateTensors.entries()) {
        if (otherKey === tensorKey || otherTensor.offset === -1) continue;

        // Check if lifetimes overlap
        const overlaps = !(
          tensor.liveEnd < otherTensor.liveStart ||
          tensor.liveStart > otherTensor.liveEnd
        );

        if (overlaps) {
          // Remove this range from available space
          const otherEnd =
            otherTensor.offset + this.alignTo256Bytes(otherTensor.size);
          freeRanges.push({ start: otherEnd, end: Infinity });

          // Update free ranges to exclude this allocation
          for (let i = 0; i < freeRanges.length; i++) {
            const range = freeRanges[i];
            if (
              range.start < otherTensor.offset &&
              range.end > otherTensor.offset
            ) {
              // Split the range
              freeRanges[i] = { start: range.start, end: otherTensor.offset };
              if (range.end > otherEnd) {
                freeRanges.push({ start: otherEnd, end: range.end });
              }
            } else if (
              range.start >= otherTensor.offset &&
              range.end <= otherEnd
            ) {
              // Remove this range entirely
              freeRanges.splice(i, 1);
              i--;
            }
          }
        }
      }

      // Find the smallest offset that can fit our tensor
      freeRanges.sort((a, b) => a.start - b.start);
      for (const range of freeRanges) {
        if (range.end - range.start >= alignedSize) {
          bestOffset = range.start;
          break;
        }
      }

      // Record allocation
      tensor.offset = bestOffset;

      // Update max memory usage
      const tensorEnd = bestOffset + alignedSize;
      maxMemoryUsage = Math.max(maxMemoryUsage, tensorEnd);

      // Create a pool for this tensor
      const poolId = `pool_${tensorKey}`;
      tensor.poolId = poolId;
      memoryPools.set(poolId, {
        size: alignedSize,
        tensors: [tensorKey],
        offset: bestOffset,
      });
    }

    // Add safety margin (10%)
    const workspaceSize = Math.ceil(maxMemoryUsage * 1.1);

    return { memoryPools, workspaceSize };
  }

  private categorizeTensorSize(size: number): string {
    const MB = 1024 * 1024;
    if (size < MB) return "small"; // < 1MB
    if (size < 10 * MB) return "medium"; // 1-10MB
    if (size < 50 * MB) return "large"; // 10-50MB
    return "xlarge"; // > 50MB
  }

  private alignTo256Bytes(size: number): number {
    return (
      Math.ceil(size / this.config.memoryAlignment!) *
      this.config.memoryAlignment!
    );
  }

  private calculateOptimalGridBlock(
    node: CudaNode,
    outputTensors: Map<string, string>,
    inputTensors: Map<string, string>
  ): { gridDim: string; blockDim: string; sharedMemSize: number } {
    // Get the primary output tensor shape for grid calculation
    const primaryOutput = Array.from(node.outputs.values())[0];
    if (!primaryOutput) {
      return {
        gridDim: "dim3(1, 1, 1)",
        blockDim: `dim3(${this.config.defaultBlockSize}, 1, 1)`,
        sharedMemSize: 0,
      };
    }

    const shape = primaryOutput.shape;

    // Calculate optimal configurations based on kernel type and tensor dimensions
    switch (node.functionName) {
      case "embedding_forward":
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
            sharedMemSize: 0,
          };
        }
        break;

      case "positional_encoding_forward":
        // Grid: (div_ceil(embed_dim, block_size), seq_len, batch_size)
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const blockSize = this.config.defaultBlockSize!;
          const gridX = Math.ceil(embedDim / blockSize);
          return {
            gridDim: `dim3(${gridX}, ${seqLen}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      case "dense_forward_2d":
        // Grid: (div_ceil(output_features, block_size), batch_size)
        if (shape.length === 2) {
          const batchSize = shape[0];
          const outputFeatures = shape[1];
          const blockSize = this.config.defaultBlockSize!;
          const gridX = Math.ceil(outputFeatures / blockSize);
          return {
            gridDim: `dim3(${Math.max(gridX, 1)}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      case "dense_forward_3d":
        // Grid: (div_ceil(output_features, block_size), seq_len, batch_size)
        if (shape.length === 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const outputFeatures = shape[2];
          const blockSize = this.config.defaultBlockSize!;
          const gridX = Math.ceil(outputFeatures / blockSize);
          return {
            gridDim: `dim3(${gridX}, ${seqLen}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      case "split_heads_forward":
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
            sharedMemSize: 0,
          };
        }
        break;

      case "batched_matmul_transpose_b_tiled":
      case "batched_matmul_tiled":
        // Fixed: Proper tiled matrix multiplication grid calculation
        if (shape.length >= 4) {
          const batchSize = shape[0];
          const numHeads = shape[1];
          const outputRows = shape[2]; // M dimension
          const outputCols = shape[3]; // N dimension

          const TILE_SIZE = 32; // Match the kernel's TILE_SIZE

          // Calculate tiles needed to cover output matrix
          const tilesM = Math.ceil(outputRows / TILE_SIZE);
          const tilesN = Math.ceil(outputCols / TILE_SIZE);
          const totalTiles = tilesM * tilesN;

          // Shared memory for two tiles plus padding
          const sharedMemSize = 2 * TILE_SIZE * (TILE_SIZE + 1) * 4;

          return {
            gridDim: `dim3(${totalTiles}, ${numHeads}, ${batchSize})`,
            blockDim: `dim3(${TILE_SIZE}, ${TILE_SIZE}, 1)`,
            sharedMemSize: 0, // Set to 0 to use static shared memory allocation
          };
        }
        break;

      case "softmax_forward":
        if (shape.length === 2) {
          // 2D case: [batch, features]
          const batchSize = shape[0];
          const features = shape[1];
          // Use enough threads to handle the reduction efficiently
          const blockSize = Math.min(
            Math.max(96, Math.ceil(features / 32) * 32),
            1024
          );
          return {
            gridDim: `dim3(${batchSize}, 1, 1)`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: blockSize * 4, // sizeof(float) per thread
          };
        } else if (shape.length === 4) {
          // 4D case: [batch, heads, seq, seq]
          const batchSize = shape[0];
          const numHeads = shape[1];
          const seqLen = shape[2];
          const blockSize = Math.min(
            Math.max(96, Math.ceil(seqLen / 32) * 32),
            1024
          );
          return {
            gridDim: `dim3(${seqLen}, ${numHeads}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: blockSize * 4,
          };
        }
        break;

      case "fused_scale_softmax_forward":
        // Same as softmax but with scale factor
        if (shape.length === 4) {
          const batchSize = shape[0];
          const numHeads = shape[1];
          const seqLen = shape[2];
          const blockSize = Math.min(
            Math.max(128, Math.ceil(seqLen / 32) * 32),
            1024
          );
          return {
            gridDim: `dim3(${seqLen}, ${numHeads}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: blockSize * 4,
          };
        } else if (shape.length === 2) {
          const batchSize = shape[0];
          const features = shape[1];
          const blockSize = Math.min(
            Math.max(96, Math.ceil(features / 32) * 32),
            1024
          );
          return {
            gridDim: `dim3(${batchSize}, 1, 1)`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: blockSize * 4,
          };
        }
        break;

      case "layer_norm_forward":
        // Grid: (seq_len, batch_size), Block: (min(feature_count, 1024))
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const featureCount = shape[2];
          const blockSize = Math.min(
            Math.max(128, Math.ceil(featureCount / 32) * 32),
            1024
          );
          return {
            gridDim: `dim3(${seqLen}, ${batchSize})`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: blockSize * 4,
          };
        }
        break;

      case "scale_forward":
      case "add_forward":
      case "relu_forward":
        // Element-wise operations: calculate total elements and use 1D grid
        const totalElements = shape.reduce((a, b) => a * b, 1);
        const blockSize = this.config.defaultBlockSize!;
        const gridSize = Math.ceil(totalElements / blockSize);
        return {
          gridDim: `dim3(${gridSize}, 1, 1)`,
          blockDim: `dim3(${blockSize}, 1, 1)`,
          sharedMemSize: 0,
        };

      case "concat_heads_forward":
        // Grid: (num_heads, batch_size), Block: (min(head_dim * seq_len, 1024))
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const numHeads = 6; // Assuming 6 heads based on your model
          const headDim = embedDim / numHeads;
          return {
            gridDim: `dim3(${seqLen}, ${batchSize})`,
            blockDim: `dim3(${embedDim}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      default:
        // Default case: use tensor-aware sizing
        if (shape.length >= 2) {
          const batchSize = shape[0];
          const features = shape[shape.length - 1];
          const blockSize = Math.min(features, this.config.defaultBlockSize!);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          const gridSize = Math.ceil(features / alignedBlockSize);
          return {
            gridDim: `dim3(${gridSize}, ${batchSize})`,
            blockDim: `dim3(${Math.min(alignedBlockSize, 1024)}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
    }

    // Fallback: improved default based on tensor size
    const totalElements = shape.reduce((a, b) => a * b, 1);
    const blockSize = this.config.defaultBlockSize!;
    const gridSize = Math.ceil(totalElements / blockSize);
    return {
      gridDim: `dim3(${Math.min(gridSize, 65535)}, 1, 1)`,
      blockDim: `dim3(${blockSize}, 1, 1)`,
      sharedMemSize: 0,
    };
  }

  private getGraphOutputs(
    graph: CudaGraph
  ): Map<string, { node: CudaNode; port: string }> {
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
}

// ============================================================================
// Export Improved Compiler as Default
// ============================================================================

export default CudaGraphCompiler;

// ============================================================================
// Utility Functions for Graph Visualization and Debugging
// ============================================================================

export class GraphDebugger {
  static printGraphStructure(graph: CudaGraph): void {
    console.log(`\n=== Graph: ${graph.name} (${graph.id}) ===`);
    console.log(`Nodes: ${graph.nodes.size}`);
    console.log(`Connections: ${graph.connections.size}`);

    // Print nodes
    console.log("\nNodes:");
    for (const node of graph.nodes.values()) {
      console.log(`  ${node.name} (${node.id})`);
      console.log(`    Inputs: ${Array.from(node.inputs.keys()).join(", ")}`);
      console.log(`    Outputs: ${Array.from(node.outputs.keys()).join(", ")}`);
      console.log(
        `    Parameters: ${Array.from(node.parameters.keys()).join(", ")}`
      );
    }

    // Print connections
    console.log("\nConnections:");
    for (const conn of graph.connections) {
      console.log(
        `  ${conn.fromNode.name}:${conn.fromPort} -> ${conn.toNode.name}:${conn.toPort}`
      );
    }

    // Print execution order
    console.log("\nExecution Order:");
    const executionOrder = graph.getExecutionOrder();
    executionOrder.forEach((node, i) => {
      console.log(`  ${i}: ${node.name}`);
    });
  }

  static generateDotGraph(graph: CudaGraph): string {
    let dot = `digraph "${graph.name}" {\n`;
    dot += `  rankdir=TB;\n`;
    dot += `  node [shape=box, style=rounded];\n\n`;

    // Add nodes
    for (const node of graph.nodes.values()) {
      const inputs = Array.from(node.inputs.keys()).join(",");
      const outputs = Array.from(node.outputs.keys()).join(",");
      const params = Array.from(node.parameters.keys()).join(",");

      let label = `${node.name}\\n`;
      if (inputs) label += `in: ${inputs}\\n`;
      if (outputs) label += `out: ${outputs}\\n`;
      if (params) label += `params: ${params}`;

      dot += `  "${node.id}" [label="${label}"];\n`;
    }

    dot += "\n";

    // Add edges
    for (const conn of graph.connections) {
      dot += `  "${conn.fromNode.id}" -> "${conn.toNode.id}" [label="${conn.fromPort}→${conn.toPort}"];\n`;
    }

    dot += "}\n";
    return dot;
  }

  static validateGraph(graph: CudaGraph): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check for disconnected nodes
    const connectedNodes = new Set<string>();
    for (const conn of graph.connections) {
      connectedNodes.add(conn.fromNode.id);
      connectedNodes.add(conn.toNode.id);
    }

    for (const node of graph.nodes.values()) {
      if (!connectedNodes.has(node.id) && node.inputs.size > 0) {
        errors.push(
          `Node ${node.name} (${node.id}) has inputs but is not connected`
        );
      }
    }

    // Check for unresolved shapes
    for (const node of graph.nodes.values()) {
      for (const [port, spec] of node.outputs) {
        if (spec.shape.includes(-1)) {
          errors.push(
            `Node ${
              node.name
            } output ${port} has unresolved shape: [${spec.shape.join(", ")}]`
          );
        }
      }
    }

    // Check for type mismatches
    for (const conn of graph.connections) {
      const fromSpec = conn.fromNode.outputs.get(conn.fromPort);
      const toSpec = conn.toNode.inputs.get(conn.toPort);

      if (fromSpec && toSpec) {
        if (fromSpec.dtype !== toSpec.dtype) {
          errors.push(
            `Type mismatch: ${conn.fromNode.name}:${conn.fromPort} (${fromSpec.dtype}) -> ${conn.toNode.name}:${conn.toPort} (${toSpec.dtype})`
          );
        }

        // Check shape compatibility (basic check)
        if (fromSpec.shape.length !== toSpec.shape.length) {
          errors.push(
            `Shape rank mismatch: ${conn.fromNode.name}:${conn.fromPort} (rank ${fromSpec.shape.length}) -> ${conn.toNode.name}:${conn.toPort} (rank ${toSpec.shape.length})`
          );
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}

// ============================================================================
// Memory Planning Visualizer
// ============================================================================

export class MemoryPlanVisualizer {
  static visualizeMemoryPlan(
    intermediateTensors: Map<string, any>,
    workspaceSize: number
  ): string {
    let visualization = "\n=== Memory Allocation Plan ===\n";
    visualization += `Total Workspace: ${workspaceSize} bytes (${(workspaceSize / 1024 / 1024).toFixed(2)} MB)\n\n`;
    
    // Sort by offset
    const sortedTensors = Array.from(intermediateTensors.entries())
      .sort((a, b) => a[1].offset - b[1].offset);
    
    // Create ASCII visualization
    const scale = 80; // characters wide
    const maxOffset = workspaceSize;
    
    visualization += "Memory Layout:\n";
    visualization += "0" + " ".repeat(scale - 10) + `${(maxOffset / 1024 / 1024).toFixed(1)}MB\n`;
    visualization += "|" + "-".repeat(scale - 2) + "|\n";
    
    for (const [key, tensor] of sortedTensors) {
      const startPos = Math.floor((tensor.offset / maxOffset) * scale);
      const endPos = Math.floor(((tensor.offset + tensor.size) / maxOffset) * scale);
      const width = Math.max(1, endPos - startPos);
      
      let line = " ".repeat(startPos) + "█".repeat(width);
      const label = ` ${key} (${(tensor.size / 1024).toFixed(1)}KB)`;
      
      visualization += line + label + "\n";
    }
    
    visualization += "|" + "-".repeat(scale - 2) + "|\n\n";
    
    // Detailed tensor information
    visualization += "Tensor Details:\n";
    visualization += "Name".padEnd(40) + "Offset".padEnd(12) + "Size".padEnd(12) + "Lifetime\n";
    visualization += "-".repeat(76) + "\n";
    
    for (const [key, tensor] of sortedTensors) {
      const name = key.padEnd(40);
      const offset = tensor.offset.toString().padEnd(12);
      const size = `${(tensor.size / 1024).toFixed(1)}KB`.padEnd(12);
      const lifetime = `[${tensor.liveStart}-${tensor.liveEnd}]`;
      
      visualization += `${name}${offset}${size}${lifetime}\n`;
    }
    
    return visualization;
  }
}

// ============================================================================
// Graph Builder Helpers
// ============================================================================

export class GraphBuilder {
  private graph: CudaGraph;
  
  constructor(name: string) {
    this.graph = new CudaGraph(name);
  }
  
  addNode(node: CudaNode): GraphBuilder {
    this.graph.addNode(node);
    return this;
  }
  
  connect(
    fromNode: CudaNode | string,
    fromPort: string,
    toNode: CudaNode | string,
    toPort: string
  ): GraphBuilder {
    const from = typeof fromNode === 'string' 
      ? this.findNodeByName(fromNode) 
      : fromNode;
    const to = typeof toNode === 'string' 
      ? this.findNodeByName(toNode) 
      : toNode;
      
    if (!from || !to) {
      throw new Error(`Cannot find nodes for connection`);
    }
    
    this.graph.connect(from, fromPort, to, toPort);
    return this;
  }
  
  private findNodeByName(name: string): CudaNode | null {
    for (const node of this.graph.nodes.values()) {
      if (node.name === name) {
        return node;
      }
    }
    return null;
  }
  
  build(): CudaGraph {
    return this.graph;
  }
}

// ============================================================================
// Example: Creating a Simple Neural Network Graph
// ============================================================================

export function createExampleGraph(): CudaGraph {
  const builder = new GraphBuilder("SimpleNeuralNetwork");
  
  // Create nodes
  const inputNode = new CudaNode("", "input")
    .addOutput("output", [32, 784], "float32");
    
  const dense1 = new CudaNode(`
    __global__ void dense_forward(
      Tensor<float> output,
      Tensor<float> input,
      Tensor<float> weights,
      Tensor<float> bias
    ) {
      int batch_idx = blockIdx.y;
      int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
      
      if (batch_idx < output.shape[0] && out_idx < output.shape[1]) {
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < input.shape[1]; in_idx++) {
          sum += input(batch_idx, in_idx) * weights(in_idx, out_idx);
        }
        output(batch_idx, out_idx) = sum + bias(out_idx);
      }
    }
  `, "dense_forward")
    .addInput("input", [32, 784], "float32")
    .addOutput("output", [32, 128], "float32");
    
  const relu = new CudaNode(`
    __global__ void relu_forward(
      Tensor<float> output,
      Tensor<float> input
    ) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = input.total_elements();
      
      if (idx < total) {
        output.data[idx] = fmaxf(0.0f, input.data[idx]);
      }
    }
  `, "relu_forward")
    .addInput("input", [32, 128], "float32")
    .addOutput("output", [32, 128], "float32");
    
  const dense2 = new CudaNode(`
    __global__ void dense_forward(
      Tensor<float> output,
      Tensor<float> input,
      Tensor<float> weights,
      Tensor<float> bias
    ) {
      int batch_idx = blockIdx.y;
      int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
      
      if (batch_idx < output.shape[0] && out_idx < output.shape[1]) {
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < input.shape[1]; in_idx++) {
          sum += input(batch_idx, in_idx) * weights(in_idx, out_idx);
        }
        output(batch_idx, out_idx) = sum + bias(out_idx);
      }
    }
  `, "dense_forward_2")
    .addInput("input", [32, 128], "float32")
    .addOutput("output", [32, 10], "float32");
  
  // Build graph
  builder
    .addNode(inputNode)
    .addNode(dense1)
    .addNode(relu)
    .addNode(dense2)
    .connect(inputNode, "output", dense1, "input")
    .connect(dense1, "output", relu, "input")
    .connect(relu, "output", dense2, "input");
    
  return builder.build();
}

// ============================================================================
// Example Usage
// ============================================================================

/*
// Example of using the improved compiler
async function example() {
  const runtime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(runtime, {
    useArrayInterface: true,
    enableBoundsChecking: true,
    verboseMemoryPlanning: true
  });
  
  // Create a graph
  const graph = createExampleGraph();
  
  // Debug the graph
  GraphDebugger.printGraphStructure(graph);
  
  // Validate the graph
  const validation = GraphDebugger.validateGraph(graph);
  if (!validation.valid) {
    console.error("Graph validation failed:", validation.errors);
    return;
  }
  
  // Compile the graph
  const inputShapes = new Map([["input", [32, 784]]]);
  const result = await compiler.compile(graph, inputShapes);
  
  console.log("Compilation successful!");
  console.log("Kernel code written to:", result.filename);
  console.log("Workspace required:", result.workspaceSize, "bytes");
  
  // Visualize memory plan
  const memoryViz = MemoryPlanVisualizer.visualizeMemoryPlan(
    compiler.getIntermediateTensors(),
    result.workspaceSize
  );
  console.log(memoryViz);
}
*/