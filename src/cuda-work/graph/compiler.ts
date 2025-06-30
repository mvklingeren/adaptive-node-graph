// ============================================================================
// CudaGraphCompiler: Compiles a CudaGraph into a set of kernels and a
// host-side execution function with improved parameter interface.
// ============================================================================

import { CudaRuntime, CudaTensor, CudaKernel } from "../cuda-abstractions.js";
import { CompilerConfig, CudaGraph, CudaNode } from "./core.js";

export class CudaGraphCompiler {
  private config: CompilerConfig;

  constructor(private runtime: CudaRuntime, config?: CompilerConfig) {
    this.config = {
      useArrayInterface: true,
      enableBoundsChecking: true, // Ensure this is true for debugging
      defaultBlockSize: 256,
      memoryAlignment: 256,
      enableMemoryPooling: true,
      verboseMemoryPlanning: true, // Enable verbose logging for memory planning
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
    const allParams = new Map<string, CudaTensor>();
    const paramNameMapping = new Map<string, string>();
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

        let globalParamName = null;
        for (const [existingName, existingTensor] of allParams.entries()) {
          if (existingTensor && existingTensor.id === tensor.id) {
            globalParamName = existingName;
            break;
          }
        }

        if (!globalParamName) {
          globalParamName = `param_${paramCounter++}_${localParamName}`;
          allParams.set(globalParamName, tensor);
        }

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
      if (node.deviceCode.trim() && !uniqueKernels.has(node.functionName)) {
        uniqueKernels.set(node.functionName, { code: node.deviceCode, node });
      }
    }
    const kernelDefinitions = Array.from(uniqueKernels.values())
      .map((k) => k.code)
      .join("\n\n");

    const executionCalls: string[] = [];
    for (const node of executionOrder) {
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

      const { gridDim, blockDim, sharedMemSize } =
        this.calculateOptimalGridBlock(node, outputTensors, inputTensors);

      executionCalls.push(
        `  ${node.functionName}<<<${gridDim}, ${blockDim}, ${sharedMemSize}>>>(${kernelCall});`
      );
      // Add CUDA_CHECK after each kernel launch for easier debugging
      executionCalls.push(`  CUDA_CHECK(cudaGetLastError());`);
    }

    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);

    const tensorInstantiations: string[] = [];
    intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
      const varName = `intermediate_${intermediateIdx}`;
      const shapeVar = `${varName}_shape`;
      const tensorVar = `${varName}_tensor`;

      if (tensorInfo.poolId?.startsWith("inplace_")) {
        const sourceKey = tensorInfo.poolId.replace("inplace_", "");
        const sourceTensor = intermediateTensors.get(sourceKey);
        if (sourceTensor && sourceTensor.offset >= 0) {
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate_at_offset(${sourceTensor.offset}, ${tensorInfo.size})); // In-place reuse of ${sourceKey}`
          );
          tensorInstantiations.push(
            `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
          );
        } else {
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate(${tensorInfo.size}));`
          );
          tensorInstantiations.push(
            `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
          );
        }
      } else {
        // Use calculated offset for proper memory reuse
        if (tensorInfo.offset >= 0) {
          tensorInstantiations.push(
            `  // Pool: ${tensorInfo.poolId || "default"}, Offset: ${tensorInfo.offset}, Size: ${tensorInfo.size} bytes`
          );
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate_at_offset(${tensorInfo.offset}, ${tensorInfo.size}));`
          );
        } else {
          // Fallback to sequential allocation
          tensorInstantiations.push(
            `  // Sequential allocation fallback - Size: ${tensorInfo.size} bytes`
          );
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate(${tensorInfo.size}));`
          );
        }
        tensorInstantiations.push(
          `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
        );
      }
      intermediateIdx++;
    }

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

#define CUDA_CHECK(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, \\
                cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
} while(0)

${tensorStruct}

class WorkspaceAllocator {
private:
    char* base_ptr;
    size_t current_offset;
    size_t total_size;
    
    inline size_t align_to_boundary(size_t size) const {
        const size_t alignment = ${this.config.memoryAlignment};
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
public:
    WorkspaceAllocator(char* workspace, size_t workspace_size) 
        : base_ptr(workspace), current_offset(0), total_size(workspace_size) {}
    
    // Sequential allocation (for fallback cases)
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
    
    // Direct offset allocation (for memory reuse)
    void* allocate_at_offset(size_t offset, size_t size) {
        size_t aligned_size = align_to_boundary(size);
        
        if (offset + aligned_size > total_size) {
            fprintf(stderr, "ERROR: Offset allocation out of bounds. Offset: %zu, Size: %zu (aligned: %zu), Total: %zu\\n", 
                    offset, size, aligned_size, total_size);
            return nullptr;
        }
        
        return base_ptr + offset;
    }
    
    size_t get_used_size() const {
        return current_offset;
    }
    
    void reset() {
        current_offset = 0;
    }
};

${kernelDefinitions}

extern "C" void executeGraph(
    void* input_data,
    const int* input_shape,
    int input_dims,
    const char* input_dtype,
    
    void* output_data,
    const int* output_shape,
    int output_dims,
    const char* output_dtype,
    
    void** param_data,
    const int** param_shapes,
    const int* param_dims,
    const char** param_dtypes,
    int param_count,
    
    char* workspace,
    size_t workspace_size
) {
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
    
    WorkspaceAllocator allocator(workspace, workspace_size);

${variableDeclarations.join("\n")}

    Tensor<float> param_tensors[${paramNames.length}];
    for (int i = 0; i < ${paramNames.length}; i++) {
        param_tensors[i] = {
            (float*)param_data[i],
            param_shapes[i],
            param_dims[i]
        };
    }

${tensorInstantiations.join("\n")}

${executionCalls.join("\n")}
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t used_memory = allocator.get_used_size();
    if (used_memory > 0) {
    }
}

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
    const int input_shape[] = {input_batch_size, input_seq_len};
    const int output_shape[] = {output_batch_size, output_vocab_size};
    
    ${this.generateParameterShapeDeclarations(graph, paramNames, paramNameMapping)}
    
    const int* param_shapes[${paramNames.length}] = {
        ${paramNames.map((_, i) => `param_shape_${i}`).join(",\n        ")}
    };
    
    const int param_dims[${paramNames.length}] = {
        ${this.generateParameterDims(graph, paramNames, paramNameMapping).join(", ")}
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
    throw new Error("Legacy kernel generation not implemented in this example");
  }

  private generateParameterShapeDeclarations(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): string {
    const declarations: string[] = [];

    paramNames.forEach((paramName, i) => {
      const tensor = this.findTensorByGlobalName(graph, paramName, paramNameMapping);
      if (tensor && tensor.shape) {
        declarations.push(
          `    const int param_shape_${i}[] = {${tensor.shape.join(
            ", "
          )}}; // ${paramName}`
        );
      } else {
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

    return declarations.join("\n");
  }

  private generateParameterDims(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): number[] {
    return paramNames.map((paramName) => {
      const tensor = this.findTensorByGlobalName(graph, paramName, paramNameMapping);
      return tensor ? tensor.shape.length : 1;
    });
  }

  private findTensorByGlobalName(
    graph: CudaGraph,
    globalName: string,
    paramNameMapping: Map<string, string>
  ): CudaTensor | null {
    for (const node of graph.nodes.values()) {
      for (const [localName, tensor] of node.parameters) {
        const nodeParamKey = `${node.id}:${localName}`;
        const mappedGlobalName = paramNameMapping.get(nodeParamKey);
        if (mappedGlobalName === globalName) {
          return tensor;
        }
      }
    }
    return null;
  }

  private getTensorStructDefinition(): string {
    const boundsCheck = this.config.enableBoundsChecking;
    return `
// Always enable for debugging purposes
#define TENSOR_BOUNDS_CHECK 1
#define TENSOR_BOUNDS_CHECK_VERBOSE 1

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
      asm("trap;");
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

    let batchSize = -1;
    for (const shape of inputShapes.values()) {
      if (shape.length > 0 && shape[0] !== -1) {
        batchSize = shape[0];
        break;
      }
    }

    if (batchSize === -1) {
      batchSize = 1;
    }

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

    let maxPasses = 3;
    for (let pass = 0; pass < maxPasses; pass++) {
      let anyShapeChanged = false;

      for (const node of executionOrder) {
        const oldOutputShapes = new Map();
        for (const [port, spec] of node.outputs) {
          oldOutputShapes.set(port, [...spec.shape]);
        }

        node.resolveShapes();

        for (const [port, spec] of node.outputs) {
          const oldShape = oldOutputShapes.get(port);
          if (
            !oldShape ||
            JSON.stringify(oldShape) !== JSON.stringify(spec.shape)
          ) {
            anyShapeChanged = true;
          }
        }

        for (const conn of graph.connections) {
          if (conn.fromNode.id === node.id) {
            const sourceSpec = node.outputs.get(conn.fromPort);
            if (sourceSpec) {
              conn.toNode.updateInputShape(conn.toPort, sourceSpec.shape);
            }
          }
        }
      }

      if (!anyShapeChanged) {
        break;
      }
    }

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
          const size = spec.shape.reduce((a, b) => a * b, 1) * 4;
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

    this.computeEnhancedLifetimes(graph, executionOrder, intermediateTensors);

    if (this.config.enableMemoryPooling) {
      this.detectInPlaceOperations(graph, executionOrder, intermediateTensors);
    }

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
    const nodeIndexMap = new Map<string, number>();
    executionOrder.forEach((node, i) => {
      nodeIndexMap.set(node.id, i);
    });

    for (const [tensorKey, tensorInfo] of intermediateTensors.entries()) {
      const [nodeId, port] = tensorKey.split(":");
      const producerIndex = nodeIndexMap.get(nodeId);
      if (producerIndex !== undefined) {
        tensorInfo.liveStart = producerIndex;
        tensorInfo.liveEnd = producerIndex;
      }
    }

    for (const [tensorKey, tensorInfo] of intermediateTensors.entries()) {
      for (const conn of graph.connections) {
        const sourceKey = `${conn.fromNode.id}:${conn.fromPort}`;
        if (sourceKey === tensorKey) {
          const targetNodeIndex = nodeIndexMap.get(conn.toNode.id);
          if (targetNodeIndex !== undefined) {
            tensorInfo.liveEnd = Math.max(tensorInfo.liveEnd, targetNodeIndex);
          }
        }
      }

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

    // Sort tensors by start time, then by size (largest first for better packing)
    const sortedTensors = Array.from(intermediateTensors.entries())
      .filter(([_, tensor]) => !tensor.poolId?.startsWith("inplace_"))
      .sort((a, b) => {
        if (a[1].liveStart !== b[1].liveStart) {
          return a[1].liveStart - b[1].liveStart;
        }
        return b[1].size - a[1].size; // Larger tensors first
      });

    // Track active allocations with their end times
    const activeAllocations: Array<{
      offset: number;
      size: number;
      endTime: number;
    }> = [];

    let maxOffset = 0;

    for (const [tensorKey, tensor] of sortedTensors) {
      const alignedSize = this.alignTo256Bytes(tensor.size);
      
      // Remove expired allocations
      const currentTime = tensor.liveStart;
      for (let i = activeAllocations.length - 1; i >= 0; i--) {
        if (activeAllocations[i].endTime < currentTime) {
          activeAllocations.splice(i, 1);
        }
      }
      
      // Find best fit offset
      let bestOffset = 0;
      let found = false;
      
      // Try to fit in gaps between active allocations
      activeAllocations.sort((a, b) => a.offset - b.offset);
      
      for (let i = 0; i <= activeAllocations.length; i++) {
        const startOffset = i === 0 ? 0 : activeAllocations[i - 1].offset + activeAllocations[i - 1].size;
        const endOffset = i === activeAllocations.length ? Infinity : activeAllocations[i].offset;
        
        if (endOffset - startOffset >= alignedSize) {
          bestOffset = startOffset;
          found = true;
          break;
        }
      }
      
      if (!found) {
        // Place at the end
        bestOffset = activeAllocations.length > 0 
          ? Math.max(...activeAllocations.map(a => a.offset + a.size))
          : 0;
      }
      
      // Align the offset
      bestOffset = Math.ceil(bestOffset / this.config.memoryAlignment!) * this.config.memoryAlignment!;
      
      tensor.offset = bestOffset;
      activeAllocations.push({
        offset: bestOffset,
        size: alignedSize,
        endTime: tensor.liveEnd
      });
      
      maxOffset = Math.max(maxOffset, bestOffset + alignedSize);
    }

    return { memoryPools, workspaceSize: maxOffset };
  }

  private categorizeTensorSize(size: number): string {
    const MB = 1024 * 1024;
    if (size < MB) return "small";
    if (size < 10 * MB) return "medium";
    if (size < 50 * MB) return "large";
    return "xlarge";
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
    const primaryOutput = Array.from(node.outputs.values())[0];
    if (!primaryOutput) {
      return {
        gridDim: "dim3(1, 1, 1)",
        blockDim: `dim3(${this.config.defaultBlockSize}, 1, 1)`,
        sharedMemSize: 0,
      };
    }

    const shape = primaryOutput.shape;

    switch (node.functionName) {
      case "embedding_forward":
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const blockSize = Math.min(embedDim, 1024);
          const alignedBlockSize = Math.ceil(blockSize / 32) * 32;
          return {
            gridDim: `dim3(${seqLen}, ${batchSize})`,
            blockDim: `dim3(${Math.min(alignedBlockSize, 1024)}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      case "positional_encoding_forward":
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const totalElements = batchSize * seqLen * embedDim;
          const blockSize = this.config.defaultBlockSize!;
          const gridSize = Math.ceil(totalElements / blockSize);
          return {
            gridDim: `dim3(${gridSize}, 1, 1)`,
            blockDim: `dim3(${blockSize}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      case "dense_forward_2d":
      case "cublas_dense_forward_2d":
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
      case "cublas_dense_forward_3d":
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
        if (shape.length >= 4) {
          const batchSize = shape[0];
          const numHeads = shape[1];
          const outputRows = shape[2];
          const outputCols = shape[3];

          const TILE_SIZE = 32;

          const tilesM = Math.ceil(outputRows / TILE_SIZE);
          const tilesN = Math.ceil(outputCols / TILE_SIZE);
          const totalTiles = tilesM * tilesN;

          const sharedMemSize = 2 * TILE_SIZE * (TILE_SIZE + 1) * 4;

          return {
            gridDim: `dim3(${totalTiles}, ${numHeads}, ${batchSize})`,
            blockDim: `dim3(${TILE_SIZE}, ${TILE_SIZE}, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      case "softmax_forward":
        if (shape.length === 2) {
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
        } else if (shape.length === 4) {
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
        const totalElements = shape.reduce((a, b) => a * b, 1);
        const blockSize = this.config.defaultBlockSize!;
        const gridSize = Math.ceil(totalElements / blockSize);
        return {
          gridDim: `dim3(${gridSize}, 1, 1)`,
          blockDim: `dim3(${blockSize}, 1, 1)`,
          sharedMemSize: 0,
        };

      case "concat_heads_forward":
        if (shape.length >= 3) {
          const batchSize = shape[0];
          const seqLen = shape[1];
          const embedDim = shape[2];
          const numHeads = 6;
          const headDim = embedDim / numHeads;
          return {
            gridDim: `dim3(${seqLen}, ${batchSize})`,
            blockDim: `dim3(${embedDim}, 1, 1)`,
            sharedMemSize: 0,
          };
        }
        break;

      default:
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
