export function generateKernelCode(params: {
  config: { memoryAlignment: number };
  kernelDefinitions: string;
  workspaceSize: number;
  paramNames: string[];
  variableDeclarations: string[];
  tensorInstantiations: string[];
  executionCalls: string[];
  graph: any; // CudaGraph type
  paramNameMapping: Map<string, string>;
  generateParameterShapeDeclarations: (graph: any, paramNames: string[], paramNameMapping: Map<string, string>) => string;
  generateParameterDims: (graph: any, paramNames: string[], paramNameMapping: Map<string, string>) => number[];
}) {
  const {
    config,
    kernelDefinitions,
    workspaceSize,
    paramNames,
    variableDeclarations,
    tensorInstantiations,
    executionCalls,
    graph,
    paramNameMapping,
    generateParameterShapeDeclarations,
    generateParameterDims,
  } = params;

  const tensorStruct = `
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

  return `
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
        const size_t alignment = ${config.memoryAlignment};
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
    
    ${generateParameterShapeDeclarations(graph, paramNames, paramNameMapping)}
    
    const int* param_shapes[${paramNames.length}] = {
        ${paramNames.map((_, i) => `param_shape_${i}`).join(",\n        ")}
    };
    
    const int param_dims[${paramNames.length}] = {
        ${generateParameterDims(graph, paramNames, paramNameMapping).join(", ")}
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
}
