
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) do {     cudaError_t err = call;     if (err != cudaSuccess) {         fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,                 cudaGetErrorString(err));         exit(1);     } } while(0)


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
      printf("FATAL: Tensor bounds error: 1D access [%d] out of bounds [0, %d) for %dD tensor\n", i, shape[0], dims);
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
      printf("FATAL: Tensor bounds error: 2D access [%d,%d] out of bounds [0,%d)x[0,%d) for %dD tensor\n", 
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
      printf("FATAL: Tensor bounds error: 3D access [%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d) for %dD tensor\n", 
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
      printf("FATAL: Tensor bounds error: 4D access [%d,%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d)x[0,%d) for %dD tensor\n", 
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
            fprintf(stderr, "ERROR: Workspace allocator out of memory. Requested: %zu bytes (aligned: %zu), Available: %zu bytes\n", 
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

      /**
       * @cuda global
       * Performs a dense layer transformation on a 2D tensor.
       * Input: [batch, input_features]
       * Output: [batch, output_features]
       */
      __global__ void dense_forward_2d(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights, 
        Tensor<float> bias
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < output.shape[1]) {
          float sum = 0.0f;
          for (int k = 0; k < input.shape[1]; ++k) { // Iterate over input_features
            sum += input(batch_idx, k) * weights(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias(output_feature_idx);
        }
      }
      


      /**
       * @cuda global
       * Optimized ReLU kernel that calculates size dynamically
       */
      __global__ void relu_forward(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = input.total_elements();
        if (idx < total_elements) {
          output.data[idx] = fmaxf(0.0f, input.data[idx]);
        }
      }
    

// ======================================================
// Main Host-Side Execution Function
// ======================================================
extern "C" void executeGraph(
  float* output_data,
  const int* output_shape,
  int output_dims,
  float* param_0_weights_data,
  const int* param_0_weights_shape,
  int param_0_weights_dims,
  float* param_1_bias_data,
  const int* param_1_bias_shape,
  int param_1_bias_dims,
  float* param_2_weights_data,
  const int* param_2_weights_shape,
  int param_2_weights_dims,
  float* param_3_bias_data,
  const int* param_3_bias_shape,
  int param_3_bias_dims,
  char* workspace,
  size_t workspace_size
) {
  // --- Input Validation ---
  if (!workspace) {
    fprintf(stderr, "Error: Null workspace pointer passed to executeGraph\n");
    return;
  }
  
  // Note: We no longer check for exact workspace size since we're using dynamic allocation
  // The allocator will report if we run out of space
  
  // --- Initialize Dynamic Memory Allocator ---
  WorkspaceAllocator allocator(workspace, workspace_size);

  // --- Variable Declarations ---
  const int intermediate_0_shape[] = {64, 784};
  const int intermediate_1_shape[] = {64, 256};
  const int intermediate_2_shape[] = {64, 256};

  // --- Tensor Struct Instantiation ---
  float* intermediate_0_data = (float*)allocator.allocate(200704);
  if (!intermediate_0_data) { fprintf(stderr, "Failed to allocate memory for intermediate_0\n"); return; }
  Tensor<float> intermediate_0_tensor = {intermediate_0_data, intermediate_0_shape, 2};
  float* intermediate_1_data = (float*)allocator.allocate(65536);
  if (!intermediate_1_data) { fprintf(stderr, "Failed to allocate memory for intermediate_1\n"); return; }
  Tensor<float> intermediate_1_tensor = {intermediate_1_data, intermediate_1_shape, 2};
  float* intermediate_2_data = (float*)allocator.allocate(65536);
  if (!intermediate_2_data) { fprintf(stderr, "Failed to allocate memory for intermediate_2\n"); return; }
  Tensor<float> intermediate_2_tensor = {intermediate_2_data, intermediate_2_shape, 2};
  Tensor<float> output = {output_data, output_shape, output_dims};
  Tensor<float> param_0_weights = {param_0_weights_data, param_0_weights_shape, param_0_weights_dims};
  Tensor<float> param_1_bias = {param_1_bias_data, param_1_bias_shape, param_1_bias_dims};
  Tensor<float> param_2_weights = {param_2_weights_data, param_2_weights_shape, param_2_weights_dims};
  Tensor<float> param_3_bias = {param_3_bias_data, param_3_bias_shape, param_3_bias_dims};

  // --- Kernel Launch Sequence ---
  dense_forward_2d<<<dim3(1, 64), dim3(256, 1, 1), 0>>>(intermediate_1_tensor, intermediate_0_tensor, param_0_weights, param_1_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(64, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_2_tensor, intermediate_1_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(1, 64), dim3(256, 1, 1), 0>>>(output, intermediate_2_tensor, param_2_weights, param_3_bias);
  CUDA_CHECK(cudaGetLastError());
  
  // --- Synchronization for completion verification ---
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // --- Report memory usage ---
  size_t used_memory = allocator.get_used_size();
  if (used_memory > 0) {
    // Uncomment for debugging memory usage
    // printf("Workspace memory used: %zu bytes (%.2f MB)\n", used_memory, used_memory / (1024.0 * 1024.0));
  }
  
  // --- End Execution Flow ---
}
    