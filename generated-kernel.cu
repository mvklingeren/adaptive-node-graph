
#include <cuda_runtime.h>
#include <math.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) do {     cudaError_t err = call;     if (err != cudaSuccess) {         fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,                 cudaGetErrorString(err));         exit(1);     } } while(0)


// Debug mode bounds checking (can be disabled for release builds)
#ifndef NDEBUG
#define TENSOR_BOUNDS_CHECK 1
#else
#define TENSOR_BOUNDS_CHECK 0
#endif

template<typename T>
struct Tensor {
  T* data;
  const int* shape;
  int dims;

  __device__ inline T& operator()(int i) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 1 || i < 0 || i >= shape[0]) {
      printf("Tensor bounds error: 1D access [%d] out of bounds [0, %d) for %dD tensor\n", i, shape[0], dims);
      // In device code, we can't throw exceptions, so we'll return a reference to the first element
      // This prevents crashes but indicates a programming error
    }
    #endif
    return data[i]; 
  }
  
  __device__ inline T& operator()(int i, int j) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 2 || i < 0 || i >= shape[0] || j < 0 || j >= shape[1]) {
      printf("Tensor bounds error: 2D access [%d,%d] out of bounds [0,%d)x[0,%d) for %dD tensor\n", 
             i, j, shape[0], shape[1], dims);
    }
    #endif
    return data[i * shape[1] + j]; 
  }
  
  __device__ inline T& operator()(int i, int j, int k) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 3 || i < 0 || i >= shape[0] || j < 0 || j >= shape[1] || k < 0 || k >= shape[2]) {
      printf("Tensor bounds error: 3D access [%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d) for %dD tensor\n", 
             i, j, k, shape[0], shape[1], shape[2], dims);
    }
    #endif
    return data[(i * shape[1] + j) * shape[2] + k]; 
  }
  
  __device__ inline T& operator()(int i, int j, int k, int l) { 
    #if TENSOR_BOUNDS_CHECK
    if (dims < 4 || i < 0 || i >= shape[0] || j < 0 || j >= shape[1] || 
        k < 0 || k >= shape[2] || l < 0 || l >= shape[3]) {
      printf("Tensor bounds error: 4D access [%d,%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d)x[0,%d) for %dD tensor\n", 
             i, j, k, l, shape[0], shape[1], shape[2], shape[3], dims);
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
// Kernel Definitions
// ======================================================

      /**
       * @cuda global
       */
      __global__ void dense_forward(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights, 
        Tensor<float> bias
      ) {
        // Each thread computes one output element.
        // Grid: (output_features / threads_per_block, batch_size)
        // Block: (threads_per_block)
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < output.shape[1]) {
          float sum = 0.0f;
          for (int k = 0; k < input.shape[1]; ++k) {
            sum += input(batch_idx, k) * weights(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias(output_feature_idx);
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void relu_forward(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = 1;
        for (int i = 0; i < input.dims; ++i) {
          size *= input.shape[i];
        }
        if (idx < size) {
          output(idx) = fmaxf(0.0f, input(idx));
        }
      }
    

// ======================================================
// Main Host-Side Execution Function
// ======================================================
extern "C" void executeGraph(
  float* input_data,
  const int* input_shape,
  int input_dims,
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
  char* workspace
) {
  // --- Variable Declarations ---
  const int intermediate_0_shape[] = {64, 256};
  const int intermediate_1_shape[] = {64, 256};

  // --- Tensor Struct Instantiation ---
  Tensor<float> intermediate_0_tensor = {(float*)(workspace + 0), intermediate_0_shape, 2};
  Tensor<float> intermediate_1_tensor = {(float*)(workspace + 65536), intermediate_1_shape, 2};
  Tensor<float> input = {(float*)input_data, input_shape, input_dims};
  Tensor<float> output = {output_data, output_shape, output_dims};
  Tensor<float> param_0_weights = {param_0_weights_data, param_0_weights_shape, param_0_weights_dims};
  Tensor<float> param_1_bias = {param_1_bias_data, param_1_bias_shape, param_1_bias_dims};
  Tensor<float> param_2_weights = {param_2_weights_data, param_2_weights_shape, param_2_weights_dims};
  Tensor<float> param_3_bias = {param_3_bias_data, param_3_bias_shape, param_3_bias_dims};

  // --- Kernel Launch Sequence ---
  dense_forward<<<dim3(4, 64), dim3(64, 1, 1), 0>>>(intermediate_0_tensor, input, param_0_weights, param_1_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(64, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_1_tensor, intermediate_0_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 64), dim3(256, 1, 1), 0>>>(output, intermediate_1_tensor, param_2_weights, param_3_bias);
  CUDA_CHECK(cudaGetLastError());
  // --- End Execution Flow ---
}
    