
#include <cuda_runtime.h>
#include <math.h>


template<typename T>
struct Tensor {
  T* data;
  const int* shape;
  int dims;

  __device__ inline T& operator()(int i) { return data[i]; }
  __device__ inline T& operator()(int i, int j) { return data[i * shape[1] + j]; }
  __device__ inline T& operator()(int i, int j, int k) { return data[(i * shape[1] + j) * shape[2] + k]; }
};


// ======================================================
// Kernel Definitions
// ======================================================

      /**
       * @cuda global
       */
      void dense_forward(
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
  float* weights_data,
  const int* weights_shape,
  int weights_dims,
  float* bias_data,
  const int* bias_shape,
  int bias_dims,
  char* workspace
) {
  // --- Variable Declarations ---
  const int intermediate_0_shape[] = {64, 256};
  const int intermediate_1_shape[] = {64, 256};

  // --- Tensor Struct Instantiation ---
  Tensor<float> intermediate_0_tensor = {(float*)(workspace + 0), intermediate_0_shape, 2};
  Tensor<float> intermediate_1_tensor = {(float*)(workspace + 0), intermediate_1_shape, 2};
  Tensor<float> input = {input_data, input_shape, input_dims};
  Tensor<float> output = {output_data, output_shape, output_dims};
  Tensor<float> weights = {weights_data, weights_shape, weights_dims};
  Tensor<float> bias = {bias_data, bias_shape, bias_dims};

  // --- Kernel Launch Sequence ---
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(intermediate_0_tensor, input, weights, bias);
  relu_forward<<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(intermediate_1_tensor, intermediate_0_tensor);
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(output, intermediate_1_tensor, weights, bias);
  // --- End Execution Flow ---
}
    