
#include <cuda_runtime.h>
#include <math.h>


template<typename T>
struct Tensor {
  T* data;
  const int* shape;
  int dims;
};


// ======================================================
// Node Device Functions
// ======================================================

      __device__ void dense_forward_0(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_0, 
        Tensor<float> bias_0
      ) {
        int input_size = input.shape[0];
        int output_size = output.shape[0];

        for (int i = 0; i < output_size; ++i) {
          float sum = 0.0f;
          for (int j = 0; j < input_size; ++j) {
            // W is row-major: W[i, j] is at index i * input_size + j
            sum += input.data[j] * weights_0.data[i * input_size + j];
          }
          output.data[i] = sum + bias_0.data[i];
        }
      }
    


      __device__ void relu_forward(Tensor<float> output, Tensor<float> input) {
        int size = 1;
        for (int i = 0; i < input.dims; ++i) {
          size *= input.shape[i];
        }
        // This is a simplified loop for element-wise operation.
        // A real implementation would use the thread index (idx).
        for (int i = 0; i < size; ++i) {
            output.data[i] = fmaxf(0.0f, input.data[i]);
        }
      }
    


      __device__ void dense_forward_1(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_1, 
        Tensor<float> bias_1
      ) {
        int input_size = input.shape[0];
        int output_size = output.shape[0];

        for (int i = 0; i < output_size; ++i) {
          float sum = 0.0f;
          for (int j = 0; j < input_size; ++j) {
            // W is row-major: W[i, j] is at index i * input_size + j
            sum += input.data[j] * weights_1.data[i * input_size + j];
          }
          output.data[i] = sum + bias_1.data[i];
        }
      }
    

// ======================================================
// Main Fused Graph Kernel
// ======================================================
extern "C" __global__ void executeGraph(
  float* input_data,
  const int* input_shape,
  int input_dims,
  float* output_data,
  const int* output_shape,
  int output_dims,
  float* weights_0_data,
  const int* weights_0_shape,
  int weights_0_dims,
  float* bias_0_data,
  const int* bias_0_shape,
  int bias_0_dims,
  float* weights_1_data,
  const int* weights_1_shape,
  int weights_1_dims,
  float* bias_1_data,
  const int* bias_1_shape,
  int bias_1_dims,
  char* workspace
) {
  // TODO: Implement tensor-based indexing
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // --- Variable Declarations ---
  const int intermediate_0_shape[] = {256};
  const int intermediate_1_shape[] = {-1, -1};

  // --- Tensor Struct Instantiation ---
  Tensor<float> intermediate_0_tensor = {(float*)(workspace + 0), intermediate_0_shape, 1};
  Tensor<float> intermediate_1_tensor = {(float*)(workspace + 0), intermediate_1_shape, 2};
  Tensor<float> input = {input_data, input_shape, input_dims};
  Tensor<float> output = {output_data, output_shape, output_dims};
  Tensor<float> weights_0 = {weights_0_data, weights_0_shape, weights_0_dims};
  Tensor<float> bias_0 = {bias_0_data, bias_0_shape, bias_0_dims};
  Tensor<float> weights_1 = {weights_1_data, weights_1_shape, weights_1_dims};
  Tensor<float> bias_1 = {bias_1_data, bias_1_shape, bias_1_dims};

  // --- Generated Execution Flow ---
    dense_forward_0(intermediate_0_tensor, input, weights_0, bias_0);
    relu_forward(intermediate_1_tensor, intermediate_0_tensor);
    dense_forward_1(output, intermediate_1_tensor, weights_1, bias_1);
  // --- End Execution Flow ---
}
    