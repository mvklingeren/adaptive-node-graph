
#include <cuda_runtime.h>
#include <math.h>

// ======================================================
// Node Device Functions
// ======================================================

      __device__ void dense_forward_0(float* out, float in, const float* weights_0, const float* bias_0) {
        // Simple multiplication, not a dot product, for this example.
        *out = in * weights_0[0] + bias_0[0];
      }
    


      __device__ void relu_forward(float* out, float in) {
        *out = fmaxf(0.0f, in);
      }
    


      __device__ void dense_forward_1(float* out, float in, const float* weights_1, const float* bias_1) {
        // Simple multiplication, not a dot product, for this example.
        *out = in * weights_1[0] + bias_1[0];
      }
    

// ======================================================
// Main Fused Graph Kernel
// ======================================================
extern "C" __global__ void executeGraph(
  const float* input,
  float* output,
  const float* weights_0,
  const float* bias_0,
  const float* weights_1,
  const float* bias_1,
  int n
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // --- Variable Declarations ---
  float dense_forward_0_output_cuda;
  float relu_forward_output_cuda;
  float dense_forward_1_output_cuda;

  // --- Generated Execution Flow ---
  // Node: dense_forward_0
  dense_forward_0(&dense_forward_0_output_cuda, input[idx], weights_0, bias_0);
  // Node: relu_forward
  relu_forward(&relu_forward_output_cuda, dense_forward_0_output_cuda);
  // Node: dense_forward_1
  dense_forward_1(&dense_forward_1_output_cuda, relu_forward_output_cuda, weights_1, bias_1);
  output[idx] = dense_forward_1_output_cuda;
  // --- End Execution Flow ---
}
    