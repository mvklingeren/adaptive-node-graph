
#include <cuda_runtime.h>
#include <math.h>

// ======================================================
// Node Device Functions
// ======================================================

      __device__ float dense_forward_0(float input, const float* weights_0, const float* bias_0) {
        // Simple multiplication, not a dot product, for this example.
        return input * weights_0[0] + bias_0[0];
      }
    


      __device__ float relu_forward(float x) {
        return fmaxf(0.0f, x);
      }
    


      __device__ float dense_forward_1(float input, const float* weights_1, const float* bias_1) {
        // Simple multiplication, not a dot product, for this example.
        return input * weights_1[0] + bias_1[0];
      }
    

// ======================================================
// Main Fused Graph Kernel
// ======================================================
extern "C" __global__ void executeGraph(
  const float* inputs,
  float* outputs,
  int n,
  const float* weights_0,
  const float* bias_0,
  const float* weights_1,
  const float* bias_1
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float value = inputs[idx];

  // --- Generated Execution Flow ---
  float dense_out_0 = dense_forward_0(value, weights_0, bias_0);
  float relu_out_0 = relu_forward(dense_out_0);
  float dense_out_1 = dense_forward_1(relu_out_0, weights_1, bias_1);
  // --- End Execution Flow ---

  outputs[idx] = dense_out_1;
}
    