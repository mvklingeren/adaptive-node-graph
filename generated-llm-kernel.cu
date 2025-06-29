
#include <cuda_runtime.h>
#include <math.h>


template<typename T>
struct Tensor {
  T* data;
  const int* shape;
  int dims;

  __device__ inline T& operator()(int i) {
    return data[i];
  }

  __device__ inline T& operator()(int i, int j) {
    return data[i * shape[1] + j];
  }

  __device__ inline T& operator()(int i, int j, int k) {
    return data[(i * shape[1] + j) * shape[2] + k];
  }
};


// ======================================================
// Node Device Functions
// ======================================================

      __device__ void embedding_forward(Tensor<float> output, Tensor<int> input, Tensor<float> embeddings) {
        int batch_idx = blockIdx.y;
        int seq_idx = blockIdx.x;
        int token_id = input(batch_idx, seq_idx);

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1]) {
          for (int i = 0; i < 128; ++i) {
            output(batch_idx, seq_idx, i) = embeddings(token_id, i);
          }
        }
      }
    


      __device__ void positional_encoding_forward(Tensor<float> output, Tensor<float> input) {
        // Placeholder for positional encoding
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = input.shape[0] * input.shape[1] * input.shape[2];
        if (idx < size) {
          output(idx) = input(idx); // Just pass through for now
        }
      }
    


      __device__ void dense_forward_0(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_0, 
        Tensor<float> bias_0
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_0(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_0(output_feature_idx);
        }
      }
    


      __device__ void dense_forward_1(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_1, 
        Tensor<float> bias_1
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_1(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_1(output_feature_idx);
        }
      }
    


      __device__ void dense_forward_2(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_2, 
        Tensor<float> bias_2
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_2(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_2(output_feature_idx);
        }
      }
    


      __device__ void split_heads(Tensor<float> output, Tensor<float> input) {
        // Input: [batch, seq_len, embed_dim]
        // Output: [batch, num_heads, seq_len, head_dim]
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int head_idx = blockIdx.x;
        int feature_idx = threadIdx.x;

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1] && head_idx < 4 && feature_idx < 32) {
          int input_idx = batch_idx * input.shape[1] * input.shape[2] + seq_idx * input.shape[2] + head_idx * 32 + feature_idx;
          int output_idx = batch_idx * 4 * input.shape[1] * 32 + head_idx * input.shape[1] * 32 + seq_idx * 32 + feature_idx;
          output(output_idx) = input(input_idx);
        }
      }
    


      __device__ void batched_matmul(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int batch_idx = blockIdx.z;
        int head_idx = blockIdx.y;
        int row = blockIdx.x;
        int col = threadIdx.x;

        if (batch_idx < a.shape[0] && head_idx < a.shape[1] && row < a.shape[2] && col < b.shape[3]) {
          float sum = 0.0f;
          for (int k = 0; k < a.shape[3]; ++k) {
            float b_val = b(batch_idx, head_idx, col, k);
            sum += a(batch_idx, head_idx, row, k) * b_val;
          }
          output(batch_idx, head_idx, row, col) = sum;
        }
      }
    


      __device__ void scale(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = 1;
        for (int i = 0; i < input.dims; ++i) {
          size *= input.shape[i];
        }
        if (idx < size) {
          output(idx) = input(idx) * 0.17677669529663687;
        }
      }
    


      __device__ void softmax_forward(Tensor<float> output, Tensor<float> input) {
        // Simplified Softmax for a single vector.
        int size = input.shape[input.dims - 1];

        // 1. Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < size; ++i) {
          if (input(i) > max_val) {
            max_val = input(i);
          }
        }

        // 2. Calculate sum of exps
        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
          sum_exp += expf(input(i) - max_val);
        }

        // 3. Calculate softmax
        for (int i = 0; i < size; ++i) {
          output(i) = expf(input(i) - max_val) / sum_exp;
        }
      }
    


      __device__ void batched_matmul(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int batch_idx = blockIdx.z;
        int head_idx = blockIdx.y;
        int row = blockIdx.x;
        int col = threadIdx.x;

        if (batch_idx < a.shape[0] && head_idx < a.shape[1] && row < a.shape[2] && col < b.shape[3]) {
          float sum = 0.0f;
          for (int k = 0; k < a.shape[3]; ++k) {
            float b_val = b(batch_idx, head_idx, k, col);
            sum += a(batch_idx, head_idx, row, k) * b_val;
          }
          output(batch_idx, head_idx, row, col) = sum;
        }
      }
    


      __device__ void concat_heads(Tensor<float> output, Tensor<float> input) {
        // Input: [batch, num_heads, seq_len, head_dim]
        // Output: [batch, seq_len, embed_dim]
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int head_idx = blockIdx.x;
        int feature_idx = threadIdx.x;

        if (batch_idx < output.shape[0] && seq_idx < output.shape[1] && head_idx < 4 && feature_idx < 32) {
          int output_idx = batch_idx * output.shape[1] * output.shape[2] + seq_idx * output.shape[2] + head_idx * 32 + feature_idx;
          int input_idx = batch_idx * 4 * output.shape[1] * 32 + head_idx * output.shape[1] * 32 + seq_idx * 32 + feature_idx;
          output(output_idx) = input(input_idx);
        }
      }
    


      __device__ void dense_forward_3(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_3, 
        Tensor<float> bias_3
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_3(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_3(output_feature_idx);
        }
      }
    


      __device__ void add_forward(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = 1;
        for (int i = 0; i < a.dims; ++i) {
          size *= a.shape[i];
        }
        if (idx < size) {
          output(idx) = a(idx) + b(idx);
        }
      }
    


      __device__ void layer_norm_forward_0(
        Tensor<float> output,
        Tensor<float> input,
        Tensor<float> gamma_0,
        Tensor<float> beta_0
      ) {
        // Simplified LayerNorm for a single feature vector.
        // A real implementation would handle batches and be more complex.
        int feature_count = input.shape[input.dims - 1];
        
        // 1. Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          mean += input(i);
        }
        mean /= feature_count;

        // 2. Calculate variance
        float variance = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          variance += (input(i) - mean) * (input(i) - mean);
        }
        variance /= feature_count;

        // 3. Normalize
        for (int i = 0; i < feature_count; ++i) {
          output(i) = (input(i) - mean) / sqrtf(variance + 0.00001) * gamma_0(i) + beta_0(i);
        }
      }
    


      __device__ void dense_forward_4(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_4, 
        Tensor<float> bias_4
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 512) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_4(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_4(output_feature_idx);
        }
      }
    


      __device__ void relu_forward(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = 1;
        for (int i = 0; i < input.dims; ++i) {
          size *= input.shape[i];
        }
        if (idx < size) {
          output(idx) = fmaxf(0.0f, input(idx));
        }
      }
    


      __device__ void dense_forward_5(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_5, 
        Tensor<float> bias_5
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 512; ++k) {
            sum += input(batch_idx, k) * weights_5(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_5(output_feature_idx);
        }
      }
    


      __device__ void layer_norm_forward_1(
        Tensor<float> output,
        Tensor<float> input,
        Tensor<float> gamma_1,
        Tensor<float> beta_1
      ) {
        // Simplified LayerNorm for a single feature vector.
        // A real implementation would handle batches and be more complex.
        int feature_count = input.shape[input.dims - 1];
        
        // 1. Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          mean += input(i);
        }
        mean /= feature_count;

        // 2. Calculate variance
        float variance = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          variance += (input(i) - mean) * (input(i) - mean);
        }
        variance /= feature_count;

        // 3. Normalize
        for (int i = 0; i < feature_count; ++i) {
          output(i) = (input(i) - mean) / sqrtf(variance + 0.00001) * gamma_1(i) + beta_1(i);
        }
      }
    


      __device__ void dense_forward_6(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_6, 
        Tensor<float> bias_6
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_6(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_6(output_feature_idx);
        }
      }
    


      __device__ void dense_forward_7(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_7, 
        Tensor<float> bias_7
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_7(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_7(output_feature_idx);
        }
      }
    


      __device__ void dense_forward_8(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_8, 
        Tensor<float> bias_8
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_8(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_8(output_feature_idx);
        }
      }
    


      __device__ void dense_forward_9(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_9, 
        Tensor<float> bias_9
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_9(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_9(output_feature_idx);
        }
      }
    


      __device__ void layer_norm_forward_2(
        Tensor<float> output,
        Tensor<float> input,
        Tensor<float> gamma_2,
        Tensor<float> beta_2
      ) {
        // Simplified LayerNorm for a single feature vector.
        // A real implementation would handle batches and be more complex.
        int feature_count = input.shape[input.dims - 1];
        
        // 1. Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          mean += input(i);
        }
        mean /= feature_count;

        // 2. Calculate variance
        float variance = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          variance += (input(i) - mean) * (input(i) - mean);
        }
        variance /= feature_count;

        // 3. Normalize
        for (int i = 0; i < feature_count; ++i) {
          output(i) = (input(i) - mean) / sqrtf(variance + 0.00001) * gamma_2(i) + beta_2(i);
        }
      }
    


      __device__ void dense_forward_10(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_10, 
        Tensor<float> bias_10
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 512) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_10(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_10(output_feature_idx);
        }
      }
    


      __device__ void dense_forward_11(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_11, 
        Tensor<float> bias_11
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 128) {
          float sum = 0.0f;
          for (int k = 0; k < 512; ++k) {
            sum += input(batch_idx, k) * weights_11(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_11(output_feature_idx);
        }
      }
    


      __device__ void layer_norm_forward_3(
        Tensor<float> output,
        Tensor<float> input,
        Tensor<float> gamma_3,
        Tensor<float> beta_3
      ) {
        // Simplified LayerNorm for a single feature vector.
        // A real implementation would handle batches and be more complex.
        int feature_count = input.shape[input.dims - 1];
        
        // 1. Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          mean += input(i);
        }
        mean /= feature_count;

        // 2. Calculate variance
        float variance = 0.0f;
        for (int i = 0; i < feature_count; ++i) {
          variance += (input(i) - mean) * (input(i) - mean);
        }
        variance /= feature_count;

        // 3. Normalize
        for (int i = 0; i < feature_count; ++i) {
          output(i) = (input(i) - mean) / sqrtf(variance + 0.00001) * gamma_3(i) + beta_3(i);
        }
      }
    


      __device__ void dense_forward_12(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights_12, 
        Tensor<float> bias_12
      ) {
        int batch_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && output_feature_idx < 1000) {
          float sum = 0.0f;
          for (int k = 0; k < 128; ++k) {
            sum += input(batch_idx, k) * weights_12(k, output_feature_idx);
          }
          output(batch_idx, output_feature_idx) = sum + bias_12(output_feature_idx);
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
  float* embeddings_data,
  const int* embeddings_shape,
  int embeddings_dims,
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
  float* weights_2_data,
  const int* weights_2_shape,
  int weights_2_dims,
  float* bias_2_data,
  const int* bias_2_shape,
  int bias_2_dims,
  float* weights_3_data,
  const int* weights_3_shape,
  int weights_3_dims,
  float* bias_3_data,
  const int* bias_3_shape,
  int bias_3_dims,
  float* gamma_0_data,
  const int* gamma_0_shape,
  int gamma_0_dims,
  float* beta_0_data,
  const int* beta_0_shape,
  int beta_0_dims,
  float* weights_4_data,
  const int* weights_4_shape,
  int weights_4_dims,
  float* bias_4_data,
  const int* bias_4_shape,
  int bias_4_dims,
  float* weights_5_data,
  const int* weights_5_shape,
  int weights_5_dims,
  float* bias_5_data,
  const int* bias_5_shape,
  int bias_5_dims,
  float* gamma_1_data,
  const int* gamma_1_shape,
  int gamma_1_dims,
  float* beta_1_data,
  const int* beta_1_shape,
  int beta_1_dims,
  float* weights_6_data,
  const int* weights_6_shape,
  int weights_6_dims,
  float* bias_6_data,
  const int* bias_6_shape,
  int bias_6_dims,
  float* weights_7_data,
  const int* weights_7_shape,
  int weights_7_dims,
  float* bias_7_data,
  const int* bias_7_shape,
  int bias_7_dims,
  float* weights_8_data,
  const int* weights_8_shape,
  int weights_8_dims,
  float* bias_8_data,
  const int* bias_8_shape,
  int bias_8_dims,
  float* weights_9_data,
  const int* weights_9_shape,
  int weights_9_dims,
  float* bias_9_data,
  const int* bias_9_shape,
  int bias_9_dims,
  float* gamma_2_data,
  const int* gamma_2_shape,
  int gamma_2_dims,
  float* beta_2_data,
  const int* beta_2_shape,
  int beta_2_dims,
  float* weights_10_data,
  const int* weights_10_shape,
  int weights_10_dims,
  float* bias_10_data,
  const int* bias_10_shape,
  int bias_10_dims,
  float* weights_11_data,
  const int* weights_11_shape,
  int weights_11_dims,
  float* bias_11_data,
  const int* bias_11_shape,
  int bias_11_dims,
  float* gamma_3_data,
  const int* gamma_3_shape,
  int gamma_3_dims,
  float* beta_3_data,
  const int* beta_3_shape,
  int beta_3_dims,
  float* weights_12_data,
  const int* weights_12_shape,
  int weights_12_dims,
  float* bias_12_data,
  const int* bias_12_shape,
  int bias_12_dims,
  char* workspace
) {
  // TODO: Implement tensor-based indexing
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // --- Variable Declarations ---
  const int intermediate_0_shape[] = {1, 256, 128};
  const int intermediate_1_shape[] = {1, 256, 128};
  const int intermediate_2_shape[] = {1, 128};
  const int intermediate_3_shape[] = {1, 128};
  const int intermediate_4_shape[] = {1, 128};
  const int intermediate_5_shape[] = {1, 4, 128, 32};
  const int intermediate_6_shape[] = {1, 4, 128, 32};
  const int intermediate_7_shape[] = {1, 4, 128, 32};
  const int intermediate_8_shape[] = {1, 4, 128, 128};
  const int intermediate_9_shape[] = {1, 4, 128, 128};
  const int intermediate_10_shape[] = {1, 4, 128, 128};
  const int intermediate_11_shape[] = {1, 4, 128, 32};
  const int intermediate_12_shape[] = {1, 128, 128};
  const int intermediate_13_shape[] = {1, 128};
  const int intermediate_14_shape[] = {1, 256, 128};
  const int intermediate_15_shape[] = {1, 256, 128};
  const int intermediate_16_shape[] = {1, 512};
  const int intermediate_17_shape[] = {1, 512};
  const int intermediate_18_shape[] = {1, 128};
  const int intermediate_19_shape[] = {1, 256, 128};
  const int intermediate_20_shape[] = {1, 256, 128};
  const int intermediate_21_shape[] = {1, 128};
  const int intermediate_22_shape[] = {1, 128};
  const int intermediate_23_shape[] = {1, 128};
  const int intermediate_24_shape[] = {1, 4, 128, 32};
  const int intermediate_25_shape[] = {1, 4, 128, 32};
  const int intermediate_26_shape[] = {1, 4, 128, 32};
  const int intermediate_27_shape[] = {1, 4, 128, 128};
  const int intermediate_28_shape[] = {1, 4, 128, 128};
  const int intermediate_29_shape[] = {1, 4, 128, 128};
  const int intermediate_30_shape[] = {1, 4, 128, 32};
  const int intermediate_31_shape[] = {1, 128, 128};
  const int intermediate_32_shape[] = {1, 128};
  const int intermediate_33_shape[] = {1, 256, 128};
  const int intermediate_34_shape[] = {1, 256, 128};
  const int intermediate_35_shape[] = {1, 512};
  const int intermediate_36_shape[] = {1, 512};
  const int intermediate_37_shape[] = {1, 128};
  const int intermediate_38_shape[] = {1, 256, 128};
  const int intermediate_39_shape[] = {1, 256, 128};

  // --- Tensor Struct Instantiation ---
  Tensor<float> intermediate_0_tensor = {(float*)(workspace + 0), intermediate_0_shape, 3};
  Tensor<float> intermediate_1_tensor = {(float*)(workspace + 0), intermediate_1_shape, 3};
  Tensor<float> intermediate_2_tensor = {(float*)(workspace + 131072), intermediate_2_shape, 2};
  Tensor<float> intermediate_3_tensor = {(float*)(workspace + 131584), intermediate_3_shape, 2};
  Tensor<float> intermediate_4_tensor = {(float*)(workspace + 132096), intermediate_4_shape, 2};
  Tensor<float> intermediate_5_tensor = {(float*)(workspace + 132608), intermediate_5_shape, 4};
  Tensor<float> intermediate_6_tensor = {(float*)(workspace + 198144), intermediate_6_shape, 4};
  Tensor<float> intermediate_7_tensor = {(float*)(workspace + 263680), intermediate_7_shape, 4};
  Tensor<float> intermediate_8_tensor = {(float*)(workspace + 329216), intermediate_8_shape, 4};
  Tensor<float> intermediate_9_tensor = {(float*)(workspace + 329216), intermediate_9_shape, 4};
  Tensor<float> intermediate_10_tensor = {(float*)(workspace + 329216), intermediate_10_shape, 4};
  Tensor<float> intermediate_11_tensor = {(float*)(workspace + 132608), intermediate_11_shape, 4};
  Tensor<float> intermediate_12_tensor = {(float*)(workspace + 132608), intermediate_12_shape, 3};
  Tensor<float> intermediate_13_tensor = {(float*)(workspace + 131072), intermediate_13_shape, 2};
  Tensor<float> intermediate_14_tensor = {(float*)(workspace + 0), intermediate_14_shape, 3};
  Tensor<float> intermediate_15_tensor = {(float*)(workspace + 0), intermediate_15_shape, 3};
  Tensor<float> intermediate_16_tensor = {(float*)(workspace + 132608), intermediate_16_shape, 2};
  Tensor<float> intermediate_17_tensor = {(float*)(workspace + 132608), intermediate_17_shape, 2};
  Tensor<float> intermediate_18_tensor = {(float*)(workspace + 131072), intermediate_18_shape, 2};
  Tensor<float> intermediate_19_tensor = {(float*)(workspace + 0), intermediate_19_shape, 3};
  Tensor<float> intermediate_20_tensor = {(float*)(workspace + 0), intermediate_20_shape, 3};
  Tensor<float> intermediate_21_tensor = {(float*)(workspace + 131072), intermediate_21_shape, 2};
  Tensor<float> intermediate_22_tensor = {(float*)(workspace + 131584), intermediate_22_shape, 2};
  Tensor<float> intermediate_23_tensor = {(float*)(workspace + 132096), intermediate_23_shape, 2};
  Tensor<float> intermediate_24_tensor = {(float*)(workspace + 132608), intermediate_24_shape, 4};
  Tensor<float> intermediate_25_tensor = {(float*)(workspace + 198144), intermediate_25_shape, 4};
  Tensor<float> intermediate_26_tensor = {(float*)(workspace + 263680), intermediate_26_shape, 4};
  Tensor<float> intermediate_27_tensor = {(float*)(workspace + 329216), intermediate_27_shape, 4};
  Tensor<float> intermediate_28_tensor = {(float*)(workspace + 329216), intermediate_28_shape, 4};
  Tensor<float> intermediate_29_tensor = {(float*)(workspace + 329216), intermediate_29_shape, 4};
  Tensor<float> intermediate_30_tensor = {(float*)(workspace + 132608), intermediate_30_shape, 4};
  Tensor<float> intermediate_31_tensor = {(float*)(workspace + 132608), intermediate_31_shape, 3};
  Tensor<float> intermediate_32_tensor = {(float*)(workspace + 131072), intermediate_32_shape, 2};
  Tensor<float> intermediate_33_tensor = {(float*)(workspace + 0), intermediate_33_shape, 3};
  Tensor<float> intermediate_34_tensor = {(float*)(workspace + 0), intermediate_34_shape, 3};
  Tensor<float> intermediate_35_tensor = {(float*)(workspace + 132608), intermediate_35_shape, 2};
  Tensor<float> intermediate_36_tensor = {(float*)(workspace + 132608), intermediate_36_shape, 2};
  Tensor<float> intermediate_37_tensor = {(float*)(workspace + 131072), intermediate_37_shape, 2};
  Tensor<float> intermediate_38_tensor = {(float*)(workspace + 0), intermediate_38_shape, 3};
  Tensor<float> intermediate_39_tensor = {(float*)(workspace + 0), intermediate_39_shape, 3};
  Tensor<float> input = {input_data, input_shape, input_dims};
  Tensor<float> output = {output_data, output_shape, output_dims};
  Tensor<float> embeddings = {embeddings_data, embeddings_shape, embeddings_dims};
  Tensor<float> weights_0 = {weights_0_data, weights_0_shape, weights_0_dims};
  Tensor<float> bias_0 = {bias_0_data, bias_0_shape, bias_0_dims};
  Tensor<float> weights_1 = {weights_1_data, weights_1_shape, weights_1_dims};
  Tensor<float> bias_1 = {bias_1_data, bias_1_shape, bias_1_dims};
  Tensor<float> weights_2 = {weights_2_data, weights_2_shape, weights_2_dims};
  Tensor<float> bias_2 = {bias_2_data, bias_2_shape, bias_2_dims};
  Tensor<float> weights_3 = {weights_3_data, weights_3_shape, weights_3_dims};
  Tensor<float> bias_3 = {bias_3_data, bias_3_shape, bias_3_dims};
  Tensor<float> gamma_0 = {gamma_0_data, gamma_0_shape, gamma_0_dims};
  Tensor<float> beta_0 = {beta_0_data, beta_0_shape, beta_0_dims};
  Tensor<float> weights_4 = {weights_4_data, weights_4_shape, weights_4_dims};
  Tensor<float> bias_4 = {bias_4_data, bias_4_shape, bias_4_dims};
  Tensor<float> weights_5 = {weights_5_data, weights_5_shape, weights_5_dims};
  Tensor<float> bias_5 = {bias_5_data, bias_5_shape, bias_5_dims};
  Tensor<float> gamma_1 = {gamma_1_data, gamma_1_shape, gamma_1_dims};
  Tensor<float> beta_1 = {beta_1_data, beta_1_shape, beta_1_dims};
  Tensor<float> weights_6 = {weights_6_data, weights_6_shape, weights_6_dims};
  Tensor<float> bias_6 = {bias_6_data, bias_6_shape, bias_6_dims};
  Tensor<float> weights_7 = {weights_7_data, weights_7_shape, weights_7_dims};
  Tensor<float> bias_7 = {bias_7_data, bias_7_shape, bias_7_dims};
  Tensor<float> weights_8 = {weights_8_data, weights_8_shape, weights_8_dims};
  Tensor<float> bias_8 = {bias_8_data, bias_8_shape, bias_8_dims};
  Tensor<float> weights_9 = {weights_9_data, weights_9_shape, weights_9_dims};
  Tensor<float> bias_9 = {bias_9_data, bias_9_shape, bias_9_dims};
  Tensor<float> gamma_2 = {gamma_2_data, gamma_2_shape, gamma_2_dims};
  Tensor<float> beta_2 = {beta_2_data, beta_2_shape, beta_2_dims};
  Tensor<float> weights_10 = {weights_10_data, weights_10_shape, weights_10_dims};
  Tensor<float> bias_10 = {bias_10_data, bias_10_shape, bias_10_dims};
  Tensor<float> weights_11 = {weights_11_data, weights_11_shape, weights_11_dims};
  Tensor<float> bias_11 = {bias_11_data, bias_11_shape, bias_11_dims};
  Tensor<float> gamma_3 = {gamma_3_data, gamma_3_shape, gamma_3_dims};
  Tensor<float> beta_3 = {beta_3_data, beta_3_shape, beta_3_dims};
  Tensor<float> weights_12 = {weights_12_data, weights_12_shape, weights_12_dims};
  Tensor<float> bias_12 = {bias_12_data, bias_12_shape, bias_12_dims};

  // --- Generated Execution Flow ---
    embedding_forward(intermediate_0_tensor, input, embeddings);
    positional_encoding_forward(intermediate_1_tensor, intermediate_0_tensor);
    dense_forward_0(intermediate_2_tensor, intermediate_1_tensor, weights_0, bias_0);
    dense_forward_1(intermediate_3_tensor, intermediate_1_tensor, weights_1, bias_1);
    dense_forward_2(intermediate_4_tensor, intermediate_1_tensor, weights_2, bias_2);
    split_heads(intermediate_5_tensor, intermediate_2_tensor);
    split_heads(intermediate_6_tensor, intermediate_3_tensor);
    split_heads(intermediate_7_tensor, intermediate_4_tensor);
    batched_matmul(intermediate_8_tensor, intermediate_5_tensor, intermediate_6_tensor);
    scale(intermediate_9_tensor, intermediate_8_tensor);
    softmax_forward(intermediate_10_tensor, intermediate_9_tensor);
    batched_matmul(intermediate_11_tensor, intermediate_10_tensor, intermediate_7_tensor);
    concat_heads(intermediate_12_tensor, intermediate_11_tensor);
    dense_forward_3(intermediate_13_tensor, intermediate_12_tensor, weights_3, bias_3);
    add_forward(intermediate_14_tensor, intermediate_1_tensor, intermediate_13_tensor);
    layer_norm_forward_0(intermediate_15_tensor, intermediate_14_tensor, gamma_0, beta_0);
    dense_forward_4(intermediate_16_tensor, intermediate_15_tensor, weights_4, bias_4);
    relu_forward(intermediate_17_tensor, intermediate_16_tensor);
    dense_forward_5(intermediate_18_tensor, intermediate_17_tensor, weights_5, bias_5);
    add_forward(intermediate_19_tensor, intermediate_15_tensor, intermediate_18_tensor);
    layer_norm_forward_1(intermediate_20_tensor, intermediate_19_tensor, gamma_1, beta_1);
    dense_forward_6(intermediate_21_tensor, intermediate_20_tensor, weights_6, bias_6);
    dense_forward_7(intermediate_22_tensor, intermediate_20_tensor, weights_7, bias_7);
    dense_forward_8(intermediate_23_tensor, intermediate_20_tensor, weights_8, bias_8);
    split_heads(intermediate_24_tensor, intermediate_21_tensor);
    split_heads(intermediate_25_tensor, intermediate_22_tensor);
    split_heads(intermediate_26_tensor, intermediate_23_tensor);
    batched_matmul(intermediate_27_tensor, intermediate_24_tensor, intermediate_25_tensor);
    scale(intermediate_28_tensor, intermediate_27_tensor);
    softmax_forward(intermediate_29_tensor, intermediate_28_tensor);
    batched_matmul(intermediate_30_tensor, intermediate_29_tensor, intermediate_26_tensor);
    concat_heads(intermediate_31_tensor, intermediate_30_tensor);
    dense_forward_9(intermediate_32_tensor, intermediate_31_tensor, weights_9, bias_9);
    add_forward(intermediate_33_tensor, intermediate_20_tensor, intermediate_32_tensor);
    layer_norm_forward_2(intermediate_34_tensor, intermediate_33_tensor, gamma_2, beta_2);
    dense_forward_10(intermediate_35_tensor, intermediate_34_tensor, weights_10, bias_10);
    relu_forward(intermediate_36_tensor, intermediate_35_tensor);
    dense_forward_11(intermediate_37_tensor, intermediate_36_tensor, weights_11, bias_11);
    add_forward(intermediate_38_tensor, intermediate_34_tensor, intermediate_37_tensor);
    layer_norm_forward_3(intermediate_39_tensor, intermediate_38_tensor, gamma_3, beta_3);
    dense_forward_12(output, intermediate_39_tensor, weights_12, bias_12);
  // --- End Execution Flow ---
}
    