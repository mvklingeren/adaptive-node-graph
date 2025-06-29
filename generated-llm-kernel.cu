
#include <cuda_runtime.h>
#include <math.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) do {     cudaError_t err = call;     if (err != cudaSuccess) {         fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,                 cudaGetErrorString(err));         exit(1);     } } while(0)


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
      __global__ void embedding_forward(Tensor<float> output, Tensor<int> input, Tensor<float> embeddings) {
        int batch_idx = blockIdx.y;
        int seq_idx = blockIdx.x;

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1]) {
          int token_id = input(batch_idx, seq_idx);
          for (int i = threadIdx.x; i < output.shape[2]; i += blockDim.x) {
            output(batch_idx, seq_idx, i) = embeddings(token_id, i);
          }
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void positional_encoding_forward(Tensor<float> output, Tensor<float> input) {
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1] && embed_idx < input.shape[2]) {
          float pos = (float)seq_idx;
          float i = (float)embed_idx;
          float val;
          if (embed_idx % 2 == 0) {
            val = sinf(pos / powf(10000.0f, (2.0f * i) / (float)input.shape[2]));
          } else {
            val = cosf(pos / powf(10000.0f, (2.0f * i) / (float)input.shape[2]));
          }
          output(batch_idx, seq_idx, embed_idx) = input(batch_idx, seq_idx, embed_idx) + val;
        }
      }
    


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
      __global__ void split_heads_forward(Tensor<float> output, Tensor<float> input) {
        // Input: [batch, seq_len, embed_dim]
        // Output: [batch, num_heads, seq_len, head_dim]
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int head_idx = blockIdx.x;
        int feature_idx = threadIdx.x;

        int num_heads = output.shape[1];
        int head_dim = output.shape[3];

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1] && head_idx < num_heads && feature_idx < head_dim) {
          int embed_idx = head_idx * head_dim + feature_idx;
          output(batch_idx, head_idx, seq_idx, feature_idx) = input(batch_idx, seq_idx, embed_idx);
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void batched_matmul_transpose_b(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int batch_idx = blockIdx.z;
        int head_idx = blockIdx.y;
        int row = blockIdx.x;
        int col = threadIdx.x;

        if (batch_idx < a.shape[0] && head_idx < a.shape[1] && row < a.shape[2] && col < output.shape[3]) {
          float sum = 0.0f;
          for (int k = 0; k < a.shape[3]; ++k) {
            sum += a(batch_idx, head_idx, row, k) * b(batch_idx, head_idx, col, k);
          }
          output(batch_idx, head_idx, row, col) = sum;
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void scale_forward(Tensor<float> output, Tensor<float> input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3];
        
        for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
            output.data[i] = input.data[i] * 0.17677669529663687;
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void softmax_forward(Tensor<float> output, Tensor<float> input) {
        // This kernel computes softmax over the last dimension.
        // It handles both 2D tensors [batch, features] and 4D tensors [batch, heads, seq, seq]
        extern __shared__ float shared_mem[];
        int tid = threadIdx.x;
        
        if (input.dims == 2) {
          // Handle 2D case: [batch, features]
          int batch_idx = blockIdx.x;
          int size = input.shape[1];
          
          // 1. Find max for numerical stability
          float max_val = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              max_val = fmaxf(max_val, input(batch_idx, i));
          }
          shared_mem[tid] = max_val;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]); }
              __syncthreads();
          }
          max_val = shared_mem[0];

          // 2. Calculate sum of exps
          float sum_exp = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              sum_exp += expf(input(batch_idx, i) - max_val);
          }
          shared_mem[tid] = sum_exp;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] += shared_mem[tid + s]; }
              __syncthreads();
          }
          sum_exp = shared_mem[0];

          // 3. Calculate softmax
          for (int i = tid; i < size; i += blockDim.x) {
              output(batch_idx, i) = expf(input(batch_idx, i) - max_val) / sum_exp;
          }
        } else {
          // Handle 4D case: [batch, heads, seq, seq]
          int batch_idx = blockIdx.z;
          int head_idx = blockIdx.y;
          int row_idx = blockIdx.x;
          int size = input.shape[3];

          // 1. Find max for numerical stability
          float max_val = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              max_val = fmaxf(max_val, input(batch_idx, head_idx, row_idx, i));
          }
          shared_mem[tid] = max_val;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]); }
              __syncthreads();
          }
          max_val = shared_mem[0];

          // 2. Calculate sum of exps
          float sum_exp = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              sum_exp += expf(input(batch_idx, head_idx, row_idx, i) - max_val);
          }
          shared_mem[tid] = sum_exp;
          __syncthreads();
          for (int s = blockDim.x / 2; s > 0; s >>= 1) {
              if (tid < s) { shared_mem[tid] += shared_mem[tid + s]; }
              __syncthreads();
          }
          sum_exp = shared_mem[0];

          // 3. Calculate softmax
          for (int i = tid; i < size; i += blockDim.x) {
              output(batch_idx, head_idx, row_idx, i) = expf(input(batch_idx, head_idx, row_idx, i) - max_val) / sum_exp;
          }
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void batched_matmul(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int batch_idx = blockIdx.z;
        int head_idx = blockIdx.y;
        int row = blockIdx.x;
        int col = threadIdx.x;

        if (batch_idx < a.shape[0] && head_idx < a.shape[1] && row < a.shape[2] && col < output.shape[3]) {
          float sum = 0.0f;
          for (int k = 0; k < a.shape[3]; ++k) {
            sum += a(batch_idx, head_idx, row, k) * b(batch_idx, head_idx, k, col);
          }
          output(batch_idx, head_idx, row, col) = sum;
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void concat_heads_forward(Tensor<float> output, Tensor<float> input) {
        // Input: [batch, num_heads, seq_len, head_dim]
        // Output: [batch, seq_len, embed_dim]
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int head_idx = blockIdx.x;
        int feature_idx = threadIdx.x;

        int num_heads = input.shape[1];
        int head_dim = input.shape[3];

        if (batch_idx < output.shape[0] && seq_idx < output.shape[1] && head_idx < num_heads && feature_idx < head_dim) {
          int embed_idx = head_idx * head_dim + feature_idx;
          output(batch_idx, seq_idx, embed_idx) = input(batch_idx, head_idx, seq_idx, feature_idx);
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void add_forward(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // This is a simplified approach. A robust implementation would handle
        // arbitrary dimensions and calculate total size on the host.
        int size = a.shape[0] * a.shape[1] * a.shape[2];

        for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
            // This assumes flattened indexing. A better way is to reconstruct multidim index.
            // For now, we'll assume the shapes are identical and can be treated as flat arrays.
            output.data[i] = a.data[i] + b.data[i];
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void layer_norm_forward(
        Tensor<float> output,
        Tensor<float> input,
        Tensor<float> gamma,
        Tensor<float> beta
      ) {
        // This kernel processes one feature vector (e.g., one token's embedding) per block.
        // Grid: (batch_size, seq_len)
        // Block: (feature_count)
        extern __shared__ float shared_mem[];
        int batch_idx = blockIdx.y;
        int seq_idx = blockIdx.x;
        int feature_count = input.shape[2];
        int tid = threadIdx.x;

        // Step 1: Calculate mean
        float sum = 0.0f;
        for (int i = tid; i < feature_count; i += blockDim.x) {
            sum += input(batch_idx, seq_idx, i);
        }
        shared_mem[tid] = sum;
        __syncthreads();

        // Parallel reduction for mean
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_mem[tid] += shared_mem[tid + s];
            }
            __syncthreads();
        }
        float mean = shared_mem[0] / feature_count;

        // Step 2: Calculate variance
        sum = 0.0f;
        for (int i = tid; i < feature_count; i += blockDim.x) {
            float dev = input(batch_idx, seq_idx, i) - mean;
            sum += dev * dev;
        }
        shared_mem[tid] = sum;
        __syncthreads();
        
        // Parallel reduction for variance
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_mem[tid] += shared_mem[tid + s];
            }
            __syncthreads();
        }
        float variance = shared_mem[0] / feature_count;
        float rsqrt_variance = rsqrtf(variance + 0.00001);

        // Step 3: Normalize
        for (int i = tid; i < feature_count; i += blockDim.x) {
            float normalized = (input(batch_idx, seq_idx, i) - mean) * rsqrt_variance;
            output(batch_idx, seq_idx, i) = normalized * gamma(i) + beta(i);
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
  float* embeddings_data,
  const int* embeddings_shape,
  int embeddings_dims,
  float* weights_data,
  const int* weights_shape,
  int weights_dims,
  float* bias_data,
  const int* bias_shape,
  int bias_dims,
  float* gamma_data,
  const int* gamma_shape,
  int gamma_dims,
  float* beta_data,
  const int* beta_shape,
  int beta_dims,
  char* workspace
) {
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
  const int intermediate_40_shape[] = {1, 1000};

  // --- Tensor Struct Instantiation ---
  Tensor<float> intermediate_0_tensor = {(float*)(workspace + 0), intermediate_0_shape, 3};
  Tensor<float> intermediate_1_tensor = {(float*)(workspace + 131072), intermediate_1_shape, 3};
  Tensor<float> intermediate_2_tensor = {(float*)(workspace + 0), intermediate_2_shape, 2};
  Tensor<float> intermediate_3_tensor = {(float*)(workspace + 262144), intermediate_3_shape, 2};
  Tensor<float> intermediate_4_tensor = {(float*)(workspace + 262656), intermediate_4_shape, 2};
  Tensor<float> intermediate_5_tensor = {(float*)(workspace + 263168), intermediate_5_shape, 4};
  Tensor<float> intermediate_6_tensor = {(float*)(workspace + 0), intermediate_6_shape, 4};
  Tensor<float> intermediate_7_tensor = {(float*)(workspace + 328704), intermediate_7_shape, 4};
  Tensor<float> intermediate_8_tensor = {(float*)(workspace + 394240), intermediate_8_shape, 4};
  Tensor<float> intermediate_9_tensor = {(float*)(workspace + 656384), intermediate_9_shape, 4};
  Tensor<float> intermediate_10_tensor = {(float*)(workspace + 394240), intermediate_10_shape, 4};
  Tensor<float> intermediate_11_tensor = {(float*)(workspace + 0), intermediate_11_shape, 4};
  Tensor<float> intermediate_12_tensor = {(float*)(workspace + 263168), intermediate_12_shape, 3};
  Tensor<float> intermediate_13_tensor = {(float*)(workspace + 0), intermediate_13_shape, 2};
  Tensor<float> intermediate_14_tensor = {(float*)(workspace + 394240), intermediate_14_shape, 3};
  Tensor<float> intermediate_15_tensor = {(float*)(workspace + 0), intermediate_15_shape, 3};
  Tensor<float> intermediate_16_tensor = {(float*)(workspace + 131072), intermediate_16_shape, 2};
  Tensor<float> intermediate_17_tensor = {(float*)(workspace + 263168), intermediate_17_shape, 2};
  Tensor<float> intermediate_18_tensor = {(float*)(workspace + 131072), intermediate_18_shape, 2};
  Tensor<float> intermediate_19_tensor = {(float*)(workspace + 394240), intermediate_19_shape, 3};
  Tensor<float> intermediate_20_tensor = {(float*)(workspace + 0), intermediate_20_shape, 3};
  Tensor<float> intermediate_21_tensor = {(float*)(workspace + 131072), intermediate_21_shape, 2};
  Tensor<float> intermediate_22_tensor = {(float*)(workspace + 262144), intermediate_22_shape, 2};
  Tensor<float> intermediate_23_tensor = {(float*)(workspace + 262656), intermediate_23_shape, 2};
  Tensor<float> intermediate_24_tensor = {(float*)(workspace + 263168), intermediate_24_shape, 4};
  Tensor<float> intermediate_25_tensor = {(float*)(workspace + 131072), intermediate_25_shape, 4};
  Tensor<float> intermediate_26_tensor = {(float*)(workspace + 328704), intermediate_26_shape, 4};
  Tensor<float> intermediate_27_tensor = {(float*)(workspace + 394240), intermediate_27_shape, 4};
  Tensor<float> intermediate_28_tensor = {(float*)(workspace + 656384), intermediate_28_shape, 4};
  Tensor<float> intermediate_29_tensor = {(float*)(workspace + 394240), intermediate_29_shape, 4};
  Tensor<float> intermediate_30_tensor = {(float*)(workspace + 131072), intermediate_30_shape, 4};
  Tensor<float> intermediate_31_tensor = {(float*)(workspace + 263168), intermediate_31_shape, 3};
  Tensor<float> intermediate_32_tensor = {(float*)(workspace + 131072), intermediate_32_shape, 2};
  Tensor<float> intermediate_33_tensor = {(float*)(workspace + 394240), intermediate_33_shape, 3};
  Tensor<float> intermediate_34_tensor = {(float*)(workspace + 0), intermediate_34_shape, 3};
  Tensor<float> intermediate_35_tensor = {(float*)(workspace + 131072), intermediate_35_shape, 2};
  Tensor<float> intermediate_36_tensor = {(float*)(workspace + 263168), intermediate_36_shape, 2};
  Tensor<float> intermediate_37_tensor = {(float*)(workspace + 131072), intermediate_37_shape, 2};
  Tensor<float> intermediate_38_tensor = {(float*)(workspace + 394240), intermediate_38_shape, 3};
  Tensor<float> intermediate_39_tensor = {(float*)(workspace + 0), intermediate_39_shape, 3};
  Tensor<float> intermediate_40_tensor = {(float*)(workspace + 131072), intermediate_40_shape, 2};
  Tensor<float> input = {input_data, input_shape, input_dims};
  Tensor<float> output = {output_data, output_shape, output_dims};
  Tensor<float> embeddings = {embeddings_data, embeddings_shape, embeddings_dims};
  Tensor<float> weights = {weights_data, weights_shape, weights_dims};
  Tensor<float> bias = {bias_data, bias_shape, bias_dims};
  Tensor<float> gamma = {gamma_data, gamma_shape, gamma_dims};
  Tensor<float> beta = {beta_data, beta_shape, beta_dims};

  // --- Kernel Launch Sequence ---
  embedding_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_0_tensor, input, embeddings);
  CUDA_CHECK(cudaGetLastError());
  positional_encoding_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_1_tensor, intermediate_0_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_2_tensor, intermediate_1_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_3_tensor, intermediate_1_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_4_tensor, intermediate_1_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_5_tensor, intermediate_2_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_6_tensor, intermediate_3_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_7_tensor, intermediate_4_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_8_tensor, intermediate_5_tensor, intermediate_6_tensor);
  CUDA_CHECK(cudaGetLastError());
  scale_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_9_tensor, intermediate_8_tensor);
  CUDA_CHECK(cudaGetLastError());
  softmax_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 1024>>>(intermediate_10_tensor, intermediate_9_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_11_tensor, intermediate_10_tensor, intermediate_7_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_12_tensor, intermediate_11_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_13_tensor, intermediate_12_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_14_tensor, intermediate_1_tensor, intermediate_13_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 1024>>>(intermediate_15_tensor, intermediate_14_tensor, gamma, beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_16_tensor, intermediate_15_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_17_tensor, intermediate_16_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_18_tensor, intermediate_17_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_19_tensor, intermediate_15_tensor, intermediate_18_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 1024>>>(intermediate_20_tensor, intermediate_19_tensor, gamma, beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_21_tensor, intermediate_20_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_22_tensor, intermediate_20_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_23_tensor, intermediate_20_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_24_tensor, intermediate_21_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_25_tensor, intermediate_22_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_26_tensor, intermediate_23_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_27_tensor, intermediate_24_tensor, intermediate_25_tensor);
  CUDA_CHECK(cudaGetLastError());
  scale_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_28_tensor, intermediate_27_tensor);
  CUDA_CHECK(cudaGetLastError());
  softmax_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 1024>>>(intermediate_29_tensor, intermediate_28_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_30_tensor, intermediate_29_tensor, intermediate_26_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_31_tensor, intermediate_30_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_32_tensor, intermediate_31_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_33_tensor, intermediate_20_tensor, intermediate_32_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 1024>>>(intermediate_34_tensor, intermediate_33_tensor, gamma, beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_35_tensor, intermediate_34_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_36_tensor, intermediate_35_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_37_tensor, intermediate_36_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_38_tensor, intermediate_34_tensor, intermediate_37_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 1024>>>(intermediate_39_tensor, intermediate_38_tensor, gamma, beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_40_tensor, intermediate_39_tensor, weights, bias);
  CUDA_CHECK(cudaGetLastError());
  softmax_forward<<<dim3(1, 1, 1), dim3(256, 1, 1), 1024>>>(output, intermediate_40_tensor);
  CUDA_CHECK(cudaGetLastError());
  // --- End Execution Flow ---
}
    