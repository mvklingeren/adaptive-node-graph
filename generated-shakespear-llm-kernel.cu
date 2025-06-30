
#include <cuda_runtime.h>
#include <cfloat>
#include <float.h>
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
      __global__ void positional_encoding_forward(Tensor<float> output, Tensor<float> input, Tensor<float> frequencies) {
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int embed_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < input.shape[0] && seq_idx < input.shape[1] && embed_idx < input.shape[2]) {
          float pos = (float)seq_idx;
          int freq_idx = embed_idx / 2;  // Integer division to get frequency index
          
          // Use pre-computed frequency values for better accuracy
          // This eliminates floating-point accumulation errors from powf() calculations
          float freq = frequencies(freq_idx);
          float val;
          if (embed_idx % 2 == 0) {
            val = sinf(pos * freq);
          } else {
            val = cosf(pos * freq);
          }
          output(batch_idx, seq_idx, embed_idx) = input(batch_idx, seq_idx, embed_idx) + val;
        }
      }
    


      /**
       * @cuda global
       * Performs a dense layer transformation on a 3D tensor.
       * Input: [batch, seq_len, input_features]
       * Output: [batch, seq_len, output_features]
       */
      __global__ void dense_forward_3d(
        Tensor<float> output, 
        Tensor<float> input, 
        Tensor<float> weights, 
        Tensor<float> bias
      ) {
        int batch_idx = blockIdx.z;
        int seq_idx = blockIdx.y;
        int output_feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx < output.shape[0] && seq_idx < output.shape[1] && output_feature_idx < output.shape[2]) {
          float sum = 0.0f;
          for (int k = 0; k < input.shape[2]; ++k) { // Iterate over input_features
            sum += input(batch_idx, seq_idx, k) * weights(k, output_feature_idx);
          }
          output(batch_idx, seq_idx, output_feature_idx) = sum + bias(output_feature_idx);
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
      __global__ void batched_matmul_transpose_b_tiled(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        const int TILE_SIZE = 32;
        
        // Add +1 to avoid bank conflicts
        __shared__ float sA[TILE_SIZE][TILE_SIZE + 1];
        __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];
        
        int batch_idx = blockIdx.z;
        int head_idx = blockIdx.y;
        
        // Calculate block position in the output matrix
        int tiles_per_row = (output.shape[3] + TILE_SIZE - 1) / TILE_SIZE;
        int block_row = blockIdx.x / tiles_per_row;
        int block_col = blockIdx.x % tiles_per_row;
        
        // Global row and column for the output element
        int row = block_row * TILE_SIZE + threadIdx.y;
        int col = block_col * TILE_SIZE + threadIdx.x;
        
        float sum = 0.0f;
        
        // Determine K dimension based on transpose
        int k_dim = b.shape[2];
        int num_tiles_k = (k_dim + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int tile = 0; tile < num_tiles_k; ++tile) {
            // Load tile of A into shared memory
            int a_row = row;
            int a_col = tile * TILE_SIZE + threadIdx.x;
            
            if (batch_idx < a.shape[0] && head_idx < a.shape[1] && 
                a_row < a.shape[2] && a_col < a.shape[3]) {
                sA[threadIdx.y][threadIdx.x] = a(batch_idx, head_idx, a_row, a_col);
            } else {
                sA[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Load tile of B into shared memory
            int b_row = tile * TILE_SIZE + threadIdx.y;
            int b_col = col;
            
            if (batch_idx < b.shape[0] && head_idx < b.shape[1]) {
                
                // For transpose B: we want B[k, n] so swap indices
                if (b_row < b.shape[3] && b_col < b.shape[2]) {
                    sB[threadIdx.y][threadIdx.x] = b(batch_idx, head_idx, b_col, b_row);
                } else {
                    sB[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
            } else {
                sB[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        // Write result to global memory
        if (batch_idx < output.shape[0] && head_idx < output.shape[1] && 
            row < output.shape[2] && col < output.shape[3]) {
            output(batch_idx, head_idx, row, col) = sum;
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void fused_scale_softmax_forward(Tensor<float> output, Tensor<float> input) {
        // This kernel computes scale + softmax in a single pass over the last dimension.
        // It handles both 2D tensors [batch, features] and 4D tensors [batch, heads, seq, seq]
        extern __shared__ float shared_mem[];
        int tid = threadIdx.x;
        
        if (input.dims == 2) {
          // Handle 2D case: [batch, features]
          int batch_idx = blockIdx.x;
          int size = input.shape[1];
          
          // 1. Find max of scaled values for numerical stability
          float max_val_thread = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              float scaled_val = input(batch_idx, i) * 0.125f;
              max_val_thread = fmaxf(max_val_thread, scaled_val);
          }
          
          // Intra-warp reduction for max_val_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              max_val_thread = fmaxf(max_val_thread, __shfl_down_sync(0xFFFFFFFF, max_val_thread, offset));
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = max_val_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
                  }
                  __syncthreads();
              }
          }
          float max_val = shared_mem[0];

          // 2. Calculate sum of exp(scaled_value - max)
          float sum_exp_thread = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              float scaled_val = input(batch_idx, i) * 0.125f;
              sum_exp_thread += expf(scaled_val - max_val);
          }
          
          // Intra-warp reduction for sum_exp_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              sum_exp_thread += __shfl_down_sync(0xFFFFFFFF, sum_exp_thread, offset);
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = sum_exp_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
                  }
                  __syncthreads();
              }
          }
          float sum_exp = shared_mem[0];

          // 3. Calculate final softmax values
          for (int i = tid; i < size; i += blockDim.x) {
              float scaled_val = input(batch_idx, i) * 0.125f;
              output(batch_idx, i) = expf(scaled_val - max_val) / sum_exp;
          }
        } else {
          // Handle 4D case: [batch, heads, seq, seq]
          int batch_idx = blockIdx.z;
          int head_idx = blockIdx.y;
          int row_idx = blockIdx.x;
          int size = input.shape[3];

          // 1. Find max of scaled values for numerical stability
          float max_val_thread = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              float scaled_val = input(batch_idx, head_idx, row_idx, i) * 0.125f;
              max_val_thread = fmaxf(max_val_thread, scaled_val);
          }
          
          // Intra-warp reduction for max_val_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              max_val_thread = fmaxf(max_val_thread, __shfl_down_sync(0xFFFFFFFF, max_val_thread, offset));
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = max_val_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
                  }
                  __syncthreads();
              }
          }
          float max_val = shared_mem[0];

          // 2. Calculate sum of exp(scaled_value - max)
          float sum_exp_thread = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              float scaled_val = input(batch_idx, head_idx, row_idx, i) * 0.125f;
              sum_exp_thread += expf(scaled_val - max_val);
          }
          
          // Intra-warp reduction for sum_exp_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              sum_exp_thread += __shfl_down_sync(0xFFFFFFFF, sum_exp_thread, offset);
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = sum_exp_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
                  }
                  __syncthreads();
              }
          }
          float sum_exp = shared_mem[0];

          // 3. Calculate final softmax values
          for (int i = tid; i < size; i += blockDim.x) {
              float scaled_val = input(batch_idx, head_idx, row_idx, i) * 0.125f;
              output(batch_idx, head_idx, row_idx, i) = expf(scaled_val - max_val) / sum_exp;
          }
        }
      }
    


      /**
       * @cuda global
       */
      __global__ void batched_matmul_tiled(Tensor<float> output, Tensor<float> a, Tensor<float> b) {
        const int TILE_SIZE = 32;
        
        // Add +1 to avoid bank conflicts
        __shared__ float sA[TILE_SIZE][TILE_SIZE + 1];
        __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];
        
        int batch_idx = blockIdx.z;
        int head_idx = blockIdx.y;
        
        // Calculate block position in the output matrix
        int tiles_per_row = (output.shape[3] + TILE_SIZE - 1) / TILE_SIZE;
        int block_row = blockIdx.x / tiles_per_row;
        int block_col = blockIdx.x % tiles_per_row;
        
        // Global row and column for the output element
        int row = block_row * TILE_SIZE + threadIdx.y;
        int col = block_col * TILE_SIZE + threadIdx.x;
        
        float sum = 0.0f;
        
        // Determine K dimension based on transpose
        int k_dim = b.shape[3];
        int num_tiles_k = (k_dim + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int tile = 0; tile < num_tiles_k; ++tile) {
            // Load tile of A into shared memory
            int a_row = row;
            int a_col = tile * TILE_SIZE + threadIdx.x;
            
            if (batch_idx < a.shape[0] && head_idx < a.shape[1] && 
                a_row < a.shape[2] && a_col < a.shape[3]) {
                sA[threadIdx.y][threadIdx.x] = a(batch_idx, head_idx, a_row, a_col);
            } else {
                sA[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Load tile of B into shared memory
            int b_row = tile * TILE_SIZE + threadIdx.y;
            int b_col = col;
            
            if (batch_idx < b.shape[0] && head_idx < b.shape[1]) {
                
                // Normal B: B[k, n]
                if (b_row < b.shape[2] && b_col < b.shape[3]) {
                    sB[threadIdx.y][threadIdx.x] = b(batch_idx, head_idx, b_row, b_col);
                } else {
                    sB[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
            } else {
                sB[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        // Write result to global memory
        if (batch_idx < output.shape[0] && head_idx < output.shape[1] && 
            row < output.shape[2] && col < output.shape[3]) {
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
        int size = output.total_elements();

        if (idx < size) {
            int a_idx = idx % a.total_elements();
            int b_idx = idx % b.total_elements();
            output.data[idx] = a.data[a_idx] + b.data[b_idx];
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
          // 1. Find max for numerical stability
          float max_val_thread = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              max_val_thread = fmaxf(max_val_thread, input(batch_idx, i));
          }
          
          // Intra-warp reduction for max_val_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              max_val_thread = fmaxf(max_val_thread, __shfl_down_sync(0xFFFFFFFF, max_val_thread, offset));
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = max_val_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
                  }
                  __syncthreads();
              }
          }
          float max_val = shared_mem[0];

          // 2. Calculate sum of exps
          float sum_exp_thread = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              sum_exp_thread += expf(input(batch_idx, i) - max_val);
          }
          
          // Intra-warp reduction for sum_exp_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              sum_exp_thread += __shfl_down_sync(0xFFFFFFFF, sum_exp_thread, offset);
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = sum_exp_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
                  }
                  __syncthreads();
              }
          }
          float sum_exp = shared_mem[0];

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
          // 1. Find max for numerical stability
          float max_val_thread = -FLT_MAX;
          for (int i = tid; i < size; i += blockDim.x) {
              max_val_thread = fmaxf(max_val_thread, input(batch_idx, head_idx, row_idx, i));
          }
          
          // Intra-warp reduction for max_val_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              max_val_thread = fmaxf(max_val_thread, __shfl_down_sync(0xFFFFFFFF, max_val_thread, offset));
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = max_val_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
                  }
                  __syncthreads();
              }
          }
          float max_val = shared_mem[0];

          // 2. Calculate sum of exps
          float sum_exp_thread = 0.0f;
          for (int i = tid; i < size; i += blockDim.x) {
              sum_exp_thread += expf(input(batch_idx, head_idx, row_idx, i) - max_val);
          }
          
          // Intra-warp reduction for sum_exp_thread using __shfl_down_sync
          for (int offset = 16; offset > 0; offset /= 2) {
              sum_exp_thread += __shfl_down_sync(0xFFFFFFFF, sum_exp_thread, offset);
          }
          
          // Store warp-leader results to shared memory
          if (threadIdx.x % 32 == 0) {
              shared_mem[threadIdx.x / 32] = sum_exp_thread;
          }
          __syncthreads();

          // Inter-warp reduction using shared memory
          if (threadIdx.x < (blockDim.x + 31) / 32) {
              for (int s = ((blockDim.x + 31) / 32) / 2; s > 0; s >>= 1) {
                  if (threadIdx.x < s) {
                      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
                  }
                  __syncthreads();
              }
          }
          float sum_exp = shared_mem[0];

          // 3. Calculate softmax
          for (int i = tid; i < size; i += blockDim.x) {
              output(batch_idx, head_idx, row_idx, i) = expf(input(batch_idx, head_idx, row_idx, i) - max_val) / sum_exp;
          }
        }
      }
    

// ======================================================
// Main Host-Side Execution Function
// ======================================================
extern "C" void executeGraph(
 int* input_data,
  const int* input_shape,
  int input_dims,
  float* output_data,
  const int* output_shape,
  int output_dims,
  float* param_0_embeddings_data,
  const int* param_0_embeddings_shape,
  int param_0_embeddings_dims,
  float* param_1_frequencies_data,
  const int* param_1_frequencies_shape,
  int param_1_frequencies_dims,
  float* param_2_weights_data,
  const int* param_2_weights_shape,
  int param_2_weights_dims,
  float* param_3_bias_data,
  const int* param_3_bias_shape,
  int param_3_bias_dims,
  float* param_4_weights_data,
  const int* param_4_weights_shape,
  int param_4_weights_dims,
  float* param_5_bias_data,
  const int* param_5_bias_shape,
  int param_5_bias_dims,
  float* param_6_weights_data,
  const int* param_6_weights_shape,
  int param_6_weights_dims,
  float* param_7_bias_data,
  const int* param_7_bias_shape,
  int param_7_bias_dims,
  float* param_8_weights_data,
  const int* param_8_weights_shape,
  int param_8_weights_dims,
  float* param_9_bias_data,
  const int* param_9_bias_shape,
  int param_9_bias_dims,
  float* param_10_gamma_data,
  const int* param_10_gamma_shape,
  int param_10_gamma_dims,
  float* param_11_beta_data,
  const int* param_11_beta_shape,
  int param_11_beta_dims,
  float* param_12_weights_data,
  const int* param_12_weights_shape,
  int param_12_weights_dims,
  float* param_13_bias_data,
  const int* param_13_bias_shape,
  int param_13_bias_dims,
  float* param_14_weights_data,
  const int* param_14_weights_shape,
  int param_14_weights_dims,
  float* param_15_bias_data,
  const int* param_15_bias_shape,
  int param_15_bias_dims,
  float* param_16_gamma_data,
  const int* param_16_gamma_shape,
  int param_16_gamma_dims,
  float* param_17_beta_data,
  const int* param_17_beta_shape,
  int param_17_beta_dims,
  float* param_18_weights_data,
  const int* param_18_weights_shape,
  int param_18_weights_dims,
  float* param_19_bias_data,
  const int* param_19_bias_shape,
  int param_19_bias_dims,
  float* param_20_weights_data,
  const int* param_20_weights_shape,
  int param_20_weights_dims,
  float* param_21_bias_data,
  const int* param_21_bias_shape,
  int param_21_bias_dims,
  float* param_22_weights_data,
  const int* param_22_weights_shape,
  int param_22_weights_dims,
  float* param_23_bias_data,
  const int* param_23_bias_shape,
  int param_23_bias_dims,
  float* param_24_weights_data,
  const int* param_24_weights_shape,
  int param_24_weights_dims,
  float* param_25_bias_data,
  const int* param_25_bias_shape,
  int param_25_bias_dims,
  float* param_26_gamma_data,
  const int* param_26_gamma_shape,
  int param_26_gamma_dims,
  float* param_27_beta_data,
  const int* param_27_beta_shape,
  int param_27_beta_dims,
  float* param_28_weights_data,
  const int* param_28_weights_shape,
  int param_28_weights_dims,
  float* param_29_bias_data,
  const int* param_29_bias_shape,
  int param_29_bias_dims,
  float* param_30_weights_data,
  const int* param_30_weights_shape,
  int param_30_weights_dims,
  float* param_31_bias_data,
  const int* param_31_bias_shape,
  int param_31_bias_dims,
  float* param_32_gamma_data,
  const int* param_32_gamma_shape,
  int param_32_gamma_dims,
  float* param_33_beta_data,
  const int* param_33_beta_shape,
  int param_33_beta_dims,
  float* param_34_weights_data,
  const int* param_34_weights_shape,
  int param_34_weights_dims,
  float* param_35_bias_data,
  const int* param_35_bias_shape,
  int param_35_bias_dims,
  float* param_36_weights_data,
  const int* param_36_weights_shape,
  int param_36_weights_dims,
  float* param_37_bias_data,
  const int* param_37_bias_shape,
  int param_37_bias_dims,
  float* param_38_weights_data,
  const int* param_38_weights_shape,
  int param_38_weights_dims,
  float* param_39_bias_data,
  const int* param_39_bias_shape,
  int param_39_bias_dims,
  float* param_40_weights_data,
  const int* param_40_weights_shape,
  int param_40_weights_dims,
  float* param_41_bias_data,
  const int* param_41_bias_shape,
  int param_41_bias_dims,
  float* param_42_gamma_data,
  const int* param_42_gamma_shape,
  int param_42_gamma_dims,
  float* param_43_beta_data,
  const int* param_43_beta_shape,
  int param_43_beta_dims,
  float* param_44_weights_data,
  const int* param_44_weights_shape,
  int param_44_weights_dims,
  float* param_45_bias_data,
  const int* param_45_bias_shape,
  int param_45_bias_dims,
  float* param_46_weights_data,
  const int* param_46_weights_shape,
  int param_46_weights_dims,
  float* param_47_bias_data,
  const int* param_47_bias_shape,
  int param_47_bias_dims,
  float* param_48_gamma_data,
  const int* param_48_gamma_shape,
  int param_48_gamma_dims,
  float* param_49_beta_data,
  const int* param_49_beta_shape,
  int param_49_beta_dims,
  float* param_50_weights_data,
  const int* param_50_weights_shape,
  int param_50_weights_dims,
  float* param_51_bias_data,
  const int* param_51_bias_shape,
  int param_51_bias_dims,
  float* param_52_weights_data,
  const int* param_52_weights_shape,
  int param_52_weights_dims,
  float* param_53_bias_data,
  const int* param_53_bias_shape,
  int param_53_bias_dims,
  float* param_54_weights_data,
  const int* param_54_weights_shape,
  int param_54_weights_dims,
  float* param_55_bias_data,
  const int* param_55_bias_shape,
  int param_55_bias_dims,
  float* param_56_weights_data,
  const int* param_56_weights_shape,
  int param_56_weights_dims,
  float* param_57_bias_data,
  const int* param_57_bias_shape,
  int param_57_bias_dims,
  float* param_58_gamma_data,
  const int* param_58_gamma_shape,
  int param_58_gamma_dims,
  float* param_59_beta_data,
  const int* param_59_beta_shape,
  int param_59_beta_dims,
  float* param_60_weights_data,
  const int* param_60_weights_shape,
  int param_60_weights_dims,
  float* param_61_bias_data,
  const int* param_61_bias_shape,
  int param_61_bias_dims,
  float* param_62_weights_data,
  const int* param_62_weights_shape,
  int param_62_weights_dims,
  float* param_63_bias_data,
  const int* param_63_bias_shape,
  int param_63_bias_dims,
  float* param_64_gamma_data,
  const int* param_64_gamma_shape,
  int param_64_gamma_dims,
  float* param_65_beta_data,
  const int* param_65_beta_shape,
  int param_65_beta_dims,
  float* param_66_weights_data,
  const int* param_66_weights_shape,
  int param_66_weights_dims,
  float* param_67_bias_data,
  const int* param_67_bias_shape,
  int param_67_bias_dims,
  float* param_68_weights_data,
  const int* param_68_weights_shape,
  int param_68_weights_dims,
  float* param_69_bias_data,
  const int* param_69_bias_shape,
  int param_69_bias_dims,
  float* param_70_weights_data,
  const int* param_70_weights_shape,
  int param_70_weights_dims,
  float* param_71_bias_data,
  const int* param_71_bias_shape,
  int param_71_bias_dims,
  float* param_72_weights_data,
  const int* param_72_weights_shape,
  int param_72_weights_dims,
  float* param_73_bias_data,
  const int* param_73_bias_shape,
  int param_73_bias_dims,
  float* param_74_gamma_data,
  const int* param_74_gamma_shape,
  int param_74_gamma_dims,
  float* param_75_beta_data,
  const int* param_75_beta_shape,
  int param_75_beta_dims,
  float* param_76_weights_data,
  const int* param_76_weights_shape,
  int param_76_weights_dims,
  float* param_77_bias_data,
  const int* param_77_bias_shape,
  int param_77_bias_dims,
  float* param_78_weights_data,
  const int* param_78_weights_shape,
  int param_78_weights_dims,
  float* param_79_bias_data,
  const int* param_79_bias_shape,
  int param_79_bias_dims,
  float* param_80_gamma_data,
  const int* param_80_gamma_shape,
  int param_80_gamma_dims,
  float* param_81_beta_data,
  const int* param_81_beta_shape,
  int param_81_beta_dims,
  float* param_82_weights_data,
  const int* param_82_weights_shape,
  int param_82_weights_dims,
  float* param_83_bias_data,
  const int* param_83_bias_shape,
  int param_83_bias_dims,
  float* param_84_weights_data,
  const int* param_84_weights_shape,
  int param_84_weights_dims,
  float* param_85_bias_data,
  const int* param_85_bias_shape,
  int param_85_bias_dims,
  float* param_86_weights_data,
  const int* param_86_weights_shape,
  int param_86_weights_dims,
  float* param_87_bias_data,
  const int* param_87_bias_shape,
  int param_87_bias_dims,
  float* param_88_weights_data,
  const int* param_88_weights_shape,
  int param_88_weights_dims,
  float* param_89_bias_data,
  const int* param_89_bias_shape,
  int param_89_bias_dims,
  float* param_90_gamma_data,
  const int* param_90_gamma_shape,
  int param_90_gamma_dims,
  float* param_91_beta_data,
  const int* param_91_beta_shape,
  int param_91_beta_dims,
  float* param_92_weights_data,
  const int* param_92_weights_shape,
  int param_92_weights_dims,
  float* param_93_bias_data,
  const int* param_93_bias_shape,
  int param_93_bias_dims,
  float* param_94_weights_data,
  const int* param_94_weights_shape,
  int param_94_weights_dims,
  float* param_95_bias_data,
  const int* param_95_bias_shape,
  int param_95_bias_dims,
  float* param_96_gamma_data,
  const int* param_96_gamma_shape,
  int param_96_gamma_dims,
  float* param_97_beta_data,
  const int* param_97_beta_shape,
  int param_97_beta_dims,
  float* param_98_weights_data,
  const int* param_98_weights_shape,
  int param_98_weights_dims,
  float* param_99_bias_data,
  const int* param_99_bias_shape,
  int param_99_bias_dims,
  char* workspace,
  size_t workspace_size
) {
 // --- Input Validation ---
 if (!workspace) {
   fprintf(stderr, "Error: Null workspace pointer passed to executeGraph\n");
   return;
 }
 
 if (workspace_size < 2199118080) {
   fprintf(stderr, "Error: Insufficient workspace size. Required: 2199118080 bytes, Provided: %zu bytes\n", workspace_size);
   return;
 }
 
 // --- Initialize Dynamic Memory Allocator ---
 WorkspaceAllocator allocator(workspace, workspace_size);

 // --- Variable Declarations ---
  const int intermediate_0_shape[] = {32, 128, 384};
  const int intermediate_1_shape[] = {32, 128, 384};
  const int intermediate_2_shape[] = {32, 128, 384};
  const int intermediate_3_shape[] = {32, 128, 384};
  const int intermediate_4_shape[] = {32, 128, 384};
  const int intermediate_5_shape[] = {32, 6, 128, 64};
  const int intermediate_6_shape[] = {32, 6, 128, 64};
  const int intermediate_7_shape[] = {32, 6, 128, 64};
  const int intermediate_8_shape[] = {32, 6, 128, 128};
  const int intermediate_9_shape[] = {32, 6, 128, 128};
  const int intermediate_10_shape[] = {32, 6, 128, 64};
  const int intermediate_11_shape[] = {32, 128, 384};
  const int intermediate_12_shape[] = {32, 128, 384};
  const int intermediate_13_shape[] = {32, 128, 384};
  const int intermediate_14_shape[] = {32, 128, 384};
  const int intermediate_15_shape[] = {32, 1536};
  const int intermediate_16_shape[] = {32, 1536};
  const int intermediate_17_shape[] = {32, 384};
  const int intermediate_18_shape[] = {32, 128, 384};
  const int intermediate_19_shape[] = {32, 128, 384};
  const int intermediate_20_shape[] = {32, 384};
  const int intermediate_21_shape[] = {32, 384};
  const int intermediate_22_shape[] = {32, 384};
  const int intermediate_23_shape[] = {32, 6, 384, 64};
  const int intermediate_24_shape[] = {32, 6, 384, 64};
  const int intermediate_25_shape[] = {32, 6, 384, 64};
  const int intermediate_26_shape[] = {32, 6, 384, 384};
  const int intermediate_27_shape[] = {32, 6, 384, 384};
  const int intermediate_28_shape[] = {32, 6, 384, 64};
  const int intermediate_29_shape[] = {32, 384, 384};
  const int intermediate_30_shape[] = {32, 384, 384};
  const int intermediate_31_shape[] = {32, 384, 384};
  const int intermediate_32_shape[] = {32, 384, 384};
  const int intermediate_33_shape[] = {32, 1536};
  const int intermediate_34_shape[] = {32, 1536};
  const int intermediate_35_shape[] = {32, 384};
  const int intermediate_36_shape[] = {32, 384, 384};
  const int intermediate_37_shape[] = {32, 384, 384};
  const int intermediate_38_shape[] = {32, 384};
  const int intermediate_39_shape[] = {32, 384};
  const int intermediate_40_shape[] = {32, 384};
  const int intermediate_41_shape[] = {32, 6, 384, 64};
  const int intermediate_42_shape[] = {32, 6, 384, 64};
  const int intermediate_43_shape[] = {32, 6, 384, 64};
  const int intermediate_44_shape[] = {32, 6, 384, 384};
  const int intermediate_45_shape[] = {32, 6, 384, 384};
  const int intermediate_46_shape[] = {32, 6, 384, 64};
  const int intermediate_47_shape[] = {32, 384, 384};
  const int intermediate_48_shape[] = {32, 384, 384};
  const int intermediate_49_shape[] = {32, 384, 384};
  const int intermediate_50_shape[] = {32, 384, 384};
  const int intermediate_51_shape[] = {32, 1536};
  const int intermediate_52_shape[] = {32, 1536};
  const int intermediate_53_shape[] = {32, 384};
  const int intermediate_54_shape[] = {32, 384, 384};
  const int intermediate_55_shape[] = {32, 384, 384};
  const int intermediate_56_shape[] = {32, 384};
  const int intermediate_57_shape[] = {32, 384};
  const int intermediate_58_shape[] = {32, 384};
  const int intermediate_59_shape[] = {32, 6, 384, 64};
  const int intermediate_60_shape[] = {32, 6, 384, 64};
  const int intermediate_61_shape[] = {32, 6, 384, 64};
  const int intermediate_62_shape[] = {32, 6, 384, 384};
  const int intermediate_63_shape[] = {32, 6, 384, 384};
  const int intermediate_64_shape[] = {32, 6, 384, 64};
  const int intermediate_65_shape[] = {32, 384, 384};
  const int intermediate_66_shape[] = {32, 384, 384};
  const int intermediate_67_shape[] = {32, 384, 384};
  const int intermediate_68_shape[] = {32, 384, 384};
  const int intermediate_69_shape[] = {32, 1536};
  const int intermediate_70_shape[] = {32, 1536};
  const int intermediate_71_shape[] = {32, 384};
  const int intermediate_72_shape[] = {32, 384, 384};
  const int intermediate_73_shape[] = {32, 384, 384};
  const int intermediate_74_shape[] = {32, 384};
  const int intermediate_75_shape[] = {32, 384};
  const int intermediate_76_shape[] = {32, 384};
  const int intermediate_77_shape[] = {32, 6, 384, 64};
  const int intermediate_78_shape[] = {32, 6, 384, 64};
  const int intermediate_79_shape[] = {32, 6, 384, 64};
  const int intermediate_80_shape[] = {32, 6, 384, 384};
  const int intermediate_81_shape[] = {32, 6, 384, 384};
  const int intermediate_82_shape[] = {32, 6, 384, 64};
  const int intermediate_83_shape[] = {32, 384, 384};
  const int intermediate_84_shape[] = {32, 384, 384};
  const int intermediate_85_shape[] = {32, 384, 384};
  const int intermediate_86_shape[] = {32, 384, 384};
  const int intermediate_87_shape[] = {32, 1536};
  const int intermediate_88_shape[] = {32, 1536};
  const int intermediate_89_shape[] = {32, 384};
  const int intermediate_90_shape[] = {32, 384, 384};
  const int intermediate_91_shape[] = {32, 384, 384};
  const int intermediate_92_shape[] = {32, 384};
  const int intermediate_93_shape[] = {32, 384};
  const int intermediate_94_shape[] = {32, 384};
  const int intermediate_95_shape[] = {32, 6, 384, 64};
  const int intermediate_96_shape[] = {32, 6, 384, 64};
  const int intermediate_97_shape[] = {32, 6, 384, 64};
  const int intermediate_98_shape[] = {32, 6, 384, 384};
  const int intermediate_99_shape[] = {32, 6, 384, 384};
  const int intermediate_100_shape[] = {32, 6, 384, 64};
  const int intermediate_101_shape[] = {32, 384, 384};
  const int intermediate_102_shape[] = {32, 384, 384};
  const int intermediate_103_shape[] = {32, 384, 384};
  const int intermediate_104_shape[] = {32, 384, 384};
  const int intermediate_105_shape[] = {32, 1536};
  const int intermediate_106_shape[] = {32, 1536};
  const int intermediate_107_shape[] = {32, 384};
  const int intermediate_108_shape[] = {32, 384, 384};
  const int intermediate_109_shape[] = {32, 384, 384};
  const int intermediate_110_shape[] = {32, 65};

 // --- Tensor Struct Instantiation ---
  // Pool: pool_cudanode_fc12c2f8-c5bb-4354-9b8b-84b256ab19b4:output, Offset: 2101346304, Size: 6291456 bytes
  float* intermediate_0_data = (float*)(workspace + 2101346304);
  Tensor<float> intermediate_0_tensor = {intermediate_0_data, intermediate_0_shape, 3};
  // Pool: pool_cudanode_e4e2acdf-c280-49e6-9d7c-f925a17bb397:output, Offset: 2107637760, Size: 6291456 bytes
  float* intermediate_1_data = (float*)(workspace + 2107637760);
  Tensor<float> intermediate_1_tensor = {intermediate_1_data, intermediate_1_shape, 3};
  // Pool: pool_cudanode_ebe17511-2b2d-4334-b6d9-d1ca3b581084:output, Offset: 2113929216, Size: 6291456 bytes
  float* intermediate_2_data = (float*)(workspace + 2113929216);
  Tensor<float> intermediate_2_tensor = {intermediate_2_data, intermediate_2_shape, 3};
  // Pool: pool_cudanode_9a9728a6-e5e5-417a-8dc7-e2856d6e96cc:output, Offset: 2120220672, Size: 6291456 bytes
  float* intermediate_3_data = (float*)(workspace + 2120220672);
  Tensor<float> intermediate_3_tensor = {intermediate_3_data, intermediate_3_shape, 3};
  // Pool: pool_cudanode_ebf2f31e-ac88-4264-b522-fbc688a9f41f:output, Offset: 2126512128, Size: 6291456 bytes
  float* intermediate_4_data = (float*)(workspace + 2126512128);
  Tensor<float> intermediate_4_tensor = {intermediate_4_data, intermediate_4_shape, 3};
  // Pool: pool_cudanode_13315113-87f3-47bb-93c8-cbedf7b21b57:output, Offset: 2132803584, Size: 6291456 bytes
  float* intermediate_5_data = (float*)(workspace + 2132803584);
  Tensor<float> intermediate_5_tensor = {intermediate_5_data, intermediate_5_shape, 4};
  // Pool: pool_cudanode_01282cbd-e8f5-4c05-b564-186dbe1485af:output, Offset: 2139095040, Size: 6291456 bytes
  float* intermediate_6_data = (float*)(workspace + 2139095040);
  Tensor<float> intermediate_6_tensor = {intermediate_6_data, intermediate_6_shape, 4};
  // Pool: pool_cudanode_d5cf4773-3640-4a53-b25f-4cb8659d089d:output, Offset: 2145386496, Size: 6291456 bytes
  float* intermediate_7_data = (float*)(workspace + 2145386496);
  Tensor<float> intermediate_7_tensor = {intermediate_7_data, intermediate_7_shape, 4};
  // Pool: pool_cudanode_15e9e13e-34b2-413d-9bcf-e422fdb3c3df:output, Offset: 2076180480, Size: 12582912 bytes
  float* intermediate_8_data = (float*)(workspace + 2076180480);
  Tensor<float> intermediate_8_tensor = {intermediate_8_data, intermediate_8_shape, 4};
  // Pool: pool_cudanode_b08c3886-4f7c-4dd4-b7c5-da3d73e5f835:output, Offset: 2088763392, Size: 12582912 bytes
  float* intermediate_9_data = (float*)(workspace + 2088763392);
  Tensor<float> intermediate_9_tensor = {intermediate_9_data, intermediate_9_shape, 4};
  // Pool: pool_cudanode_45c8ed81-9c2f-4291-af37-a68210b31eaa:output, Offset: 2151677952, Size: 6291456 bytes
  float* intermediate_10_data = (float*)(workspace + 2151677952);
  Tensor<float> intermediate_10_tensor = {intermediate_10_data, intermediate_10_shape, 4};
  // Pool: pool_cudanode_8e47be00-e041-4da6-b2cb-a825dfd4e7bf:output, Offset: 2157969408, Size: 6291456 bytes
  float* intermediate_11_data = (float*)(workspace + 2157969408);
  Tensor<float> intermediate_11_tensor = {intermediate_11_data, intermediate_11_shape, 3};
  // Pool: pool_cudanode_fc850f0c-4202-4221-a6f7-dfa82d1e4695:output, Offset: 2164260864, Size: 6291456 bytes
  float* intermediate_12_data = (float*)(workspace + 2164260864);
  Tensor<float> intermediate_12_tensor = {intermediate_12_data, intermediate_12_shape, 3};
  // Pool: pool_cudanode_ce82a590-baca-490b-96f8-be85409daede:output, Offset: 2170552320, Size: 6291456 bytes
  float* intermediate_13_data = (float*)(workspace + 2170552320);
  Tensor<float> intermediate_13_tensor = {intermediate_13_data, intermediate_13_shape, 3};
  // Pool: pool_cudanode_a1303d29-56d4-4cac-b280-087e6243db50:output, Offset: 2176843776, Size: 6291456 bytes
  float* intermediate_14_data = (float*)(workspace + 2176843776);
  Tensor<float> intermediate_14_tensor = {intermediate_14_data, intermediate_14_shape, 3};
  // Pool: pool_cudanode_0847cf3d-c0cf-4ad9-8033-a34e538941c5:output, Offset: 2195718144, Size: 196608 bytes
  float* intermediate_15_data = (float*)(workspace + 2195718144);
  Tensor<float> intermediate_15_tensor = {intermediate_15_data, intermediate_15_shape, 2};
  // Pool: pool_cudanode_e064cb2a-ac3b-4353-9ea8-c75ccbb06d24:output, Offset: 2195914752, Size: 196608 bytes
  float* intermediate_16_data = (float*)(workspace + 2195914752);
  Tensor<float> intermediate_16_tensor = {intermediate_16_data, intermediate_16_shape, 2};
  // Pool: pool_cudanode_b2d9fad8-c426-4689-9408-1af104ecceb3:output, Offset: 2198077440, Size: 49152 bytes
  float* intermediate_17_data = (float*)(workspace + 2198077440);
  Tensor<float> intermediate_17_tensor = {intermediate_17_data, intermediate_17_shape, 2};
  // Pool: pool_cudanode_a5983964-c6e9-40da-82ba-61f6bd8e2296:output, Offset: 2183135232, Size: 6291456 bytes
  float* intermediate_18_data = (float*)(workspace + 2183135232);
  Tensor<float> intermediate_18_tensor = {intermediate_18_data, intermediate_18_shape, 3};
  // Pool: pool_cudanode_367a88b8-7575-402c-915d-607932695bbc:output, Offset: 2189426688, Size: 6291456 bytes
  float* intermediate_19_data = (float*)(workspace + 2189426688);
  Tensor<float> intermediate_19_tensor = {intermediate_19_data, intermediate_19_shape, 3};
  // Pool: pool_cudanode_3afbc0d1-ed43-4257-8f66-cf426a03ae1b:output, Offset: 2198126592, Size: 49152 bytes
  float* intermediate_20_data = (float*)(workspace + 2198126592);
  Tensor<float> intermediate_20_tensor = {intermediate_20_data, intermediate_20_shape, 2};
  // Pool: pool_cudanode_688736a1-be22-405a-880d-616cafed783e:output, Offset: 2198175744, Size: 49152 bytes
  float* intermediate_21_data = (float*)(workspace + 2198175744);
  Tensor<float> intermediate_21_tensor = {intermediate_21_data, intermediate_21_shape, 2};
  // Pool: pool_cudanode_899c7055-469a-4fec-8a2f-cfb1688f8873:output, Offset: 2198224896, Size: 49152 bytes
  float* intermediate_22_data = (float*)(workspace + 2198224896);
  Tensor<float> intermediate_22_tensor = {intermediate_22_data, intermediate_22_shape, 2};
  // Pool: pool_cudanode_e5190500-712d-44fe-9e21-7d194823ffd3:output, Offset: 1132462080, Size: 18874368 bytes
  float* intermediate_23_data = (float*)(workspace + 1132462080);
  Tensor<float> intermediate_23_tensor = {intermediate_23_data, intermediate_23_shape, 4};
  // Pool: pool_cudanode_f0730386-fc5c-4e14-a7f8-77ea7a7b48aa:output, Offset: 1151336448, Size: 18874368 bytes
  float* intermediate_24_data = (float*)(workspace + 1151336448);
  Tensor<float> intermediate_24_tensor = {intermediate_24_data, intermediate_24_shape, 4};
  // Pool: pool_cudanode_6c1ff3d0-4616-4c6d-8961-c36cb555e234:output, Offset: 1170210816, Size: 18874368 bytes
  float* intermediate_25_data = (float*)(workspace + 1170210816);
  Tensor<float> intermediate_25_tensor = {intermediate_25_data, intermediate_25_shape, 4};
  // Pool: pool_cudanode_7358e2bd-62da-49a7-af95-796cab7e921c:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_26_data = (float*)(workspace + 0);
  Tensor<float> intermediate_26_tensor = {intermediate_26_data, intermediate_26_shape, 4};
  // Pool: pool_cudanode_9b29362f-9b45-4085-9415-7b1ee327f840:output, Offset: 113246208, Size: 113246208 bytes
  float* intermediate_27_data = (float*)(workspace + 113246208);
  Tensor<float> intermediate_27_tensor = {intermediate_27_data, intermediate_27_shape, 4};
  // Pool: pool_cudanode_dcc14e11-ce85-4def-9534-d6dcc7ee6e97:output, Offset: 1189085184, Size: 18874368 bytes
  float* intermediate_28_data = (float*)(workspace + 1189085184);
  Tensor<float> intermediate_28_tensor = {intermediate_28_data, intermediate_28_shape, 4};
  // Pool: pool_cudanode_8246bc03-914e-4229-9eab-df222689fb50:output, Offset: 1207959552, Size: 18874368 bytes
  float* intermediate_29_data = (float*)(workspace + 1207959552);
  Tensor<float> intermediate_29_tensor = {intermediate_29_data, intermediate_29_shape, 3};
  // Pool: pool_cudanode_22a226a9-e1b0-4d2d-9552-93ce98a6cc57:output, Offset: 1226833920, Size: 18874368 bytes
  float* intermediate_30_data = (float*)(workspace + 1226833920);
  Tensor<float> intermediate_30_tensor = {intermediate_30_data, intermediate_30_shape, 3};
  // Pool: pool_cudanode_d871ce7f-1d6d-41df-9732-5c0f95927d61:output, Offset: 1245708288, Size: 18874368 bytes
  float* intermediate_31_data = (float*)(workspace + 1245708288);
  Tensor<float> intermediate_31_tensor = {intermediate_31_data, intermediate_31_shape, 3};
  // Pool: pool_cudanode_d1b389b8-7eb2-4979-8a20-1ae73ec7f033:output, Offset: 1264582656, Size: 18874368 bytes
  float* intermediate_32_data = (float*)(workspace + 1264582656);
  Tensor<float> intermediate_32_tensor = {intermediate_32_data, intermediate_32_shape, 3};
  // Pool: pool_cudanode_eea829c9-2c44-45ad-91f5-319e12b5a0f6:output, Offset: 2196111360, Size: 196608 bytes
  float* intermediate_33_data = (float*)(workspace + 2196111360);
  Tensor<float> intermediate_33_tensor = {intermediate_33_data, intermediate_33_shape, 2};
  // Pool: pool_cudanode_39dd89a4-e198-4fb1-b34b-fa06f9bf8cae:output, Offset: 2196307968, Size: 196608 bytes
  float* intermediate_34_data = (float*)(workspace + 2196307968);
  Tensor<float> intermediate_34_tensor = {intermediate_34_data, intermediate_34_shape, 2};
  // Pool: pool_cudanode_f1c6da1f-71cc-40a0-b8af-84dde77ad0c5:output, Offset: 2198274048, Size: 49152 bytes
  float* intermediate_35_data = (float*)(workspace + 2198274048);
  Tensor<float> intermediate_35_tensor = {intermediate_35_data, intermediate_35_shape, 2};
  // Pool: pool_cudanode_93f692a8-4227-41e6-8c89-8b9826d41b3b:output, Offset: 1283457024, Size: 18874368 bytes
  float* intermediate_36_data = (float*)(workspace + 1283457024);
  Tensor<float> intermediate_36_tensor = {intermediate_36_data, intermediate_36_shape, 3};
  // Pool: pool_cudanode_a7eb5714-8683-46ec-ae3a-caa1c5ec4248:output, Offset: 1302331392, Size: 18874368 bytes
  float* intermediate_37_data = (float*)(workspace + 1302331392);
  Tensor<float> intermediate_37_tensor = {intermediate_37_data, intermediate_37_shape, 3};
  // Pool: pool_cudanode_7f450324-5bd3-404a-b3b9-d2b7e00dddf1:output, Offset: 2198323200, Size: 49152 bytes
  float* intermediate_38_data = (float*)(workspace + 2198323200);
  Tensor<float> intermediate_38_tensor = {intermediate_38_data, intermediate_38_shape, 2};
  // Pool: pool_cudanode_66290b7b-18e2-4b3b-a930-31ed321687df:output, Offset: 2198372352, Size: 49152 bytes
  float* intermediate_39_data = (float*)(workspace + 2198372352);
  Tensor<float> intermediate_39_tensor = {intermediate_39_data, intermediate_39_shape, 2};
  // Pool: pool_cudanode_31e08e84-69f7-4acb-9476-00e45c1ca971:output, Offset: 2198421504, Size: 49152 bytes
  float* intermediate_40_data = (float*)(workspace + 2198421504);
  Tensor<float> intermediate_40_tensor = {intermediate_40_data, intermediate_40_shape, 2};
  // Pool: pool_cudanode_c263c98e-9b7a-4fb2-afc9-444954061817:output, Offset: 1321205760, Size: 18874368 bytes
  float* intermediate_41_data = (float*)(workspace + 1321205760);
  Tensor<float> intermediate_41_tensor = {intermediate_41_data, intermediate_41_shape, 4};
  // Pool: pool_cudanode_0a20dbf4-e75d-4fe3-b622-76d8b31b8232:output, Offset: 1340080128, Size: 18874368 bytes
  float* intermediate_42_data = (float*)(workspace + 1340080128);
  Tensor<float> intermediate_42_tensor = {intermediate_42_data, intermediate_42_shape, 4};
  // Pool: pool_cudanode_1176b639-f9d2-47a6-8c93-585d0e738fc6:output, Offset: 1358954496, Size: 18874368 bytes
  float* intermediate_43_data = (float*)(workspace + 1358954496);
  Tensor<float> intermediate_43_tensor = {intermediate_43_data, intermediate_43_shape, 4};
  // Pool: pool_cudanode_efda1109-314d-4b2f-9760-8459a00ea3b2:output, Offset: 226492416, Size: 113246208 bytes
  float* intermediate_44_data = (float*)(workspace + 226492416);
  Tensor<float> intermediate_44_tensor = {intermediate_44_data, intermediate_44_shape, 4};
  // Pool: pool_cudanode_0b6d1a1c-adf0-42eb-81d2-35c107d1d2c7:output, Offset: 339738624, Size: 113246208 bytes
  float* intermediate_45_data = (float*)(workspace + 339738624);
  Tensor<float> intermediate_45_tensor = {intermediate_45_data, intermediate_45_shape, 4};
  // Pool: pool_cudanode_0239d774-779d-45ce-945e-7cd7e42c8f1f:output, Offset: 1377828864, Size: 18874368 bytes
  float* intermediate_46_data = (float*)(workspace + 1377828864);
  Tensor<float> intermediate_46_tensor = {intermediate_46_data, intermediate_46_shape, 4};
  // Pool: pool_cudanode_ce632f5e-12e0-4812-a2c6-5fc78776087e:output, Offset: 1396703232, Size: 18874368 bytes
  float* intermediate_47_data = (float*)(workspace + 1396703232);
  Tensor<float> intermediate_47_tensor = {intermediate_47_data, intermediate_47_shape, 3};
  // Pool: pool_cudanode_b876f286-d41f-4dc8-9f77-0868d5192bc7:output, Offset: 1415577600, Size: 18874368 bytes
  float* intermediate_48_data = (float*)(workspace + 1415577600);
  Tensor<float> intermediate_48_tensor = {intermediate_48_data, intermediate_48_shape, 3};
  // Pool: pool_cudanode_81aa3f9e-c633-4148-a6ac-ff3a50d3ebd6:output, Offset: 1434451968, Size: 18874368 bytes
  float* intermediate_49_data = (float*)(workspace + 1434451968);
  Tensor<float> intermediate_49_tensor = {intermediate_49_data, intermediate_49_shape, 3};
  // Pool: pool_cudanode_cc99b7d8-ff98-486f-8d76-9a0355a89ab6:output, Offset: 1453326336, Size: 18874368 bytes
  float* intermediate_50_data = (float*)(workspace + 1453326336);
  Tensor<float> intermediate_50_tensor = {intermediate_50_data, intermediate_50_shape, 3};
  // Pool: pool_cudanode_632c7dd6-9493-4f09-98bc-bead66dff44a:output, Offset: 2196504576, Size: 196608 bytes
  float* intermediate_51_data = (float*)(workspace + 2196504576);
  Tensor<float> intermediate_51_tensor = {intermediate_51_data, intermediate_51_shape, 2};
  // Pool: pool_cudanode_4feb0f53-f9ad-4527-8b24-2526e00d33a8:output, Offset: 2196701184, Size: 196608 bytes
  float* intermediate_52_data = (float*)(workspace + 2196701184);
  Tensor<float> intermediate_52_tensor = {intermediate_52_data, intermediate_52_shape, 2};
  // Pool: pool_cudanode_76c11bc8-5181-4149-9b1b-717fab5062e5:output, Offset: 2198470656, Size: 49152 bytes
  float* intermediate_53_data = (float*)(workspace + 2198470656);
  Tensor<float> intermediate_53_tensor = {intermediate_53_data, intermediate_53_shape, 2};
  // Pool: pool_cudanode_9c101160-95d7-4c6c-b9f1-37b90644f1bf:output, Offset: 1472200704, Size: 18874368 bytes
  float* intermediate_54_data = (float*)(workspace + 1472200704);
  Tensor<float> intermediate_54_tensor = {intermediate_54_data, intermediate_54_shape, 3};
  // Pool: pool_cudanode_66869773-5b27-40e4-ab33-07afbca8c61c:output, Offset: 1491075072, Size: 18874368 bytes
  float* intermediate_55_data = (float*)(workspace + 1491075072);
  Tensor<float> intermediate_55_tensor = {intermediate_55_data, intermediate_55_shape, 3};
  // Pool: pool_cudanode_5aec6712-3ae7-4f74-a2c4-9f690c8807fc:output, Offset: 2198519808, Size: 49152 bytes
  float* intermediate_56_data = (float*)(workspace + 2198519808);
  Tensor<float> intermediate_56_tensor = {intermediate_56_data, intermediate_56_shape, 2};
  // Pool: pool_cudanode_699d1b92-d7eb-496a-b8cc-8cf323d681e6:output, Offset: 2198568960, Size: 49152 bytes
  float* intermediate_57_data = (float*)(workspace + 2198568960);
  Tensor<float> intermediate_57_tensor = {intermediate_57_data, intermediate_57_shape, 2};
  // Pool: pool_cudanode_fa5840b0-8798-4b72-a497-a584ec3da9f2:output, Offset: 2198618112, Size: 49152 bytes
  float* intermediate_58_data = (float*)(workspace + 2198618112);
  Tensor<float> intermediate_58_tensor = {intermediate_58_data, intermediate_58_shape, 2};
  // Pool: pool_cudanode_a73bde9d-1776-4cf3-a2ee-6dee948a7d19:output, Offset: 1509949440, Size: 18874368 bytes
  float* intermediate_59_data = (float*)(workspace + 1509949440);
  Tensor<float> intermediate_59_tensor = {intermediate_59_data, intermediate_59_shape, 4};
  // Pool: pool_cudanode_a7583406-612c-43f6-87aa-2e4fa9f5f797:output, Offset: 1528823808, Size: 18874368 bytes
  float* intermediate_60_data = (float*)(workspace + 1528823808);
  Tensor<float> intermediate_60_tensor = {intermediate_60_data, intermediate_60_shape, 4};
  // Pool: pool_cudanode_dcb98cb1-5200-4038-aea6-0ba858793495:output, Offset: 1547698176, Size: 18874368 bytes
  float* intermediate_61_data = (float*)(workspace + 1547698176);
  Tensor<float> intermediate_61_tensor = {intermediate_61_data, intermediate_61_shape, 4};
  // Pool: pool_cudanode_af6222c1-ca8c-4d76-ac1f-d7b236ad23dd:output, Offset: 452984832, Size: 113246208 bytes
  float* intermediate_62_data = (float*)(workspace + 452984832);
  Tensor<float> intermediate_62_tensor = {intermediate_62_data, intermediate_62_shape, 4};
  // Pool: pool_cudanode_55272f08-f248-45bc-90c7-08ea57228cc8:output, Offset: 566231040, Size: 113246208 bytes
  float* intermediate_63_data = (float*)(workspace + 566231040);
  Tensor<float> intermediate_63_tensor = {intermediate_63_data, intermediate_63_shape, 4};
  // Pool: pool_cudanode_8b648202-8641-4665-82a2-4df2b96d71c6:output, Offset: 1566572544, Size: 18874368 bytes
  float* intermediate_64_data = (float*)(workspace + 1566572544);
  Tensor<float> intermediate_64_tensor = {intermediate_64_data, intermediate_64_shape, 4};
  // Pool: pool_cudanode_1652cc21-9c92-426a-adf4-d7ee341f355f:output, Offset: 1585446912, Size: 18874368 bytes
  float* intermediate_65_data = (float*)(workspace + 1585446912);
  Tensor<float> intermediate_65_tensor = {intermediate_65_data, intermediate_65_shape, 3};
  // Pool: pool_cudanode_e5e56a82-6457-4b8a-a91f-78f4357f7d67:output, Offset: 1604321280, Size: 18874368 bytes
  float* intermediate_66_data = (float*)(workspace + 1604321280);
  Tensor<float> intermediate_66_tensor = {intermediate_66_data, intermediate_66_shape, 3};
  // Pool: pool_cudanode_4da87e26-3cf7-4715-ac42-310c96fa418b:output, Offset: 1623195648, Size: 18874368 bytes
  float* intermediate_67_data = (float*)(workspace + 1623195648);
  Tensor<float> intermediate_67_tensor = {intermediate_67_data, intermediate_67_shape, 3};
  // Pool: pool_cudanode_2f7db56b-cebb-4ab7-8fe0-77ad502df37c:output, Offset: 1642070016, Size: 18874368 bytes
  float* intermediate_68_data = (float*)(workspace + 1642070016);
  Tensor<float> intermediate_68_tensor = {intermediate_68_data, intermediate_68_shape, 3};
  // Pool: pool_cudanode_b5910d39-2be7-430e-81c4-4678cbc72346:output, Offset: 2196897792, Size: 196608 bytes
  float* intermediate_69_data = (float*)(workspace + 2196897792);
  Tensor<float> intermediate_69_tensor = {intermediate_69_data, intermediate_69_shape, 2};
  // Pool: pool_cudanode_144e15ec-ecf6-4a6d-9e77-119b8dae1add:output, Offset: 2197094400, Size: 196608 bytes
  float* intermediate_70_data = (float*)(workspace + 2197094400);
  Tensor<float> intermediate_70_tensor = {intermediate_70_data, intermediate_70_shape, 2};
  // Pool: pool_cudanode_8f2dedc6-01ad-4ffc-bb41-ac41179bcc26:output, Offset: 2198667264, Size: 49152 bytes
  float* intermediate_71_data = (float*)(workspace + 2198667264);
  Tensor<float> intermediate_71_tensor = {intermediate_71_data, intermediate_71_shape, 2};
  // Pool: pool_cudanode_4c32734d-c729-4f5d-af48-b2beefbf93f6:output, Offset: 1660944384, Size: 18874368 bytes
  float* intermediate_72_data = (float*)(workspace + 1660944384);
  Tensor<float> intermediate_72_tensor = {intermediate_72_data, intermediate_72_shape, 3};
  // Pool: pool_cudanode_bf4fe223-685a-4ce6-8abe-4da45c57518a:output, Offset: 1679818752, Size: 18874368 bytes
  float* intermediate_73_data = (float*)(workspace + 1679818752);
  Tensor<float> intermediate_73_tensor = {intermediate_73_data, intermediate_73_shape, 3};
  // Pool: pool_cudanode_dcd2ec8f-7fd7-434c-b3ca-88ae8ccc4938:output, Offset: 2198716416, Size: 49152 bytes
  float* intermediate_74_data = (float*)(workspace + 2198716416);
  Tensor<float> intermediate_74_tensor = {intermediate_74_data, intermediate_74_shape, 2};
  // Pool: pool_cudanode_76514100-8679-42fa-bb02-25d586478f78:output, Offset: 2198765568, Size: 49152 bytes
  float* intermediate_75_data = (float*)(workspace + 2198765568);
  Tensor<float> intermediate_75_tensor = {intermediate_75_data, intermediate_75_shape, 2};
  // Pool: pool_cudanode_83ae2350-df6a-401a-a07a-bd2dd645e2de:output, Offset: 2198814720, Size: 49152 bytes
  float* intermediate_76_data = (float*)(workspace + 2198814720);
  Tensor<float> intermediate_76_tensor = {intermediate_76_data, intermediate_76_shape, 2};
  // Pool: pool_cudanode_c1068322-831f-4112-b29c-51b65640cbd4:output, Offset: 1698693120, Size: 18874368 bytes
  float* intermediate_77_data = (float*)(workspace + 1698693120);
  Tensor<float> intermediate_77_tensor = {intermediate_77_data, intermediate_77_shape, 4};
  // Pool: pool_cudanode_2ac3f629-cda0-4de8-a6fc-de589838a2dc:output, Offset: 1717567488, Size: 18874368 bytes
  float* intermediate_78_data = (float*)(workspace + 1717567488);
  Tensor<float> intermediate_78_tensor = {intermediate_78_data, intermediate_78_shape, 4};
  // Pool: pool_cudanode_3b9abf59-466f-4f9f-933a-f0d2d35b67fb:output, Offset: 1736441856, Size: 18874368 bytes
  float* intermediate_79_data = (float*)(workspace + 1736441856);
  Tensor<float> intermediate_79_tensor = {intermediate_79_data, intermediate_79_shape, 4};
  // Pool: pool_cudanode_e2ef595a-8cd7-4ddc-a685-015e52c1a1d5:output, Offset: 679477248, Size: 113246208 bytes
  float* intermediate_80_data = (float*)(workspace + 679477248);
  Tensor<float> intermediate_80_tensor = {intermediate_80_data, intermediate_80_shape, 4};
  // Pool: pool_cudanode_d75ae1d5-0774-4719-9ace-af141d8fda8a:output, Offset: 792723456, Size: 113246208 bytes
  float* intermediate_81_data = (float*)(workspace + 792723456);
  Tensor<float> intermediate_81_tensor = {intermediate_81_data, intermediate_81_shape, 4};
  // Pool: pool_cudanode_61c6fd75-eb9a-40de-a55c-17e074d5acba:output, Offset: 1755316224, Size: 18874368 bytes
  float* intermediate_82_data = (float*)(workspace + 1755316224);
  Tensor<float> intermediate_82_tensor = {intermediate_82_data, intermediate_82_shape, 4};
  // Pool: pool_cudanode_cc2d1108-d0cb-498c-a3bf-904b2841be0f:output, Offset: 1774190592, Size: 18874368 bytes
  float* intermediate_83_data = (float*)(workspace + 1774190592);
  Tensor<float> intermediate_83_tensor = {intermediate_83_data, intermediate_83_shape, 3};
  // Pool: pool_cudanode_8fe1014f-2952-4cdf-8dc3-f65b0155daed:output, Offset: 1793064960, Size: 18874368 bytes
  float* intermediate_84_data = (float*)(workspace + 1793064960);
  Tensor<float> intermediate_84_tensor = {intermediate_84_data, intermediate_84_shape, 3};
  // Pool: pool_cudanode_f069ead9-e7f9-475b-a31b-fcf5dca7d8be:output, Offset: 1811939328, Size: 18874368 bytes
  float* intermediate_85_data = (float*)(workspace + 1811939328);
  Tensor<float> intermediate_85_tensor = {intermediate_85_data, intermediate_85_shape, 3};
  // Pool: pool_cudanode_7e8aa974-a927-4827-a967-85b446e5c937:output, Offset: 1830813696, Size: 18874368 bytes
  float* intermediate_86_data = (float*)(workspace + 1830813696);
  Tensor<float> intermediate_86_tensor = {intermediate_86_data, intermediate_86_shape, 3};
  // Pool: pool_cudanode_559933ef-a5e8-4e0b-9193-157b14a1f56b:output, Offset: 2197291008, Size: 196608 bytes
  float* intermediate_87_data = (float*)(workspace + 2197291008);
  Tensor<float> intermediate_87_tensor = {intermediate_87_data, intermediate_87_shape, 2};
  // Pool: pool_cudanode_cd04dc8d-4ad1-44e6-af6e-29358b1f233c:output, Offset: 2197487616, Size: 196608 bytes
  float* intermediate_88_data = (float*)(workspace + 2197487616);
  Tensor<float> intermediate_88_tensor = {intermediate_88_data, intermediate_88_shape, 2};
  // Pool: pool_cudanode_4b62e6f5-450e-4d31-8462-73418bfaa5b5:output, Offset: 2198863872, Size: 49152 bytes
  float* intermediate_89_data = (float*)(workspace + 2198863872);
  Tensor<float> intermediate_89_tensor = {intermediate_89_data, intermediate_89_shape, 2};
  // Pool: pool_cudanode_0ef1e205-f02a-41ed-9934-effe7c500ed5:output, Offset: 1849688064, Size: 18874368 bytes
  float* intermediate_90_data = (float*)(workspace + 1849688064);
  Tensor<float> intermediate_90_tensor = {intermediate_90_data, intermediate_90_shape, 3};
  // Pool: pool_cudanode_18e62d86-f2aa-4ea5-8a0f-f392124134b9:output, Offset: 1868562432, Size: 18874368 bytes
  float* intermediate_91_data = (float*)(workspace + 1868562432);
  Tensor<float> intermediate_91_tensor = {intermediate_91_data, intermediate_91_shape, 3};
  // Pool: pool_cudanode_ec06de81-c646-4d9f-b9f1-430d90524c82:output, Offset: 2198913024, Size: 49152 bytes
  float* intermediate_92_data = (float*)(workspace + 2198913024);
  Tensor<float> intermediate_92_tensor = {intermediate_92_data, intermediate_92_shape, 2};
  // Pool: pool_cudanode_a86b8413-07c1-4022-beb1-1ebf9c373401:output, Offset: 2198962176, Size: 49152 bytes
  float* intermediate_93_data = (float*)(workspace + 2198962176);
  Tensor<float> intermediate_93_tensor = {intermediate_93_data, intermediate_93_shape, 2};
  // Pool: pool_cudanode_df5fa1d9-9212-46d2-ab2b-6a5ac6e04585:output, Offset: 2199011328, Size: 49152 bytes
  float* intermediate_94_data = (float*)(workspace + 2199011328);
  Tensor<float> intermediate_94_tensor = {intermediate_94_data, intermediate_94_shape, 2};
  // Pool: pool_cudanode_de710529-89bc-44d8-86ef-8dfa69b02dca:output, Offset: 1887436800, Size: 18874368 bytes
  float* intermediate_95_data = (float*)(workspace + 1887436800);
  Tensor<float> intermediate_95_tensor = {intermediate_95_data, intermediate_95_shape, 4};
  // Pool: pool_cudanode_534ac6ae-b97a-4f08-9a71-76e353e65e6f:output, Offset: 1906311168, Size: 18874368 bytes
  float* intermediate_96_data = (float*)(workspace + 1906311168);
  Tensor<float> intermediate_96_tensor = {intermediate_96_data, intermediate_96_shape, 4};
  // Pool: pool_cudanode_3a011716-b2f6-462c-8d6f-9cb0474d8f48:output, Offset: 1925185536, Size: 18874368 bytes
  float* intermediate_97_data = (float*)(workspace + 1925185536);
  Tensor<float> intermediate_97_tensor = {intermediate_97_data, intermediate_97_shape, 4};
  // Pool: pool_cudanode_420e1d0c-5ea6-4749-bce3-b4f857a3a40c:output, Offset: 905969664, Size: 113246208 bytes
  float* intermediate_98_data = (float*)(workspace + 905969664);
  Tensor<float> intermediate_98_tensor = {intermediate_98_data, intermediate_98_shape, 4};
  // Pool: pool_cudanode_798d8b09-e1f6-4d47-942b-f7533f138f65:output, Offset: 1019215872, Size: 113246208 bytes
  float* intermediate_99_data = (float*)(workspace + 1019215872);
  Tensor<float> intermediate_99_tensor = {intermediate_99_data, intermediate_99_shape, 4};
  // Pool: pool_cudanode_b1a59782-3b7a-4f25-b6c5-cf779f0dfd33:output, Offset: 1944059904, Size: 18874368 bytes
  float* intermediate_100_data = (float*)(workspace + 1944059904);
  Tensor<float> intermediate_100_tensor = {intermediate_100_data, intermediate_100_shape, 4};
  // Pool: pool_cudanode_a1a41431-79ba-427b-8936-56caa5d6feb0:output, Offset: 1962934272, Size: 18874368 bytes
  float* intermediate_101_data = (float*)(workspace + 1962934272);
  Tensor<float> intermediate_101_tensor = {intermediate_101_data, intermediate_101_shape, 3};
  // Pool: pool_cudanode_924c56a3-a3d9-417e-8ad9-f9564169cb48:output, Offset: 1981808640, Size: 18874368 bytes
  float* intermediate_102_data = (float*)(workspace + 1981808640);
  Tensor<float> intermediate_102_tensor = {intermediate_102_data, intermediate_102_shape, 3};
  // Pool: pool_cudanode_89685afc-aa5b-4d7d-8dc8-53519ee2c75b:output, Offset: 2000683008, Size: 18874368 bytes
  float* intermediate_103_data = (float*)(workspace + 2000683008);
  Tensor<float> intermediate_103_tensor = {intermediate_103_data, intermediate_103_shape, 3};
  // Pool: pool_cudanode_78c937cf-7f77-432e-a37e-7fb191e28a31:output, Offset: 2019557376, Size: 18874368 bytes
  float* intermediate_104_data = (float*)(workspace + 2019557376);
  Tensor<float> intermediate_104_tensor = {intermediate_104_data, intermediate_104_shape, 3};
  // Pool: pool_cudanode_ff20b597-36d4-40be-a7d8-e2824260388e:output, Offset: 2197684224, Size: 196608 bytes
  float* intermediate_105_data = (float*)(workspace + 2197684224);
  Tensor<float> intermediate_105_tensor = {intermediate_105_data, intermediate_105_shape, 2};
  // Pool: pool_cudanode_ec8ab86d-db9b-4c60-9e96-484bf9ed74bb:output, Offset: 2197880832, Size: 196608 bytes
  float* intermediate_106_data = (float*)(workspace + 2197880832);
  Tensor<float> intermediate_106_tensor = {intermediate_106_data, intermediate_106_shape, 2};
  // Pool: pool_cudanode_bf143c36-31d4-49a0-8274-f8c0d2949e27:output, Offset: 2199060480, Size: 49152 bytes
  float* intermediate_107_data = (float*)(workspace + 2199060480);
  Tensor<float> intermediate_107_tensor = {intermediate_107_data, intermediate_107_shape, 2};
  // Pool: pool_cudanode_b98850a9-314b-4b8c-8619-acf99aeec74a:output, Offset: 2038431744, Size: 18874368 bytes
  float* intermediate_108_data = (float*)(workspace + 2038431744);
  Tensor<float> intermediate_108_tensor = {intermediate_108_data, intermediate_108_shape, 3};
  // Pool: pool_cudanode_d4fbf5a8-302a-4cec-8180-05aea2e6ab92:output, Offset: 2057306112, Size: 18874368 bytes
  float* intermediate_109_data = (float*)(workspace + 2057306112);
  Tensor<float> intermediate_109_tensor = {intermediate_109_data, intermediate_109_shape, 3};
  // Pool: pool_cudanode_833de466-f52e-4534-b794-49945878517a:output, Offset: 2199109632, Size: 8320 bytes
  float* intermediate_110_data = (float*)(workspace + 2199109632);
  Tensor<float> intermediate_110_tensor = {intermediate_110_data, intermediate_110_shape, 2};
  Tensor<int> input = {(int*)input_data, input_shape, input_dims};
  Tensor<float> output = {output_data, output_shape, output_dims};
  Tensor<float> param_0_embeddings = {param_0_embeddings_data, param_0_embeddings_shape, param_0_embeddings_dims};
  Tensor<float> param_1_frequencies = {param_1_frequencies_data, param_1_frequencies_shape, param_1_frequencies_dims};
  Tensor<float> param_2_weights = {param_2_weights_data, param_2_weights_shape, param_2_weights_dims};
  Tensor<float> param_3_bias = {param_3_bias_data, param_3_bias_shape, param_3_bias_dims};
  Tensor<float> param_4_weights = {param_4_weights_data, param_4_weights_shape, param_4_weights_dims};
  Tensor<float> param_5_bias = {param_5_bias_data, param_5_bias_shape, param_5_bias_dims};
  Tensor<float> param_6_weights = {param_6_weights_data, param_6_weights_shape, param_6_weights_dims};
  Tensor<float> param_7_bias = {param_7_bias_data, param_7_bias_shape, param_7_bias_dims};
  Tensor<float> param_8_weights = {param_8_weights_data, param_8_weights_shape, param_8_weights_dims};
  Tensor<float> param_9_bias = {param_9_bias_data, param_9_bias_shape, param_9_bias_dims};
  Tensor<float> param_10_gamma = {param_10_gamma_data, param_10_gamma_shape, param_10_gamma_dims};
  Tensor<float> param_11_beta = {param_11_beta_data, param_11_beta_shape, param_11_beta_dims};
  Tensor<float> param_12_weights = {param_12_weights_data, param_12_weights_shape, param_12_weights_dims};
  Tensor<float> param_13_bias = {param_13_bias_data, param_13_bias_shape, param_13_bias_dims};
  Tensor<float> param_14_weights = {param_14_weights_data, param_14_weights_shape, param_14_weights_dims};
  Tensor<float> param_15_bias = {param_15_bias_data, param_15_bias_shape, param_15_bias_dims};
  Tensor<float> param_16_gamma = {param_16_gamma_data, param_16_gamma_shape, param_16_gamma_dims};
  Tensor<float> param_17_beta = {param_17_beta_data, param_17_beta_shape, param_17_beta_dims};
  Tensor<float> param_18_weights = {param_18_weights_data, param_18_weights_shape, param_18_weights_dims};
  Tensor<float> param_19_bias = {param_19_bias_data, param_19_bias_shape, param_19_bias_dims};
  Tensor<float> param_20_weights = {param_20_weights_data, param_20_weights_shape, param_20_weights_dims};
  Tensor<float> param_21_bias = {param_21_bias_data, param_21_bias_shape, param_21_bias_dims};
  Tensor<float> param_22_weights = {param_22_weights_data, param_22_weights_shape, param_22_weights_dims};
  Tensor<float> param_23_bias = {param_23_bias_data, param_23_bias_shape, param_23_bias_dims};
  Tensor<float> param_24_weights = {param_24_weights_data, param_24_weights_shape, param_24_weights_dims};
  Tensor<float> param_25_bias = {param_25_bias_data, param_25_bias_shape, param_25_bias_dims};
  Tensor<float> param_26_gamma = {param_26_gamma_data, param_26_gamma_shape, param_26_gamma_dims};
  Tensor<float> param_27_beta = {param_27_beta_data, param_27_beta_shape, param_27_beta_dims};
  Tensor<float> param_28_weights = {param_28_weights_data, param_28_weights_shape, param_28_weights_dims};
  Tensor<float> param_29_bias = {param_29_bias_data, param_29_bias_shape, param_29_bias_dims};
  Tensor<float> param_30_weights = {param_30_weights_data, param_30_weights_shape, param_30_weights_dims};
  Tensor<float> param_31_bias = {param_31_bias_data, param_31_bias_shape, param_31_bias_dims};
  Tensor<float> param_32_gamma = {param_32_gamma_data, param_32_gamma_shape, param_32_gamma_dims};
  Tensor<float> param_33_beta = {param_33_beta_data, param_33_beta_shape, param_33_beta_dims};
  Tensor<float> param_34_weights = {param_34_weights_data, param_34_weights_shape, param_34_weights_dims};
  Tensor<float> param_35_bias = {param_35_bias_data, param_35_bias_shape, param_35_bias_dims};
  Tensor<float> param_36_weights = {param_36_weights_data, param_36_weights_shape, param_36_weights_dims};
  Tensor<float> param_37_bias = {param_37_bias_data, param_37_bias_shape, param_37_bias_dims};
  Tensor<float> param_38_weights = {param_38_weights_data, param_38_weights_shape, param_38_weights_dims};
  Tensor<float> param_39_bias = {param_39_bias_data, param_39_bias_shape, param_39_bias_dims};
  Tensor<float> param_40_weights = {param_40_weights_data, param_40_weights_shape, param_40_weights_dims};
  Tensor<float> param_41_bias = {param_41_bias_data, param_41_bias_shape, param_41_bias_dims};
  Tensor<float> param_42_gamma = {param_42_gamma_data, param_42_gamma_shape, param_42_gamma_dims};
  Tensor<float> param_43_beta = {param_43_beta_data, param_43_beta_shape, param_43_beta_dims};
  Tensor<float> param_44_weights = {param_44_weights_data, param_44_weights_shape, param_44_weights_dims};
  Tensor<float> param_45_bias = {param_45_bias_data, param_45_bias_shape, param_45_bias_dims};
  Tensor<float> param_46_weights = {param_46_weights_data, param_46_weights_shape, param_46_weights_dims};
  Tensor<float> param_47_bias = {param_47_bias_data, param_47_bias_shape, param_47_bias_dims};
  Tensor<float> param_48_gamma = {param_48_gamma_data, param_48_gamma_shape, param_48_gamma_dims};
  Tensor<float> param_49_beta = {param_49_beta_data, param_49_beta_shape, param_49_beta_dims};
  Tensor<float> param_50_weights = {param_50_weights_data, param_50_weights_shape, param_50_weights_dims};
  Tensor<float> param_51_bias = {param_51_bias_data, param_51_bias_shape, param_51_bias_dims};
  Tensor<float> param_52_weights = {param_52_weights_data, param_52_weights_shape, param_52_weights_dims};
  Tensor<float> param_53_bias = {param_53_bias_data, param_53_bias_shape, param_53_bias_dims};
  Tensor<float> param_54_weights = {param_54_weights_data, param_54_weights_shape, param_54_weights_dims};
  Tensor<float> param_55_bias = {param_55_bias_data, param_55_bias_shape, param_55_bias_dims};
  Tensor<float> param_56_weights = {param_56_weights_data, param_56_weights_shape, param_56_weights_dims};
  Tensor<float> param_57_bias = {param_57_bias_data, param_57_bias_shape, param_57_bias_dims};
  Tensor<float> param_58_gamma = {param_58_gamma_data, param_58_gamma_shape, param_58_gamma_dims};
  Tensor<float> param_59_beta = {param_59_beta_data, param_59_beta_shape, param_59_beta_dims};
  Tensor<float> param_60_weights = {param_60_weights_data, param_60_weights_shape, param_60_weights_dims};
  Tensor<float> param_61_bias = {param_61_bias_data, param_61_bias_shape, param_61_bias_dims};
  Tensor<float> param_62_weights = {param_62_weights_data, param_62_weights_shape, param_62_weights_dims};
  Tensor<float> param_63_bias = {param_63_bias_data, param_63_bias_shape, param_63_bias_dims};
  Tensor<float> param_64_gamma = {param_64_gamma_data, param_64_gamma_shape, param_64_gamma_dims};
  Tensor<float> param_65_beta = {param_65_beta_data, param_65_beta_shape, param_65_beta_dims};
  Tensor<float> param_66_weights = {param_66_weights_data, param_66_weights_shape, param_66_weights_dims};
  Tensor<float> param_67_bias = {param_67_bias_data, param_67_bias_shape, param_67_bias_dims};
  Tensor<float> param_68_weights = {param_68_weights_data, param_68_weights_shape, param_68_weights_dims};
  Tensor<float> param_69_bias = {param_69_bias_data, param_69_bias_shape, param_69_bias_dims};
  Tensor<float> param_70_weights = {param_70_weights_data, param_70_weights_shape, param_70_weights_dims};
  Tensor<float> param_71_bias = {param_71_bias_data, param_71_bias_shape, param_71_bias_dims};
  Tensor<float> param_72_weights = {param_72_weights_data, param_72_weights_shape, param_72_weights_dims};
  Tensor<float> param_73_bias = {param_73_bias_data, param_73_bias_shape, param_73_bias_dims};
  Tensor<float> param_74_gamma = {param_74_gamma_data, param_74_gamma_shape, param_74_gamma_dims};
  Tensor<float> param_75_beta = {param_75_beta_data, param_75_beta_shape, param_75_beta_dims};
  Tensor<float> param_76_weights = {param_76_weights_data, param_76_weights_shape, param_76_weights_dims};
  Tensor<float> param_77_bias = {param_77_bias_data, param_77_bias_shape, param_77_bias_dims};
  Tensor<float> param_78_weights = {param_78_weights_data, param_78_weights_shape, param_78_weights_dims};
  Tensor<float> param_79_bias = {param_79_bias_data, param_79_bias_shape, param_79_bias_dims};
  Tensor<float> param_80_gamma = {param_80_gamma_data, param_80_gamma_shape, param_80_gamma_dims};
  Tensor<float> param_81_beta = {param_81_beta_data, param_81_beta_shape, param_81_beta_dims};
  Tensor<float> param_82_weights = {param_82_weights_data, param_82_weights_shape, param_82_weights_dims};
  Tensor<float> param_83_bias = {param_83_bias_data, param_83_bias_shape, param_83_bias_dims};
  Tensor<float> param_84_weights = {param_84_weights_data, param_84_weights_shape, param_84_weights_dims};
  Tensor<float> param_85_bias = {param_85_bias_data, param_85_bias_shape, param_85_bias_dims};
  Tensor<float> param_86_weights = {param_86_weights_data, param_86_weights_shape, param_86_weights_dims};
  Tensor<float> param_87_bias = {param_87_bias_data, param_87_bias_shape, param_87_bias_dims};
  Tensor<float> param_88_weights = {param_88_weights_data, param_88_weights_shape, param_88_weights_dims};
  Tensor<float> param_89_bias = {param_89_bias_data, param_89_bias_shape, param_89_bias_dims};
  Tensor<float> param_90_gamma = {param_90_gamma_data, param_90_gamma_shape, param_90_gamma_dims};
  Tensor<float> param_91_beta = {param_91_beta_data, param_91_beta_shape, param_91_beta_dims};
  Tensor<float> param_92_weights = {param_92_weights_data, param_92_weights_shape, param_92_weights_dims};
  Tensor<float> param_93_bias = {param_93_bias_data, param_93_bias_shape, param_93_bias_dims};
  Tensor<float> param_94_weights = {param_94_weights_data, param_94_weights_shape, param_94_weights_dims};
  Tensor<float> param_95_bias = {param_95_bias_data, param_95_bias_shape, param_95_bias_dims};
  Tensor<float> param_96_gamma = {param_96_gamma_data, param_96_gamma_shape, param_96_gamma_dims};
  Tensor<float> param_97_beta = {param_97_beta_data, param_97_beta_shape, param_97_beta_dims};
  Tensor<float> param_98_weights = {param_98_weights_data, param_98_weights_shape, param_98_weights_dims};
  Tensor<float> param_99_bias = {param_99_bias_data, param_99_bias_shape, param_99_bias_dims};

 // --- Kernel Launch Sequence ---
  embedding_forward<<<dim3(128, 32), dim3(384, 1, 1), 0>>>(intermediate_0_tensor, input, param_0_embeddings);
  CUDA_CHECK(cudaGetLastError());
  positional_encoding_forward<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_1_tensor, intermediate_0_tensor, param_1_frequencies);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_2_tensor, intermediate_1_tensor, param_2_weights, param_3_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_3_tensor, intermediate_1_tensor, param_4_weights, param_5_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_4_tensor, intermediate_1_tensor, param_6_weights, param_7_bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 128, 32), dim3(64, 1, 1), 0>>>(intermediate_5_tensor, intermediate_2_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 128, 32), dim3(64, 1, 1), 0>>>(intermediate_6_tensor, intermediate_3_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 128, 32), dim3(64, 1, 1), 0>>>(intermediate_7_tensor, intermediate_4_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(16, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_8_tensor, intermediate_5_tensor, intermediate_6_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(128, 6, 32), dim3(128, 1, 1), 512>>>(intermediate_9_tensor, intermediate_8_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(8, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_10_tensor, intermediate_9_tensor, intermediate_7_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(128, 32), dim3(384, 1, 1), 0>>>(intermediate_11_tensor, intermediate_10_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_12_tensor, intermediate_11_tensor, param_8_weights, param_9_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(6144, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_13_tensor, intermediate_1_tensor, intermediate_12_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(128, 32), dim3(384, 1, 1), 1536>>>(intermediate_14_tensor, intermediate_13_tensor, param_10_gamma, param_11_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_15_tensor, intermediate_14_tensor, param_12_weights, param_13_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_16_tensor, intermediate_15_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_17_tensor, intermediate_16_tensor, param_14_weights, param_15_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(6144, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_18_tensor, intermediate_14_tensor, intermediate_17_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(128, 32), dim3(384, 1, 1), 1536>>>(intermediate_19_tensor, intermediate_18_tensor, param_16_gamma, param_17_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_20_tensor, intermediate_19_tensor, param_18_weights, param_19_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_21_tensor, intermediate_19_tensor, param_20_weights, param_21_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_22_tensor, intermediate_19_tensor, param_22_weights, param_23_bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_23_tensor, intermediate_20_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_24_tensor, intermediate_21_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_25_tensor, intermediate_22_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_26_tensor, intermediate_23_tensor, intermediate_24_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(intermediate_27_tensor, intermediate_26_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_28_tensor, intermediate_27_tensor, intermediate_25_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(intermediate_29_tensor, intermediate_28_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_30_tensor, intermediate_29_tensor, param_24_weights, param_25_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_31_tensor, intermediate_19_tensor, intermediate_30_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_32_tensor, intermediate_31_tensor, param_26_gamma, param_27_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_33_tensor, intermediate_32_tensor, param_28_weights, param_29_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_34_tensor, intermediate_33_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_35_tensor, intermediate_34_tensor, param_30_weights, param_31_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_36_tensor, intermediate_32_tensor, intermediate_35_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_37_tensor, intermediate_36_tensor, param_32_gamma, param_33_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_38_tensor, intermediate_37_tensor, param_34_weights, param_35_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_39_tensor, intermediate_37_tensor, param_36_weights, param_37_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_40_tensor, intermediate_37_tensor, param_38_weights, param_39_bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_41_tensor, intermediate_38_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_42_tensor, intermediate_39_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_43_tensor, intermediate_40_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_44_tensor, intermediate_41_tensor, intermediate_42_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(intermediate_45_tensor, intermediate_44_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_46_tensor, intermediate_45_tensor, intermediate_43_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(intermediate_47_tensor, intermediate_46_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_48_tensor, intermediate_47_tensor, param_40_weights, param_41_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_49_tensor, intermediate_37_tensor, intermediate_48_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_50_tensor, intermediate_49_tensor, param_42_gamma, param_43_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_51_tensor, intermediate_50_tensor, param_44_weights, param_45_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_52_tensor, intermediate_51_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_53_tensor, intermediate_52_tensor, param_46_weights, param_47_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_54_tensor, intermediate_50_tensor, intermediate_53_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_55_tensor, intermediate_54_tensor, param_48_gamma, param_49_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_56_tensor, intermediate_55_tensor, param_50_weights, param_51_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_57_tensor, intermediate_55_tensor, param_52_weights, param_53_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_58_tensor, intermediate_55_tensor, param_54_weights, param_55_bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_59_tensor, intermediate_56_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_60_tensor, intermediate_57_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_61_tensor, intermediate_58_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_62_tensor, intermediate_59_tensor, intermediate_60_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(intermediate_63_tensor, intermediate_62_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_64_tensor, intermediate_63_tensor, intermediate_61_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(intermediate_65_tensor, intermediate_64_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_66_tensor, intermediate_65_tensor, param_56_weights, param_57_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_67_tensor, intermediate_55_tensor, intermediate_66_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_68_tensor, intermediate_67_tensor, param_58_gamma, param_59_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_69_tensor, intermediate_68_tensor, param_60_weights, param_61_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_70_tensor, intermediate_69_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_71_tensor, intermediate_70_tensor, param_62_weights, param_63_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_72_tensor, intermediate_68_tensor, intermediate_71_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_73_tensor, intermediate_72_tensor, param_64_gamma, param_65_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_74_tensor, intermediate_73_tensor, param_66_weights, param_67_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_75_tensor, intermediate_73_tensor, param_68_weights, param_69_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_76_tensor, intermediate_73_tensor, param_70_weights, param_71_bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_77_tensor, intermediate_74_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_78_tensor, intermediate_75_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_79_tensor, intermediate_76_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_80_tensor, intermediate_77_tensor, intermediate_78_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(intermediate_81_tensor, intermediate_80_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_82_tensor, intermediate_81_tensor, intermediate_79_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(intermediate_83_tensor, intermediate_82_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_84_tensor, intermediate_83_tensor, param_72_weights, param_73_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_85_tensor, intermediate_73_tensor, intermediate_84_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_86_tensor, intermediate_85_tensor, param_74_gamma, param_75_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_87_tensor, intermediate_86_tensor, param_76_weights, param_77_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_88_tensor, intermediate_87_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_89_tensor, intermediate_88_tensor, param_78_weights, param_79_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_90_tensor, intermediate_86_tensor, intermediate_89_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_91_tensor, intermediate_90_tensor, param_80_gamma, param_81_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_92_tensor, intermediate_91_tensor, param_82_weights, param_83_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_93_tensor, intermediate_91_tensor, param_84_weights, param_85_bias);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_94_tensor, intermediate_91_tensor, param_86_weights, param_87_bias);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_95_tensor, intermediate_92_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_96_tensor, intermediate_93_tensor);
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(intermediate_97_tensor, intermediate_94_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_98_tensor, intermediate_95_tensor, intermediate_96_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(intermediate_99_tensor, intermediate_98_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(intermediate_100_tensor, intermediate_99_tensor, intermediate_97_tensor);
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(intermediate_101_tensor, intermediate_100_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_102_tensor, intermediate_101_tensor, param_88_weights, param_89_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_103_tensor, intermediate_91_tensor, intermediate_102_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_104_tensor, intermediate_103_tensor, param_90_gamma, param_91_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_105_tensor, intermediate_104_tensor, param_92_weights, param_93_bias);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_106_tensor, intermediate_105_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_107_tensor, intermediate_106_tensor, param_94_weights, param_95_bias);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_108_tensor, intermediate_104_tensor, intermediate_107_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_109_tensor, intermediate_108_tensor, param_96_gamma, param_97_beta);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(1, 32), dim3(256, 1, 1), 0>>>(intermediate_110_tensor, intermediate_109_tensor, param_98_weights, param_99_bias);
  CUDA_CHECK(cudaGetLastError());
  softmax_forward<<<dim3(32, 1, 1), dim3(96, 1, 1), 384>>>(output, intermediate_110_tensor);
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
   