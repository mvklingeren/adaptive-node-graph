
#include <cuda_runtime.h>
#include <cfloat>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


// Debug mode bounds checking (can be disabled for release builds)
#ifndef NDEBUG
#define TENSOR_BOUNDS_CHECK 1
#define TENSOR_BOUNDS_CHECK_VERBOSE 1
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
      printf("FATAL: Tensor bounds error: 3D access [%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d) for %dD tensor\n", 
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
      printf("FATAL: Tensor bounds error: 4D access [%d,%d,%d,%d] out of bounds [0,%d)x[0,%d)x[0,%d)x[0,%d) for %dD tensor\n", 
             i, j, k, l, shape[0], shape[1], shape[2], shape[3], dims);
      #endif
      #if defined(__CUDA_ARCH__)
      asm("trap;");
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
    inline size_t align_to_boundary(size_t size) const {
        const size_t alignment = 256;
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
public:
    WorkspaceAllocator(char* workspace, size_t workspace_size) 
        : base_ptr(workspace), current_offset(0), total_size(workspace_size) {}
    
    void* allocate(size_t size) {
        size_t aligned_size = align_to_boundary(size);
        
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
// Main Host-Side Execution Function (Array-Based Interface)
// ======================================================
extern "C" void executeGraph(
    // Input tensor
    void* input_data,
    const int* input_shape,
    int input_dims,
    const char* input_dtype,
    
    // Output tensor
    void* output_data,
    const int* output_shape,
    int output_dims,
    const char* output_dtype,
    
    // Parameters bundle
    void** param_data,
    const int** param_shapes,
    const int* param_dims,
    const char** param_dtypes,
    int param_count,
    
    // Workspace
    char* workspace,
    size_t workspace_size
) {
    // --- Input Validation ---
    if (!workspace) {
        fprintf(stderr, "Error: Null workspace pointer passed to executeGraph\n");
        return;
    }
    
    if (workspace_size < 124570829) {
        fprintf(stderr, "Error: Insufficient workspace size. Required: 124570829 bytes, Provided: %zu bytes\n", workspace_size);
        return;
    }
    
    if (param_count != 100) {
        fprintf(stderr, "Error: Expected 100 parameters, got %d\n", param_count);
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

    // --- Create Parameter Tensors ---
    Tensor<float> param_tensors[100];
    for (int i = 0; i < 100; i++) {
        param_tensors[i] = {
            (float*)param_data[i],
            param_shapes[i],
            param_dims[i]
        };
    }

    // --- Tensor Struct Instantiation ---
  // Pool: pool_cudanode_dc45077c-3602-4a05-b77b-b3e055f0557a:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_0_data = (float*)(workspace + 0);
  Tensor<float> intermediate_0_tensor = {intermediate_0_data, intermediate_0_shape, 3};
  // Pool: pool_cudanode_ed6b58c8-e737-4236-8847-9935769e9e48:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_1_data = (float*)(workspace + 0);
  Tensor<float> intermediate_1_tensor = {intermediate_1_data, intermediate_1_shape, 3};
  // Pool: pool_cudanode_481b5d11-a382-458b-83ce-7d3a4026acfe:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_2_data = (float*)(workspace + 0);
  Tensor<float> intermediate_2_tensor = {intermediate_2_data, intermediate_2_shape, 3};
  // Pool: pool_cudanode_7040dfc6-1638-45fd-872f-17638cb1eca6:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_3_data = (float*)(workspace + 0);
  Tensor<float> intermediate_3_tensor = {intermediate_3_data, intermediate_3_shape, 3};
  // Pool: pool_cudanode_afbac0ff-522f-4f64-965e-524aa347a7d3:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_4_data = (float*)(workspace + 0);
  Tensor<float> intermediate_4_tensor = {intermediate_4_data, intermediate_4_shape, 3};
  // Pool: pool_cudanode_99162404-52a3-47f2-860b-f4e281de8406:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_5_data = (float*)(workspace + 0);
  Tensor<float> intermediate_5_tensor = {intermediate_5_data, intermediate_5_shape, 4};
  // Pool: pool_cudanode_eb49f7e7-e490-4092-bffd-64dddf018251:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_6_data = (float*)(workspace + 0);
  Tensor<float> intermediate_6_tensor = {intermediate_6_data, intermediate_6_shape, 4};
  // Pool: pool_cudanode_57a4ecbb-24f5-46af-8337-29cec8d46b5c:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_7_data = (float*)(workspace + 0);
  Tensor<float> intermediate_7_tensor = {intermediate_7_data, intermediate_7_shape, 4};
  // Pool: pool_cudanode_79263bfd-4d77-40b7-a734-3fbe6c64766d:output, Offset: 0, Size: 12582912 bytes
  float* intermediate_8_data = (float*)(workspace + 0);
  Tensor<float> intermediate_8_tensor = {intermediate_8_data, intermediate_8_shape, 4};
  // Pool: pool_cudanode_5aeeeb2c-6093-4971-bc8c-9781b2548719:output, Offset: 0, Size: 12582912 bytes
  float* intermediate_9_data = (float*)(workspace + 0);
  Tensor<float> intermediate_9_tensor = {intermediate_9_data, intermediate_9_shape, 4};
  // Pool: pool_cudanode_ccac2b1e-36e7-4b06-960a-5328bc71f95b:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_10_data = (float*)(workspace + 0);
  Tensor<float> intermediate_10_tensor = {intermediate_10_data, intermediate_10_shape, 4};
  // Pool: pool_cudanode_2bfa0e39-8456-4eff-a423-1fd06a5e7ca7:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_11_data = (float*)(workspace + 0);
  Tensor<float> intermediate_11_tensor = {intermediate_11_data, intermediate_11_shape, 3};
  // Pool: pool_cudanode_8379cc95-3d4d-44e5-91fb-e18ce4d5723b:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_12_data = (float*)(workspace + 0);
  Tensor<float> intermediate_12_tensor = {intermediate_12_data, intermediate_12_shape, 3};
  // Pool: pool_cudanode_b760b99a-bcc2-4a1d-91fc-45e15b3111ce:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_13_data = (float*)(workspace + 0);
  Tensor<float> intermediate_13_tensor = {intermediate_13_data, intermediate_13_shape, 3};
  // Pool: pool_cudanode_3ae92399-7b2e-40c7-9af6-d8e4d0dd70f1:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_14_data = (float*)(workspace + 0);
  Tensor<float> intermediate_14_tensor = {intermediate_14_data, intermediate_14_shape, 3};
  // Pool: pool_cudanode_a4fcabb0-3a3d-444f-89a6-68d2e89abe92:output, Offset: 0, Size: 196608 bytes
  float* intermediate_15_data = (float*)(workspace + 0);
  Tensor<float> intermediate_15_tensor = {intermediate_15_data, intermediate_15_shape, 2};
  // Pool: pool_cudanode_09603499-47af-4309-b493-427210e5925f:output, Offset: 0, Size: 196608 bytes
  float* intermediate_16_data = (float*)(workspace + 0);
  Tensor<float> intermediate_16_tensor = {intermediate_16_data, intermediate_16_shape, 2};
  // Pool: pool_cudanode_bb8b115e-cfcb-4cdc-a37e-054f8dc8264a:output, Offset: 0, Size: 49152 bytes
  float* intermediate_17_data = (float*)(workspace + 0);
  Tensor<float> intermediate_17_tensor = {intermediate_17_data, intermediate_17_shape, 2};
  // Pool: pool_cudanode_f97e6687-9cdd-45a7-a1e7-2d2a27ccab97:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_18_data = (float*)(workspace + 0);
  Tensor<float> intermediate_18_tensor = {intermediate_18_data, intermediate_18_shape, 3};
  // Pool: pool_cudanode_2ca845d7-9888-4217-b14f-ca1f7bfb6263:output, Offset: 0, Size: 6291456 bytes
  float* intermediate_19_data = (float*)(workspace + 0);
  Tensor<float> intermediate_19_tensor = {intermediate_19_data, intermediate_19_shape, 3};
  // Pool: pool_cudanode_71545400-7d26-46cd-a503-bbe3c6fee4c9:output, Offset: 0, Size: 49152 bytes
  float* intermediate_20_data = (float*)(workspace + 0);
  Tensor<float> intermediate_20_tensor = {intermediate_20_data, intermediate_20_shape, 2};
  // Pool: pool_cudanode_d1960ecc-05a4-4bc6-9b74-8f6501d1194d:output, Offset: 0, Size: 49152 bytes
  float* intermediate_21_data = (float*)(workspace + 0);
  Tensor<float> intermediate_21_tensor = {intermediate_21_data, intermediate_21_shape, 2};
  // Pool: pool_cudanode_90f73823-9d31-45a4-b57e-8118d9cfe9a1:output, Offset: 0, Size: 49152 bytes
  float* intermediate_22_data = (float*)(workspace + 0);
  Tensor<float> intermediate_22_tensor = {intermediate_22_data, intermediate_22_shape, 2};
  // Pool: pool_cudanode_a1dfe4ab-9d25-4e3b-9c07-c8c2b99159c2:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_23_data = (float*)(workspace + 0);
  Tensor<float> intermediate_23_tensor = {intermediate_23_data, intermediate_23_shape, 4};
  // Pool: pool_cudanode_5da4f8b5-6623-4473-b9d5-eb0fcddf7352:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_24_data = (float*)(workspace + 0);
  Tensor<float> intermediate_24_tensor = {intermediate_24_data, intermediate_24_shape, 4};
  // Pool: pool_cudanode_0d3508a0-82e4-422a-91c1-9fc2f2cb97e3:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_25_data = (float*)(workspace + 0);
  Tensor<float> intermediate_25_tensor = {intermediate_25_data, intermediate_25_shape, 4};
  // Pool: pool_cudanode_1a1c07b7-9c21-4754-885e-af7a0b67c510:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_26_data = (float*)(workspace + 0);
  Tensor<float> intermediate_26_tensor = {intermediate_26_data, intermediate_26_shape, 4};
  // Pool: pool_cudanode_d9e82151-3e0d-4adf-8c60-7cd4e0eee001:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_27_data = (float*)(workspace + 0);
  Tensor<float> intermediate_27_tensor = {intermediate_27_data, intermediate_27_shape, 4};
  // Pool: pool_cudanode_6eaa8505-3ebb-433b-a54a-ab7604a6034e:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_28_data = (float*)(workspace + 0);
  Tensor<float> intermediate_28_tensor = {intermediate_28_data, intermediate_28_shape, 4};
  // Pool: pool_cudanode_de2d32e7-858a-49a4-9e0b-03836b497313:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_29_data = (float*)(workspace + 0);
  Tensor<float> intermediate_29_tensor = {intermediate_29_data, intermediate_29_shape, 3};
  // Pool: pool_cudanode_4832541a-a611-45d4-8e35-3bc541426e65:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_30_data = (float*)(workspace + 0);
  Tensor<float> intermediate_30_tensor = {intermediate_30_data, intermediate_30_shape, 3};
  // Pool: pool_cudanode_3e048527-2270-476d-bb44-d6cfc45341c1:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_31_data = (float*)(workspace + 0);
  Tensor<float> intermediate_31_tensor = {intermediate_31_data, intermediate_31_shape, 3};
  // Pool: pool_cudanode_8c85bbfa-1591-48a6-a6b4-64cce7d50540:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_32_data = (float*)(workspace + 0);
  Tensor<float> intermediate_32_tensor = {intermediate_32_data, intermediate_32_shape, 3};
  // Pool: pool_cudanode_86c7daf9-0938-402a-97fa-629bd9fca695:output, Offset: 0, Size: 196608 bytes
  float* intermediate_33_data = (float*)(workspace + 0);
  Tensor<float> intermediate_33_tensor = {intermediate_33_data, intermediate_33_shape, 2};
  // Pool: pool_cudanode_a875047b-494b-4987-b6ca-fc8fd6213fa4:output, Offset: 0, Size: 196608 bytes
  float* intermediate_34_data = (float*)(workspace + 0);
  Tensor<float> intermediate_34_tensor = {intermediate_34_data, intermediate_34_shape, 2};
  // Pool: pool_cudanode_437b3a37-2c5b-4ced-808f-7556a9a5a6cf:output, Offset: 0, Size: 49152 bytes
  float* intermediate_35_data = (float*)(workspace + 0);
  Tensor<float> intermediate_35_tensor = {intermediate_35_data, intermediate_35_shape, 2};
  // Pool: pool_cudanode_20868664-1582-476b-8add-b6f06fd13ad0:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_36_data = (float*)(workspace + 0);
  Tensor<float> intermediate_36_tensor = {intermediate_36_data, intermediate_36_shape, 3};
  // Pool: pool_cudanode_b8cbcf4b-9069-4f4c-aae4-d2fbafd43e5a:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_37_data = (float*)(workspace + 0);
  Tensor<float> intermediate_37_tensor = {intermediate_37_data, intermediate_37_shape, 3};
  // Pool: pool_cudanode_2bcf284c-d1d5-466c-ba3f-dd32227b67d2:output, Offset: 0, Size: 49152 bytes
  float* intermediate_38_data = (float*)(workspace + 0);
  Tensor<float> intermediate_38_tensor = {intermediate_38_data, intermediate_38_shape, 2};
  // Pool: pool_cudanode_0d4e3e44-5f80-4aa0-9e67-aadcf6da6036:output, Offset: 0, Size: 49152 bytes
  float* intermediate_39_data = (float*)(workspace + 0);
  Tensor<float> intermediate_39_tensor = {intermediate_39_data, intermediate_39_shape, 2};
  // Pool: pool_cudanode_40221395-8a82-4d91-abbd-89ce6fae75e0:output, Offset: 0, Size: 49152 bytes
  float* intermediate_40_data = (float*)(workspace + 0);
  Tensor<float> intermediate_40_tensor = {intermediate_40_data, intermediate_40_shape, 2};
  // Pool: pool_cudanode_5ccdc26f-cefe-4f51-9a33-44f61c250634:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_41_data = (float*)(workspace + 0);
  Tensor<float> intermediate_41_tensor = {intermediate_41_data, intermediate_41_shape, 4};
  // Pool: pool_cudanode_61483904-7363-4ba2-95b0-2f5e4673127b:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_42_data = (float*)(workspace + 0);
  Tensor<float> intermediate_42_tensor = {intermediate_42_data, intermediate_42_shape, 4};
  // Pool: pool_cudanode_1592309a-e5f9-4491-bead-af3041d9ba35:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_43_data = (float*)(workspace + 0);
  Tensor<float> intermediate_43_tensor = {intermediate_43_data, intermediate_43_shape, 4};
  // Pool: pool_cudanode_d6adaafc-1d5d-4a1d-a59f-fa218788fc22:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_44_data = (float*)(workspace + 0);
  Tensor<float> intermediate_44_tensor = {intermediate_44_data, intermediate_44_shape, 4};
  // Pool: pool_cudanode_e4d79f73-0f0d-4852-81c7-f61e93960e3a:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_45_data = (float*)(workspace + 0);
  Tensor<float> intermediate_45_tensor = {intermediate_45_data, intermediate_45_shape, 4};
  // Pool: pool_cudanode_27fb7057-4996-4e89-8ff6-3d09b78d48d5:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_46_data = (float*)(workspace + 0);
  Tensor<float> intermediate_46_tensor = {intermediate_46_data, intermediate_46_shape, 4};
  // Pool: pool_cudanode_034d324d-bc34-4502-ac06-cc9753377a8f:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_47_data = (float*)(workspace + 0);
  Tensor<float> intermediate_47_tensor = {intermediate_47_data, intermediate_47_shape, 3};
  // Pool: pool_cudanode_d2db1407-ea3b-4714-bf49-d3c73b2e2dd1:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_48_data = (float*)(workspace + 0);
  Tensor<float> intermediate_48_tensor = {intermediate_48_data, intermediate_48_shape, 3};
  // Pool: pool_cudanode_9985052e-4b73-4e11-8b73-30245011b167:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_49_data = (float*)(workspace + 0);
  Tensor<float> intermediate_49_tensor = {intermediate_49_data, intermediate_49_shape, 3};
  // Pool: pool_cudanode_2dd58af1-83e8-4f8d-afae-9d30d702a879:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_50_data = (float*)(workspace + 0);
  Tensor<float> intermediate_50_tensor = {intermediate_50_data, intermediate_50_shape, 3};
  // Pool: pool_cudanode_0a1a7956-4ebe-4d16-b481-9a2775ee838f:output, Offset: 0, Size: 196608 bytes
  float* intermediate_51_data = (float*)(workspace + 0);
  Tensor<float> intermediate_51_tensor = {intermediate_51_data, intermediate_51_shape, 2};
  // Pool: pool_cudanode_b4f6a03d-e308-4a26-bbd0-712468a8af4a:output, Offset: 0, Size: 196608 bytes
  float* intermediate_52_data = (float*)(workspace + 0);
  Tensor<float> intermediate_52_tensor = {intermediate_52_data, intermediate_52_shape, 2};
  // Pool: pool_cudanode_56f040e1-26a7-4018-9e76-26c831b9f69a:output, Offset: 0, Size: 49152 bytes
  float* intermediate_53_data = (float*)(workspace + 0);
  Tensor<float> intermediate_53_tensor = {intermediate_53_data, intermediate_53_shape, 2};
  // Pool: pool_cudanode_60675dd4-9566-4473-a8f9-bb6efa708d62:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_54_data = (float*)(workspace + 0);
  Tensor<float> intermediate_54_tensor = {intermediate_54_data, intermediate_54_shape, 3};
  // Pool: pool_cudanode_7ef24141-f886-4ac2-994c-2f6924b64391:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_55_data = (float*)(workspace + 0);
  Tensor<float> intermediate_55_tensor = {intermediate_55_data, intermediate_55_shape, 3};
  // Pool: pool_cudanode_982858f2-e853-4e96-8c75-3b34f5f23c51:output, Offset: 0, Size: 49152 bytes
  float* intermediate_56_data = (float*)(workspace + 0);
  Tensor<float> intermediate_56_tensor = {intermediate_56_data, intermediate_56_shape, 2};
  // Pool: pool_cudanode_bc1ef245-f9e4-4f21-9ec4-5fc6381a2291:output, Offset: 0, Size: 49152 bytes
  float* intermediate_57_data = (float*)(workspace + 0);
  Tensor<float> intermediate_57_tensor = {intermediate_57_data, intermediate_57_shape, 2};
  // Pool: pool_cudanode_7658d578-99e6-4f32-9a1a-bb4b12ac57f3:output, Offset: 0, Size: 49152 bytes
  float* intermediate_58_data = (float*)(workspace + 0);
  Tensor<float> intermediate_58_tensor = {intermediate_58_data, intermediate_58_shape, 2};
  // Pool: pool_cudanode_9fc56357-9c62-44df-84f5-3e9b831050e6:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_59_data = (float*)(workspace + 0);
  Tensor<float> intermediate_59_tensor = {intermediate_59_data, intermediate_59_shape, 4};
  // Pool: pool_cudanode_813e2fbe-b4db-41ec-be7f-3941e27b2527:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_60_data = (float*)(workspace + 0);
  Tensor<float> intermediate_60_tensor = {intermediate_60_data, intermediate_60_shape, 4};
  // Pool: pool_cudanode_99acc56a-1e14-419f-b6a1-5fcafc25df40:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_61_data = (float*)(workspace + 0);
  Tensor<float> intermediate_61_tensor = {intermediate_61_data, intermediate_61_shape, 4};
  // Pool: pool_cudanode_f7f82af9-4d9e-4531-9757-7e01ecda1665:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_62_data = (float*)(workspace + 0);
  Tensor<float> intermediate_62_tensor = {intermediate_62_data, intermediate_62_shape, 4};
  // Pool: pool_cudanode_b4f328b0-ded5-46b3-958a-c0ecc2904ec7:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_63_data = (float*)(workspace + 0);
  Tensor<float> intermediate_63_tensor = {intermediate_63_data, intermediate_63_shape, 4};
  // Pool: pool_cudanode_20aca3a4-d0d1-481a-9bb8-567390ddbf13:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_64_data = (float*)(workspace + 0);
  Tensor<float> intermediate_64_tensor = {intermediate_64_data, intermediate_64_shape, 4};
  // Pool: pool_cudanode_7a61ea36-ad9f-447b-ac34-8f40f53b7018:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_65_data = (float*)(workspace + 0);
  Tensor<float> intermediate_65_tensor = {intermediate_65_data, intermediate_65_shape, 3};
  // Pool: pool_cudanode_56c79d06-6905-4f8a-b68a-62bcc6ba0423:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_66_data = (float*)(workspace + 0);
  Tensor<float> intermediate_66_tensor = {intermediate_66_data, intermediate_66_shape, 3};
  // Pool: pool_cudanode_cc927b26-de3e-40ee-b33f-d8286cdb5518:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_67_data = (float*)(workspace + 0);
  Tensor<float> intermediate_67_tensor = {intermediate_67_data, intermediate_67_shape, 3};
  // Pool: pool_cudanode_5ea4759b-3060-44bd-ada8-5615c4e7bf03:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_68_data = (float*)(workspace + 0);
  Tensor<float> intermediate_68_tensor = {intermediate_68_data, intermediate_68_shape, 3};
  // Pool: pool_cudanode_9bde7445-f8e9-423c-8601-f3bc62450d3c:output, Offset: 0, Size: 196608 bytes
  float* intermediate_69_data = (float*)(workspace + 0);
  Tensor<float> intermediate_69_tensor = {intermediate_69_data, intermediate_69_shape, 2};
  // Pool: pool_cudanode_3f13d34f-1357-4391-811d-28581ec1ba35:output, Offset: 0, Size: 196608 bytes
  float* intermediate_70_data = (float*)(workspace + 0);
  Tensor<float> intermediate_70_tensor = {intermediate_70_data, intermediate_70_shape, 2};
  // Pool: pool_cudanode_56fbcac4-d8ed-4611-8ed8-19f5b025715b:output, Offset: 0, Size: 49152 bytes
  float* intermediate_71_data = (float*)(workspace + 0);
  Tensor<float> intermediate_71_tensor = {intermediate_71_data, intermediate_71_shape, 2};
  // Pool: pool_cudanode_7dd89a92-bf0f-4b91-8858-8b7771824ec6:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_72_data = (float*)(workspace + 0);
  Tensor<float> intermediate_72_tensor = {intermediate_72_data, intermediate_72_shape, 3};
  // Pool: pool_cudanode_a212482c-efca-4090-a7ee-1757090a1bf6:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_73_data = (float*)(workspace + 0);
  Tensor<float> intermediate_73_tensor = {intermediate_73_data, intermediate_73_shape, 3};
  // Pool: pool_cudanode_f94373a2-931e-4753-9ce2-fe76aa8e5e32:output, Offset: 0, Size: 49152 bytes
  float* intermediate_74_data = (float*)(workspace + 0);
  Tensor<float> intermediate_74_tensor = {intermediate_74_data, intermediate_74_shape, 2};
  // Pool: pool_cudanode_d8f950c7-379c-4f63-b795-e6fc2a647205:output, Offset: 0, Size: 49152 bytes
  float* intermediate_75_data = (float*)(workspace + 0);
  Tensor<float> intermediate_75_tensor = {intermediate_75_data, intermediate_75_shape, 2};
  // Pool: pool_cudanode_7b8bff37-0a7d-4534-91fc-91ed3f8b9487:output, Offset: 0, Size: 49152 bytes
  float* intermediate_76_data = (float*)(workspace + 0);
  Tensor<float> intermediate_76_tensor = {intermediate_76_data, intermediate_76_shape, 2};
  // Pool: pool_cudanode_5efb835e-49ae-4122-a23e-4f1d70295cd3:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_77_data = (float*)(workspace + 0);
  Tensor<float> intermediate_77_tensor = {intermediate_77_data, intermediate_77_shape, 4};
  // Pool: pool_cudanode_1da0f499-478d-4466-bf4e-f6cc5984c84e:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_78_data = (float*)(workspace + 0);
  Tensor<float> intermediate_78_tensor = {intermediate_78_data, intermediate_78_shape, 4};
  // Pool: pool_cudanode_d7a0afbb-0330-46b2-aa14-54e5bfdcb90b:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_79_data = (float*)(workspace + 0);
  Tensor<float> intermediate_79_tensor = {intermediate_79_data, intermediate_79_shape, 4};
  // Pool: pool_cudanode_45f08651-6a82-4808-a257-c647ee9a2eb2:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_80_data = (float*)(workspace + 0);
  Tensor<float> intermediate_80_tensor = {intermediate_80_data, intermediate_80_shape, 4};
  // Pool: pool_cudanode_8f6de94a-2309-44c5-ad33-d66f6a55ecb0:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_81_data = (float*)(workspace + 0);
  Tensor<float> intermediate_81_tensor = {intermediate_81_data, intermediate_81_shape, 4};
  // Pool: pool_cudanode_eff5c967-1fe5-4e23-bd0d-26663e1a3f59:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_82_data = (float*)(workspace + 0);
  Tensor<float> intermediate_82_tensor = {intermediate_82_data, intermediate_82_shape, 4};
  // Pool: pool_cudanode_a7a47dbb-4700-4bb2-b547-d9a3e19efb83:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_83_data = (float*)(workspace + 0);
  Tensor<float> intermediate_83_tensor = {intermediate_83_data, intermediate_83_shape, 3};
  // Pool: pool_cudanode_21731677-1945-4023-91a0-3029c101d2c6:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_84_data = (float*)(workspace + 0);
  Tensor<float> intermediate_84_tensor = {intermediate_84_data, intermediate_84_shape, 3};
  // Pool: pool_cudanode_366d0d1d-8254-4f6f-b787-4db47324c1ce:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_85_data = (float*)(workspace + 0);
  Tensor<float> intermediate_85_tensor = {intermediate_85_data, intermediate_85_shape, 3};
  // Pool: pool_cudanode_0078b306-138f-4ba8-aa74-6cf44d20d85e:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_86_data = (float*)(workspace + 0);
  Tensor<float> intermediate_86_tensor = {intermediate_86_data, intermediate_86_shape, 3};
  // Pool: pool_cudanode_62dc0ad6-45e3-40ea-9429-2120e033cea5:output, Offset: 0, Size: 196608 bytes
  float* intermediate_87_data = (float*)(workspace + 0);
  Tensor<float> intermediate_87_tensor = {intermediate_87_data, intermediate_87_shape, 2};
  // Pool: pool_cudanode_eaa9da97-af54-4106-9f28-e7543010dd9d:output, Offset: 0, Size: 196608 bytes
  float* intermediate_88_data = (float*)(workspace + 0);
  Tensor<float> intermediate_88_tensor = {intermediate_88_data, intermediate_88_shape, 2};
  // Pool: pool_cudanode_69477c03-47cb-4b45-b0e1-702a97572ca6:output, Offset: 0, Size: 49152 bytes
  float* intermediate_89_data = (float*)(workspace + 0);
  Tensor<float> intermediate_89_tensor = {intermediate_89_data, intermediate_89_shape, 2};
  // Pool: pool_cudanode_db532a5d-ebd7-42c2-a25a-e7f92621ca7e:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_90_data = (float*)(workspace + 0);
  Tensor<float> intermediate_90_tensor = {intermediate_90_data, intermediate_90_shape, 3};
  // Pool: pool_cudanode_617e3394-ebca-490d-8ca9-799252ca7f97:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_91_data = (float*)(workspace + 0);
  Tensor<float> intermediate_91_tensor = {intermediate_91_data, intermediate_91_shape, 3};
  // Pool: pool_cudanode_edcc2015-0903-42dd-8a80-aa27fda1265e:output, Offset: 0, Size: 49152 bytes
  float* intermediate_92_data = (float*)(workspace + 0);
  Tensor<float> intermediate_92_tensor = {intermediate_92_data, intermediate_92_shape, 2};
  // Pool: pool_cudanode_1034cd9c-a2b2-40d8-8487-c8e2b0261df4:output, Offset: 0, Size: 49152 bytes
  float* intermediate_93_data = (float*)(workspace + 0);
  Tensor<float> intermediate_93_tensor = {intermediate_93_data, intermediate_93_shape, 2};
  // Pool: pool_cudanode_e943ea66-270b-4ee2-afee-cea4e4e38bb2:output, Offset: 0, Size: 49152 bytes
  float* intermediate_94_data = (float*)(workspace + 0);
  Tensor<float> intermediate_94_tensor = {intermediate_94_data, intermediate_94_shape, 2};
  // Pool: pool_cudanode_4800b006-a5b6-4ef0-9d45-b0565b820b2d:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_95_data = (float*)(workspace + 0);
  Tensor<float> intermediate_95_tensor = {intermediate_95_data, intermediate_95_shape, 4};
  // Pool: pool_cudanode_ed783b70-cd89-4684-afa9-d21cd4243a88:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_96_data = (float*)(workspace + 0);
  Tensor<float> intermediate_96_tensor = {intermediate_96_data, intermediate_96_shape, 4};
  // Pool: pool_cudanode_8b398ceb-af27-4e53-a416-38e5ae597956:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_97_data = (float*)(workspace + 0);
  Tensor<float> intermediate_97_tensor = {intermediate_97_data, intermediate_97_shape, 4};
  // Pool: pool_cudanode_e3a8abb9-1cd7-4dbd-a6bd-896a48d6446a:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_98_data = (float*)(workspace + 0);
  Tensor<float> intermediate_98_tensor = {intermediate_98_data, intermediate_98_shape, 4};
  // Pool: pool_cudanode_d264aaf0-3ccf-40ef-afcb-6bd86ebfe0da:output, Offset: 0, Size: 113246208 bytes
  float* intermediate_99_data = (float*)(workspace + 0);
  Tensor<float> intermediate_99_tensor = {intermediate_99_data, intermediate_99_shape, 4};
  // Pool: pool_cudanode_9a3f8586-a7ad-4f93-b363-d53afb468e1b:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_100_data = (float*)(workspace + 0);
  Tensor<float> intermediate_100_tensor = {intermediate_100_data, intermediate_100_shape, 4};
  // Pool: pool_cudanode_53ea9b55-6a6a-429d-a1e1-dcae866f398c:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_101_data = (float*)(workspace + 0);
  Tensor<float> intermediate_101_tensor = {intermediate_101_data, intermediate_101_shape, 3};
  // Pool: pool_cudanode_977af48d-6ea1-4f95-871b-cc5143991b2a:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_102_data = (float*)(workspace + 0);
  Tensor<float> intermediate_102_tensor = {intermediate_102_data, intermediate_102_shape, 3};
  // Pool: pool_cudanode_4049a539-39d8-4ecc-b0c7-82db1c2fdd23:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_103_data = (float*)(workspace + 0);
  Tensor<float> intermediate_103_tensor = {intermediate_103_data, intermediate_103_shape, 3};
  // Pool: pool_cudanode_fec0860f-8772-4597-b942-ccd804284ee8:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_104_data = (float*)(workspace + 0);
  Tensor<float> intermediate_104_tensor = {intermediate_104_data, intermediate_104_shape, 3};
  // Pool: pool_cudanode_9da5c968-5c0a-4839-9c13-9bffeab11ced:output, Offset: 0, Size: 196608 bytes
  float* intermediate_105_data = (float*)(workspace + 0);
  Tensor<float> intermediate_105_tensor = {intermediate_105_data, intermediate_105_shape, 2};
  // Pool: pool_cudanode_212ee999-c188-490b-b57c-795acc810c7a:output, Offset: 0, Size: 196608 bytes
  float* intermediate_106_data = (float*)(workspace + 0);
  Tensor<float> intermediate_106_tensor = {intermediate_106_data, intermediate_106_shape, 2};
  // Pool: pool_cudanode_f411fce1-e91d-4c30-8bdd-14a6d33b1047:output, Offset: 0, Size: 49152 bytes
  float* intermediate_107_data = (float*)(workspace + 0);
  Tensor<float> intermediate_107_tensor = {intermediate_107_data, intermediate_107_shape, 2};
  // Pool: pool_cudanode_f8a60b97-e190-4947-b9de-015c631b7c60:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_108_data = (float*)(workspace + 0);
  Tensor<float> intermediate_108_tensor = {intermediate_108_data, intermediate_108_shape, 3};
  // Pool: pool_cudanode_e8583053-04e7-41c7-8432-c1ac7a7011da:output, Offset: 0, Size: 18874368 bytes
  float* intermediate_109_data = (float*)(workspace + 0);
  Tensor<float> intermediate_109_tensor = {intermediate_109_data, intermediate_109_shape, 3};
  // Pool: pool_cudanode_ecb830b9-4a5a-47fa-b5b1-81bfcc5f0e22:output, Offset: 0, Size: 8320 bytes
  float* intermediate_110_data = (float*)(workspace + 0);
  Tensor<float> intermediate_110_tensor = {intermediate_110_data, intermediate_110_shape, 2};
  Tensor<int> input = {(int*)input_data, input_shape, input_dims};
  Tensor<float> output = {(float*)output_data, output_shape, output_dims};

    // --- Kernel Launch Sequence ---
  embedding_forward<<<dim3(128, 32), dim3(384, 1, 1), 0>>>(embedding_forward(intermediate_0_tensor, input, param_tensors[0]););
  CUDA_CHECK(cudaGetLastError());
  positional_encoding_forward<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(positional_encoding_forward(intermediate_1_tensor, intermediate_0_tensor, param_tensors[1]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_2_tensor, intermediate_1_tensor, param_tensors[2], param_tensors[3]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_3_tensor, intermediate_1_tensor, param_tensors[4], param_tensors[5]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_4_tensor, intermediate_1_tensor, param_tensors[6], param_tensors[7]););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 128, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_5_tensor, intermediate_2_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 128, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_6_tensor, intermediate_3_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 128, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_7_tensor, intermediate_4_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(16, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_transpose_b_tiled(intermediate_8_tensor, intermediate_5_tensor, intermediate_6_tensor););
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(128, 6, 32), dim3(128, 1, 1), 512>>>(fused_scale_softmax_forward(intermediate_9_tensor, intermediate_8_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(8, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_tiled(intermediate_10_tensor, intermediate_9_tensor, intermediate_7_tensor););
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(128, 32), dim3(384, 1, 1), 0>>>(concat_heads_forward(intermediate_11_tensor, intermediate_10_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_12_tensor, intermediate_11_tensor, param_tensors[8], param_tensors[9]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(6144, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_13_tensor, intermediate_1_tensor, intermediate_12_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(128, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_14_tensor, intermediate_13_tensor, param_tensors[10], param_tensors[11]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_15_tensor, intermediate_14_tensor, param_tensors[12], param_tensors[13]););
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(relu_forward(intermediate_16_tensor, intermediate_15_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_17_tensor, intermediate_16_tensor, param_tensors[14], param_tensors[15]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(6144, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_18_tensor, intermediate_14_tensor, intermediate_17_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(128, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_19_tensor, intermediate_18_tensor, param_tensors[16], param_tensors[17]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_20_tensor, intermediate_19_tensor, param_tensors[18], param_tensors[19]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_21_tensor, intermediate_19_tensor, param_tensors[20], param_tensors[21]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_22_tensor, intermediate_19_tensor, param_tensors[22], param_tensors[23]););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_23_tensor, intermediate_20_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_24_tensor, intermediate_21_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_25_tensor, intermediate_22_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_transpose_b_tiled(intermediate_26_tensor, intermediate_23_tensor, intermediate_24_tensor););
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(fused_scale_softmax_forward(intermediate_27_tensor, intermediate_26_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_tiled(intermediate_28_tensor, intermediate_27_tensor, intermediate_25_tensor););
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(concat_heads_forward(intermediate_29_tensor, intermediate_28_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_30_tensor, intermediate_29_tensor, param_tensors[24], param_tensors[25]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_31_tensor, intermediate_19_tensor, intermediate_30_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_32_tensor, intermediate_31_tensor, param_tensors[26], param_tensors[27]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_33_tensor, intermediate_32_tensor, param_tensors[28], param_tensors[29]););
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(relu_forward(intermediate_34_tensor, intermediate_33_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_35_tensor, intermediate_34_tensor, param_tensors[30], param_tensors[31]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_36_tensor, intermediate_32_tensor, intermediate_35_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_37_tensor, intermediate_36_tensor, param_tensors[32], param_tensors[33]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_38_tensor, intermediate_37_tensor, param_tensors[34], param_tensors[35]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_39_tensor, intermediate_37_tensor, param_tensors[36], param_tensors[37]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_40_tensor, intermediate_37_tensor, param_tensors[38], param_tensors[39]););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_41_tensor, intermediate_38_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_42_tensor, intermediate_39_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_43_tensor, intermediate_40_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_transpose_b_tiled(intermediate_44_tensor, intermediate_41_tensor, intermediate_42_tensor););
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(fused_scale_softmax_forward(intermediate_45_tensor, intermediate_44_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_tiled(intermediate_46_tensor, intermediate_45_tensor, intermediate_43_tensor););
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(concat_heads_forward(intermediate_47_tensor, intermediate_46_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_48_tensor, intermediate_47_tensor, param_tensors[40], param_tensors[41]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_49_tensor, intermediate_37_tensor, intermediate_48_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_50_tensor, intermediate_49_tensor, param_tensors[42], param_tensors[43]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_51_tensor, intermediate_50_tensor, param_tensors[44], param_tensors[45]););
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(relu_forward(intermediate_52_tensor, intermediate_51_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_53_tensor, intermediate_52_tensor, param_tensors[46], param_tensors[47]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_54_tensor, intermediate_50_tensor, intermediate_53_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_55_tensor, intermediate_54_tensor, param_tensors[48], param_tensors[49]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_56_tensor, intermediate_55_tensor, param_tensors[50], param_tensors[51]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_57_tensor, intermediate_55_tensor, param_tensors[52], param_tensors[53]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_58_tensor, intermediate_55_tensor, param_tensors[54], param_tensors[55]););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_59_tensor, intermediate_56_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_60_tensor, intermediate_57_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_61_tensor, intermediate_58_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_transpose_b_tiled(intermediate_62_tensor, intermediate_59_tensor, intermediate_60_tensor););
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(fused_scale_softmax_forward(intermediate_63_tensor, intermediate_62_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_tiled(intermediate_64_tensor, intermediate_63_tensor, intermediate_61_tensor););
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(concat_heads_forward(intermediate_65_tensor, intermediate_64_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_66_tensor, intermediate_65_tensor, param_tensors[56], param_tensors[57]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_67_tensor, intermediate_55_tensor, intermediate_66_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_68_tensor, intermediate_67_tensor, param_tensors[58], param_tensors[59]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_69_tensor, intermediate_68_tensor, param_tensors[60], param_tensors[61]););
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(relu_forward(intermediate_70_tensor, intermediate_69_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_71_tensor, intermediate_70_tensor, param_tensors[62], param_tensors[63]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_72_tensor, intermediate_68_tensor, intermediate_71_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_73_tensor, intermediate_72_tensor, param_tensors[64], param_tensors[65]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_74_tensor, intermediate_73_tensor, param_tensors[66], param_tensors[67]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_75_tensor, intermediate_73_tensor, param_tensors[68], param_tensors[69]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_76_tensor, intermediate_73_tensor, param_tensors[70], param_tensors[71]););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_77_tensor, intermediate_74_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_78_tensor, intermediate_75_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_79_tensor, intermediate_76_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_transpose_b_tiled(intermediate_80_tensor, intermediate_77_tensor, intermediate_78_tensor););
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(fused_scale_softmax_forward(intermediate_81_tensor, intermediate_80_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_tiled(intermediate_82_tensor, intermediate_81_tensor, intermediate_79_tensor););
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(concat_heads_forward(intermediate_83_tensor, intermediate_82_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_84_tensor, intermediate_83_tensor, param_tensors[72], param_tensors[73]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_85_tensor, intermediate_73_tensor, intermediate_84_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_86_tensor, intermediate_85_tensor, param_tensors[74], param_tensors[75]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_87_tensor, intermediate_86_tensor, param_tensors[76], param_tensors[77]););
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(relu_forward(intermediate_88_tensor, intermediate_87_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_89_tensor, intermediate_88_tensor, param_tensors[78], param_tensors[79]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_90_tensor, intermediate_86_tensor, intermediate_89_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_91_tensor, intermediate_90_tensor, param_tensors[80], param_tensors[81]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_92_tensor, intermediate_91_tensor, param_tensors[82], param_tensors[83]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_93_tensor, intermediate_91_tensor, param_tensors[84], param_tensors[85]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_94_tensor, intermediate_91_tensor, param_tensors[86], param_tensors[87]););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_95_tensor, intermediate_92_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_96_tensor, intermediate_93_tensor););
  CUDA_CHECK(cudaGetLastError());
  split_heads_forward<<<dim3(6, 384, 32), dim3(64, 1, 1), 0>>>(split_heads_forward(intermediate_97_tensor, intermediate_94_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_transpose_b_tiled<<<dim3(144, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_transpose_b_tiled(intermediate_98_tensor, intermediate_95_tensor, intermediate_96_tensor););
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(384, 6, 32), dim3(384, 1, 1), 1536>>>(fused_scale_softmax_forward(intermediate_99_tensor, intermediate_98_tensor););
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(24, 6, 32), dim3(32, 32, 1), 0>>>(batched_matmul_tiled(intermediate_100_tensor, intermediate_99_tensor, intermediate_97_tensor););
  CUDA_CHECK(cudaGetLastError());
  concat_heads_forward<<<dim3(384, 32), dim3(384, 1, 1), 0>>>(concat_heads_forward(intermediate_101_tensor, intermediate_100_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(dense_forward_3d(intermediate_102_tensor, intermediate_101_tensor, param_tensors[88], param_tensors[89]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_103_tensor, intermediate_91_tensor, intermediate_102_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_104_tensor, intermediate_103_tensor, param_tensors[90], param_tensors[91]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_105_tensor, intermediate_104_tensor, param_tensors[92], param_tensors[93]););
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(relu_forward(intermediate_106_tensor, intermediate_105_tensor););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_107_tensor, intermediate_106_tensor, param_tensors[94], param_tensors[95]););
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(add_forward(intermediate_108_tensor, intermediate_104_tensor, intermediate_107_tensor););
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(layer_norm_forward(intermediate_109_tensor, intermediate_108_tensor, param_tensors[96], param_tensors[97]););
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(1, 32), dim3(256, 1, 1), 0>>>(dense_forward_2d(intermediate_110_tensor, intermediate_109_tensor, param_tensors[98], param_tensors[99]););
  CUDA_CHECK(cudaGetLastError());
  softmax_forward<<<dim3(32, 1, 1), dim3(96, 1, 1), 384>>>(softmax_forward(output, intermediate_110_tensor););
  CUDA_CHECK(cudaGetLastError());
    
    // --- Synchronization for completion verification ---
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // --- Report memory usage ---
    size_t used_memory = allocator.get_used_size();
    if (used_memory > 0) {
        // Uncomment for debugging memory usage
        // printf("Workspace memory used: %zu bytes (%.2f MB)\n", used_memory, used_memory / (1024.0 * 1024.0));
    }
}

// ======================================================
// Simplified C-style Interface (for easier FFI)
// ======================================================
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
    // Fixed shapes for this specific model
    const int input_shape[] = {input_batch_size, input_seq_len};
    const int output_shape[] = {output_batch_size, output_vocab_size};
    
    // Parameter shapes are hardcoded for this specific model configuration
        const int param_shape_0[] = {65, 384}; // param_0_embeddings\n    const int param_shape_1[] = {192}; // param_1_frequencies\n    const int param_shape_2[] = {384, 384}; // param_2_weights\n    const int param_shape_3[] = {384}; // param_3_bias\n    const int param_shape_4[] = {384, 384}; // param_4_weights\n    const int param_shape_5[] = {384}; // param_5_bias\n    const int param_shape_6[] = {384, 384}; // param_6_weights\n    const int param_shape_7[] = {384}; // param_7_bias\n    const int param_shape_8[] = {384, 384}; // param_8_weights\n    const int param_shape_9[] = {384}; // param_9_bias\n    const int param_shape_10[] = {384}; // param_10_gamma\n    const int param_shape_11[] = {384}; // param_11_beta\n    const int param_shape_12[] = {384, 384}; // param_12_weights\n    const int param_shape_13[] = {384}; // param_13_bias\n    const int param_shape_14[] = {384, 384}; // param_14_weights\n    const int param_shape_15[] = {384}; // param_15_bias\n    const int param_shape_16[] = {384}; // param_16_gamma\n    const int param_shape_17[] = {384}; // param_17_beta\n    const int param_shape_18[] = {384, 384}; // param_18_weights\n    const int param_shape_19[] = {384}; // param_19_bias\n    const int param_shape_20[] = {384, 384}; // param_20_weights\n    const int param_shape_21[] = {384}; // param_21_bias\n    const int param_shape_22[] = {384, 384}; // param_22_weights\n    const int param_shape_23[] = {384}; // param_23_bias\n    const int param_shape_24[] = {384, 384}; // param_24_weights\n    const int param_shape_25[] = {384}; // param_25_bias\n    const int param_shape_26[] = {384}; // param_26_gamma\n    const int param_shape_27[] = {384}; // param_27_beta\n    const int param_shape_28[] = {384, 384}; // param_28_weights\n    const int param_shape_29[] = {384}; // param_29_bias\n    const int param_shape_30[] = {384, 384}; // param_30_weights\n    const int param_shape_31[] = {384}; // param_31_bias\n    const int param_shape_32[] = {384}; // param_32_gamma\n    const int param_shape_33[] = {384}; // param_33_beta\n    const int param_shape_34[] = {384, 384}; // param_34_weights\n    const int param_shape_35[] = {384}; // param_35_bias\n    const int param_shape_36[] = {384, 384}; // param_36_weights\n    const int param_shape_37[] = {384}; // param_37_bias\n    const int param_shape_38[] = {384, 384}; // param_38_weights\n    const int param_shape_39[] = {384}; // param_39_bias\n    const int param_shape_40[] = {384, 384}; // param_40_weights\n    const int param_shape_41[] = {384}; // param_41_bias\n    const int param_shape_42[] = {384}; // param_42_gamma\n    const int param_shape_43[] = {384}; // param_43_beta\n    const int param_shape_44[] = {384, 384}; // param_44_weights\n    const int param_shape_45[] = {384}; // param_45_bias\n    const int param_shape_46[] = {384, 384}; // param_46_weights\n    const int param_shape_47[] = {384}; // param_47_bias\n    const int param_shape_48[] = {384}; // param_48_gamma\n    const int param_shape_49[] = {384}; // param_49_beta\n    const int param_shape_50[] = {384, 384}; // param_50_weights\n    const int param_shape_51[] = {384}; // param_51_bias\n    const int param_shape_52[] = {384, 384}; // param_52_weights\n    const int param_shape_53[] = {384}; // param_53_bias\n    const int param_shape_54[] = {384, 384}; // param_54_weights\n    const int param_shape_55[] = {384}; // param_55_bias\n    const int param_shape_56[] = {384, 384}; // param_56_weights\n    const int param_shape_57[] = {384}; // param_57_bias\n    const int param_shape_58[] = {384}; // param_58_gamma\n    const int param_shape_59[] = {384}; // param_59_beta\n    const int param_shape_60[] = {384, 384}; // param_60_weights\n    const int param_shape_61[] = {384}; // param_61_bias\n    const int param_shape_62[] = {384, 384}; // param_62_weights\n    const int param_shape_63[] = {384}; // param_63_bias\n    const int param_shape_64[] = {384}; // param_64_gamma\n    const int param_shape_65[] = {384}; // param_65_beta\n    const int param_shape_66[] = {384, 384}; // param_66_weights\n    const int param_shape_67[] = {384}; // param_67_bias\n    const int param_shape_68[] = {384, 384}; // param_68_weights\n    const int param_shape_69[] = {384}; // param_69_bias\n    const int param_shape_70[] = {384, 384}; // param_70_weights\n    const int param_shape_71[] = {384}; // param_71_bias\n    const int param_shape_72[] = {384, 384}; // param_72_weights\n    const int param_shape_73[] = {384}; // param_73_bias\n    const int param_shape_74[] = {384}; // param_74_gamma\n    const int param_shape_75[] = {384}; // param_75_beta\n    const int param_shape_76[] = {384, 384}; // param_76_weights\n    const int param_shape_77[] = {384}; // param_77_bias\n    const int param_shape_78[] = {384, 384}; // param_78_weights\n    const int param_shape_79[] = {384}; // param_79_bias\n    const int param_shape_80[] = {384}; // param_80_gamma\n    const int param_shape_81[] = {384}; // param_81_beta\n    const int param_shape_82[] = {384, 384}; // param_82_weights\n    const int param_shape_83[] = {384}; // param_83_bias\n    const int param_shape_84[] = {384, 384}; // param_84_weights\n    const int param_shape_85[] = {384}; // param_85_bias\n    const int param_shape_86[] = {384, 384}; // param_86_weights\n    const int param_shape_87[] = {384}; // param_87_bias\n    const int param_shape_88[] = {384, 384}; // param_88_weights\n    const int param_shape_89[] = {384}; // param_89_bias\n    const int param_shape_90[] = {384}; // param_90_gamma\n    const int param_shape_91[] = {384}; // param_91_beta\n    const int param_shape_92[] = {384, 384}; // param_92_weights\n    const int param_shape_93[] = {384}; // param_93_bias\n    const int param_shape_94[] = {384, 384}; // param_94_weights\n    const int param_shape_95[] = {384}; // param_95_bias\n    const int param_shape_96[] = {384}; // param_96_gamma\n    const int param_shape_97[] = {384}; // param_97_beta\n    const int param_shape_98[] = {384, 384}; // param_98_weights\n    const int param_shape_99[] = {384}; // param_99_bias
    
    const int* param_shapes[100] = {
        param_shape_0,\n        param_shape_1,\n        param_shape_2,\n        param_shape_3,\n        param_shape_4,\n        param_shape_5,\n        param_shape_6,\n        param_shape_7,\n        param_shape_8,\n        param_shape_9,\n        param_shape_10,\n        param_shape_11,\n        param_shape_12,\n        param_shape_13,\n        param_shape_14,\n        param_shape_15,\n        param_shape_16,\n        param_shape_17,\n        param_shape_18,\n        param_shape_19,\n        param_shape_20,\n        param_shape_21,\n        param_shape_22,\n        param_shape_23,\n        param_shape_24,\n        param_shape_25,\n        param_shape_26,\n        param_shape_27,\n        param_shape_28,\n        param_shape_29,\n        param_shape_30,\n        param_shape_31,\n        param_shape_32,\n        param_shape_33,\n        param_shape_34,\n        param_shape_35,\n        param_shape_36,\n        param_shape_37,\n        param_shape_38,\n        param_shape_39,\n        param_shape_40,\n        param_shape_41,\n        param_shape_42,\n        param_shape_43,\n        param_shape_44,\n        param_shape_45,\n        param_shape_46,\n        param_shape_47,\n        param_shape_48,\n        param_shape_49,\n        param_shape_50,\n        param_shape_51,\n        param_shape_52,\n        param_shape_53,\n        param_shape_54,\n        param_shape_55,\n        param_shape_56,\n        param_shape_57,\n        param_shape_58,\n        param_shape_59,\n        param_shape_60,\n        param_shape_61,\n        param_shape_62,\n        param_shape_63,\n        param_shape_64,\n        param_shape_65,\n        param_shape_66,\n        param_shape_67,\n        param_shape_68,\n        param_shape_69,\n        param_shape_70,\n        param_shape_71,\n        param_shape_72,\n        param_shape_73,\n        param_shape_74,\n        param_shape_75,\n        param_shape_76,\n        param_shape_77,\n        param_shape_78,\n        param_shape_79,\n        param_shape_80,\n        param_shape_81,\n        param_shape_82,\n        param_shape_83,\n        param_shape_84,\n        param_shape_85,\n        param_shape_86,\n        param_shape_87,\n        param_shape_88,\n        param_shape_89,\n        param_shape_90,\n        param_shape_91,\n        param_shape_92,\n        param_shape_93,\n        param_shape_94,\n        param_shape_95,\n        param_shape_96,\n        param_shape_97,\n        param_shape_98,\n        param_shape_99
    };
    
    const int param_dims[100] = {
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1
    };
    
    const char* param_dtypes[100] = {
        "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32"
    };
    
    executeGraph(
        input_data, input_shape, 2, "int32",
        output_data, output_shape, 2, "float32",
        param_data, param_shapes, param_dims, param_dtypes, 100,
        workspace, workspace_size
    );
}
