
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
        
        // Calculate total size dynamically based on tensor dimensions
        int size = a.total_elements();
        
        // Handle broadcasting: if b has fewer elements, broadcast it
        int b_size = b.total_elements();
        bool broadcast_b = (b_size < size);
        
        // Stride-based element-wise addition with broadcasting support
        for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
            if (broadcast_b) {
                // Simple broadcasting: repeat b's elements cyclically
                // This works for cases like [batch, seq, features] + [batch, features]
                int b_idx = i % b_size;
                output.data[i] = a.data[i] + b.data[b_idx];
            } else {
                // Direct element-wise addition
                output.data[i] = a.data[i] + b.data[i];
            }
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
  
  // Note: We no longer check for exact workspace size since we're using dynamic allocation
  // The allocator will report if we run out of space
  
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
  float* intermediate_0_data = (float*)allocator.allocate(6291456);
  if (!intermediate_0_data) { fprintf(stderr, "Failed to allocate memory for intermediate_0\n"); return; }
  Tensor<float> intermediate_0_tensor = {intermediate_0_data, intermediate_0_shape, 3};
  float* intermediate_1_data = (float*)allocator.allocate(6291456);
  if (!intermediate_1_data) { fprintf(stderr, "Failed to allocate memory for intermediate_1\n"); return; }
  Tensor<float> intermediate_1_tensor = {intermediate_1_data, intermediate_1_shape, 3};
  float* intermediate_2_data = (float*)allocator.allocate(6291456);
  if (!intermediate_2_data) { fprintf(stderr, "Failed to allocate memory for intermediate_2\n"); return; }
  Tensor<float> intermediate_2_tensor = {intermediate_2_data, intermediate_2_shape, 3};
  float* intermediate_3_data = (float*)allocator.allocate(6291456);
  if (!intermediate_3_data) { fprintf(stderr, "Failed to allocate memory for intermediate_3\n"); return; }
  Tensor<float> intermediate_3_tensor = {intermediate_3_data, intermediate_3_shape, 3};
  float* intermediate_4_data = (float*)allocator.allocate(6291456);
  if (!intermediate_4_data) { fprintf(stderr, "Failed to allocate memory for intermediate_4\n"); return; }
  Tensor<float> intermediate_4_tensor = {intermediate_4_data, intermediate_4_shape, 3};
  float* intermediate_5_data = (float*)allocator.allocate(6291456);
  if (!intermediate_5_data) { fprintf(stderr, "Failed to allocate memory for intermediate_5\n"); return; }
  Tensor<float> intermediate_5_tensor = {intermediate_5_data, intermediate_5_shape, 4};
  float* intermediate_6_data = (float*)allocator.allocate(6291456);
  if (!intermediate_6_data) { fprintf(stderr, "Failed to allocate memory for intermediate_6\n"); return; }
  Tensor<float> intermediate_6_tensor = {intermediate_6_data, intermediate_6_shape, 4};
  float* intermediate_7_data = (float*)allocator.allocate(6291456);
  if (!intermediate_7_data) { fprintf(stderr, "Failed to allocate memory for intermediate_7\n"); return; }
  Tensor<float> intermediate_7_tensor = {intermediate_7_data, intermediate_7_shape, 4};
  float* intermediate_8_data = (float*)allocator.allocate(12582912);
  if (!intermediate_8_data) { fprintf(stderr, "Failed to allocate memory for intermediate_8\n"); return; }
  Tensor<float> intermediate_8_tensor = {intermediate_8_data, intermediate_8_shape, 4};
  float* intermediate_9_data = (float*)allocator.allocate(12582912);
  if (!intermediate_9_data) { fprintf(stderr, "Failed to allocate memory for intermediate_9\n"); return; }
  Tensor<float> intermediate_9_tensor = {intermediate_9_data, intermediate_9_shape, 4};
  float* intermediate_10_data = (float*)allocator.allocate(6291456);
  if (!intermediate_10_data) { fprintf(stderr, "Failed to allocate memory for intermediate_10\n"); return; }
  Tensor<float> intermediate_10_tensor = {intermediate_10_data, intermediate_10_shape, 4};
  float* intermediate_11_data = (float*)allocator.allocate(6291456);
  if (!intermediate_11_data) { fprintf(stderr, "Failed to allocate memory for intermediate_11\n"); return; }
  Tensor<float> intermediate_11_tensor = {intermediate_11_data, intermediate_11_shape, 3};
  float* intermediate_12_data = (float*)allocator.allocate(6291456);
  if (!intermediate_12_data) { fprintf(stderr, "Failed to allocate memory for intermediate_12\n"); return; }
  Tensor<float> intermediate_12_tensor = {intermediate_12_data, intermediate_12_shape, 3};
  float* intermediate_13_data = (float*)allocator.allocate(6291456);
  if (!intermediate_13_data) { fprintf(stderr, "Failed to allocate memory for intermediate_13\n"); return; }
  Tensor<float> intermediate_13_tensor = {intermediate_13_data, intermediate_13_shape, 3};
  float* intermediate_14_data = (float*)allocator.allocate(6291456);
  if (!intermediate_14_data) { fprintf(stderr, "Failed to allocate memory for intermediate_14\n"); return; }
  Tensor<float> intermediate_14_tensor = {intermediate_14_data, intermediate_14_shape, 3};
  float* intermediate_15_data = (float*)allocator.allocate(196608);
  if (!intermediate_15_data) { fprintf(stderr, "Failed to allocate memory for intermediate_15\n"); return; }
  Tensor<float> intermediate_15_tensor = {intermediate_15_data, intermediate_15_shape, 2};
  float* intermediate_16_data = (float*)allocator.allocate(196608);
  if (!intermediate_16_data) { fprintf(stderr, "Failed to allocate memory for intermediate_16\n"); return; }
  Tensor<float> intermediate_16_tensor = {intermediate_16_data, intermediate_16_shape, 2};
  float* intermediate_17_data = (float*)allocator.allocate(49152);
  if (!intermediate_17_data) { fprintf(stderr, "Failed to allocate memory for intermediate_17\n"); return; }
  Tensor<float> intermediate_17_tensor = {intermediate_17_data, intermediate_17_shape, 2};
  float* intermediate_18_data = (float*)allocator.allocate(6291456);
  if (!intermediate_18_data) { fprintf(stderr, "Failed to allocate memory for intermediate_18\n"); return; }
  Tensor<float> intermediate_18_tensor = {intermediate_18_data, intermediate_18_shape, 3};
  float* intermediate_19_data = (float*)allocator.allocate(6291456);
  if (!intermediate_19_data) { fprintf(stderr, "Failed to allocate memory for intermediate_19\n"); return; }
  Tensor<float> intermediate_19_tensor = {intermediate_19_data, intermediate_19_shape, 3};
  float* intermediate_20_data = (float*)allocator.allocate(49152);
  if (!intermediate_20_data) { fprintf(stderr, "Failed to allocate memory for intermediate_20\n"); return; }
  Tensor<float> intermediate_20_tensor = {intermediate_20_data, intermediate_20_shape, 2};
  float* intermediate_21_data = (float*)allocator.allocate(49152);
  if (!intermediate_21_data) { fprintf(stderr, "Failed to allocate memory for intermediate_21\n"); return; }
  Tensor<float> intermediate_21_tensor = {intermediate_21_data, intermediate_21_shape, 2};
  float* intermediate_22_data = (float*)allocator.allocate(49152);
  if (!intermediate_22_data) { fprintf(stderr, "Failed to allocate memory for intermediate_22\n"); return; }
  Tensor<float> intermediate_22_tensor = {intermediate_22_data, intermediate_22_shape, 2};
  float* intermediate_23_data = (float*)allocator.allocate(18874368);
  if (!intermediate_23_data) { fprintf(stderr, "Failed to allocate memory for intermediate_23\n"); return; }
  Tensor<float> intermediate_23_tensor = {intermediate_23_data, intermediate_23_shape, 4};
  float* intermediate_24_data = (float*)allocator.allocate(18874368);
  if (!intermediate_24_data) { fprintf(stderr, "Failed to allocate memory for intermediate_24\n"); return; }
  Tensor<float> intermediate_24_tensor = {intermediate_24_data, intermediate_24_shape, 4};
  float* intermediate_25_data = (float*)allocator.allocate(18874368);
  if (!intermediate_25_data) { fprintf(stderr, "Failed to allocate memory for intermediate_25\n"); return; }
  Tensor<float> intermediate_25_tensor = {intermediate_25_data, intermediate_25_shape, 4};
  float* intermediate_26_data = (float*)allocator.allocate(113246208);
  if (!intermediate_26_data) { fprintf(stderr, "Failed to allocate memory for intermediate_26\n"); return; }
  Tensor<float> intermediate_26_tensor = {intermediate_26_data, intermediate_26_shape, 4};
  float* intermediate_27_data = (float*)allocator.allocate(113246208);
  if (!intermediate_27_data) { fprintf(stderr, "Failed to allocate memory for intermediate_27\n"); return; }
  Tensor<float> intermediate_27_tensor = {intermediate_27_data, intermediate_27_shape, 4};
  float* intermediate_28_data = (float*)allocator.allocate(18874368);
  if (!intermediate_28_data) { fprintf(stderr, "Failed to allocate memory for intermediate_28\n"); return; }
  Tensor<float> intermediate_28_tensor = {intermediate_28_data, intermediate_28_shape, 4};
  float* intermediate_29_data = (float*)allocator.allocate(18874368);
  if (!intermediate_29_data) { fprintf(stderr, "Failed to allocate memory for intermediate_29\n"); return; }
  Tensor<float> intermediate_29_tensor = {intermediate_29_data, intermediate_29_shape, 3};
  float* intermediate_30_data = (float*)allocator.allocate(18874368);
  if (!intermediate_30_data) { fprintf(stderr, "Failed to allocate memory for intermediate_30\n"); return; }
  Tensor<float> intermediate_30_tensor = {intermediate_30_data, intermediate_30_shape, 3};
  float* intermediate_31_data = (float*)allocator.allocate(18874368);
  if (!intermediate_31_data) { fprintf(stderr, "Failed to allocate memory for intermediate_31\n"); return; }
  Tensor<float> intermediate_31_tensor = {intermediate_31_data, intermediate_31_shape, 3};
  float* intermediate_32_data = (float*)allocator.allocate(18874368);
  if (!intermediate_32_data) { fprintf(stderr, "Failed to allocate memory for intermediate_32\n"); return; }
  Tensor<float> intermediate_32_tensor = {intermediate_32_data, intermediate_32_shape, 3};
  float* intermediate_33_data = (float*)allocator.allocate(196608);
  if (!intermediate_33_data) { fprintf(stderr, "Failed to allocate memory for intermediate_33\n"); return; }
  Tensor<float> intermediate_33_tensor = {intermediate_33_data, intermediate_33_shape, 2};
  float* intermediate_34_data = (float*)allocator.allocate(196608);
  if (!intermediate_34_data) { fprintf(stderr, "Failed to allocate memory for intermediate_34\n"); return; }
  Tensor<float> intermediate_34_tensor = {intermediate_34_data, intermediate_34_shape, 2};
  float* intermediate_35_data = (float*)allocator.allocate(49152);
  if (!intermediate_35_data) { fprintf(stderr, "Failed to allocate memory for intermediate_35\n"); return; }
  Tensor<float> intermediate_35_tensor = {intermediate_35_data, intermediate_35_shape, 2};
  float* intermediate_36_data = (float*)allocator.allocate(18874368);
  if (!intermediate_36_data) { fprintf(stderr, "Failed to allocate memory for intermediate_36\n"); return; }
  Tensor<float> intermediate_36_tensor = {intermediate_36_data, intermediate_36_shape, 3};
  float* intermediate_37_data = (float*)allocator.allocate(18874368);
  if (!intermediate_37_data) { fprintf(stderr, "Failed to allocate memory for intermediate_37\n"); return; }
  Tensor<float> intermediate_37_tensor = {intermediate_37_data, intermediate_37_shape, 3};
  float* intermediate_38_data = (float*)allocator.allocate(49152);
  if (!intermediate_38_data) { fprintf(stderr, "Failed to allocate memory for intermediate_38\n"); return; }
  Tensor<float> intermediate_38_tensor = {intermediate_38_data, intermediate_38_shape, 2};
  float* intermediate_39_data = (float*)allocator.allocate(49152);
  if (!intermediate_39_data) { fprintf(stderr, "Failed to allocate memory for intermediate_39\n"); return; }
  Tensor<float> intermediate_39_tensor = {intermediate_39_data, intermediate_39_shape, 2};
  float* intermediate_40_data = (float*)allocator.allocate(49152);
  if (!intermediate_40_data) { fprintf(stderr, "Failed to allocate memory for intermediate_40\n"); return; }
  Tensor<float> intermediate_40_tensor = {intermediate_40_data, intermediate_40_shape, 2};
  float* intermediate_41_data = (float*)allocator.allocate(18874368);
  if (!intermediate_41_data) { fprintf(stderr, "Failed to allocate memory for intermediate_41\n"); return; }
  Tensor<float> intermediate_41_tensor = {intermediate_41_data, intermediate_41_shape, 4};
  float* intermediate_42_data = (float*)allocator.allocate(18874368);
  if (!intermediate_42_data) { fprintf(stderr, "Failed to allocate memory for intermediate_42\n"); return; }
  Tensor<float> intermediate_42_tensor = {intermediate_42_data, intermediate_42_shape, 4};
  float* intermediate_43_data = (float*)allocator.allocate(18874368);
  if (!intermediate_43_data) { fprintf(stderr, "Failed to allocate memory for intermediate_43\n"); return; }
  Tensor<float> intermediate_43_tensor = {intermediate_43_data, intermediate_43_shape, 4};
  float* intermediate_44_data = (float*)allocator.allocate(113246208);
  if (!intermediate_44_data) { fprintf(stderr, "Failed to allocate memory for intermediate_44\n"); return; }
  Tensor<float> intermediate_44_tensor = {intermediate_44_data, intermediate_44_shape, 4};
  float* intermediate_45_data = (float*)allocator.allocate(113246208);
  if (!intermediate_45_data) { fprintf(stderr, "Failed to allocate memory for intermediate_45\n"); return; }
  Tensor<float> intermediate_45_tensor = {intermediate_45_data, intermediate_45_shape, 4};
  float* intermediate_46_data = (float*)allocator.allocate(18874368);
  if (!intermediate_46_data) { fprintf(stderr, "Failed to allocate memory for intermediate_46\n"); return; }
  Tensor<float> intermediate_46_tensor = {intermediate_46_data, intermediate_46_shape, 4};
  float* intermediate_47_data = (float*)allocator.allocate(18874368);
  if (!intermediate_47_data) { fprintf(stderr, "Failed to allocate memory for intermediate_47\n"); return; }
  Tensor<float> intermediate_47_tensor = {intermediate_47_data, intermediate_47_shape, 3};
  float* intermediate_48_data = (float*)allocator.allocate(18874368);
  if (!intermediate_48_data) { fprintf(stderr, "Failed to allocate memory for intermediate_48\n"); return; }
  Tensor<float> intermediate_48_tensor = {intermediate_48_data, intermediate_48_shape, 3};
  float* intermediate_49_data = (float*)allocator.allocate(18874368);
  if (!intermediate_49_data) { fprintf(stderr, "Failed to allocate memory for intermediate_49\n"); return; }
  Tensor<float> intermediate_49_tensor = {intermediate_49_data, intermediate_49_shape, 3};
  float* intermediate_50_data = (float*)allocator.allocate(18874368);
  if (!intermediate_50_data) { fprintf(stderr, "Failed to allocate memory for intermediate_50\n"); return; }
  Tensor<float> intermediate_50_tensor = {intermediate_50_data, intermediate_50_shape, 3};
  float* intermediate_51_data = (float*)allocator.allocate(196608);
  if (!intermediate_51_data) { fprintf(stderr, "Failed to allocate memory for intermediate_51\n"); return; }
  Tensor<float> intermediate_51_tensor = {intermediate_51_data, intermediate_51_shape, 2};
  float* intermediate_52_data = (float*)allocator.allocate(196608);
  if (!intermediate_52_data) { fprintf(stderr, "Failed to allocate memory for intermediate_52\n"); return; }
  Tensor<float> intermediate_52_tensor = {intermediate_52_data, intermediate_52_shape, 2};
  float* intermediate_53_data = (float*)allocator.allocate(49152);
  if (!intermediate_53_data) { fprintf(stderr, "Failed to allocate memory for intermediate_53\n"); return; }
  Tensor<float> intermediate_53_tensor = {intermediate_53_data, intermediate_53_shape, 2};
  float* intermediate_54_data = (float*)allocator.allocate(18874368);
  if (!intermediate_54_data) { fprintf(stderr, "Failed to allocate memory for intermediate_54\n"); return; }
  Tensor<float> intermediate_54_tensor = {intermediate_54_data, intermediate_54_shape, 3};
  float* intermediate_55_data = (float*)allocator.allocate(18874368);
  if (!intermediate_55_data) { fprintf(stderr, "Failed to allocate memory for intermediate_55\n"); return; }
  Tensor<float> intermediate_55_tensor = {intermediate_55_data, intermediate_55_shape, 3};
  float* intermediate_56_data = (float*)allocator.allocate(49152);
  if (!intermediate_56_data) { fprintf(stderr, "Failed to allocate memory for intermediate_56\n"); return; }
  Tensor<float> intermediate_56_tensor = {intermediate_56_data, intermediate_56_shape, 2};
  float* intermediate_57_data = (float*)allocator.allocate(49152);
  if (!intermediate_57_data) { fprintf(stderr, "Failed to allocate memory for intermediate_57\n"); return; }
  Tensor<float> intermediate_57_tensor = {intermediate_57_data, intermediate_57_shape, 2};
  float* intermediate_58_data = (float*)allocator.allocate(49152);
  if (!intermediate_58_data) { fprintf(stderr, "Failed to allocate memory for intermediate_58\n"); return; }
  Tensor<float> intermediate_58_tensor = {intermediate_58_data, intermediate_58_shape, 2};
  float* intermediate_59_data = (float*)allocator.allocate(18874368);
  if (!intermediate_59_data) { fprintf(stderr, "Failed to allocate memory for intermediate_59\n"); return; }
  Tensor<float> intermediate_59_tensor = {intermediate_59_data, intermediate_59_shape, 4};
  float* intermediate_60_data = (float*)allocator.allocate(18874368);
  if (!intermediate_60_data) { fprintf(stderr, "Failed to allocate memory for intermediate_60\n"); return; }
  Tensor<float> intermediate_60_tensor = {intermediate_60_data, intermediate_60_shape, 4};
  float* intermediate_61_data = (float*)allocator.allocate(18874368);
  if (!intermediate_61_data) { fprintf(stderr, "Failed to allocate memory for intermediate_61\n"); return; }
  Tensor<float> intermediate_61_tensor = {intermediate_61_data, intermediate_61_shape, 4};
  float* intermediate_62_data = (float*)allocator.allocate(113246208);
  if (!intermediate_62_data) { fprintf(stderr, "Failed to allocate memory for intermediate_62\n"); return; }
  Tensor<float> intermediate_62_tensor = {intermediate_62_data, intermediate_62_shape, 4};
  float* intermediate_63_data = (float*)allocator.allocate(113246208);
  if (!intermediate_63_data) { fprintf(stderr, "Failed to allocate memory for intermediate_63\n"); return; }
  Tensor<float> intermediate_63_tensor = {intermediate_63_data, intermediate_63_shape, 4};
  float* intermediate_64_data = (float*)allocator.allocate(18874368);
  if (!intermediate_64_data) { fprintf(stderr, "Failed to allocate memory for intermediate_64\n"); return; }
  Tensor<float> intermediate_64_tensor = {intermediate_64_data, intermediate_64_shape, 4};
  float* intermediate_65_data = (float*)allocator.allocate(18874368);
  if (!intermediate_65_data) { fprintf(stderr, "Failed to allocate memory for intermediate_65\n"); return; }
  Tensor<float> intermediate_65_tensor = {intermediate_65_data, intermediate_65_shape, 3};
  float* intermediate_66_data = (float*)allocator.allocate(18874368);
  if (!intermediate_66_data) { fprintf(stderr, "Failed to allocate memory for intermediate_66\n"); return; }
  Tensor<float> intermediate_66_tensor = {intermediate_66_data, intermediate_66_shape, 3};
  float* intermediate_67_data = (float*)allocator.allocate(18874368);
  if (!intermediate_67_data) { fprintf(stderr, "Failed to allocate memory for intermediate_67\n"); return; }
  Tensor<float> intermediate_67_tensor = {intermediate_67_data, intermediate_67_shape, 3};
  float* intermediate_68_data = (float*)allocator.allocate(18874368);
  if (!intermediate_68_data) { fprintf(stderr, "Failed to allocate memory for intermediate_68\n"); return; }
  Tensor<float> intermediate_68_tensor = {intermediate_68_data, intermediate_68_shape, 3};
  float* intermediate_69_data = (float*)allocator.allocate(196608);
  if (!intermediate_69_data) { fprintf(stderr, "Failed to allocate memory for intermediate_69\n"); return; }
  Tensor<float> intermediate_69_tensor = {intermediate_69_data, intermediate_69_shape, 2};
  float* intermediate_70_data = (float*)allocator.allocate(196608);
  if (!intermediate_70_data) { fprintf(stderr, "Failed to allocate memory for intermediate_70\n"); return; }
  Tensor<float> intermediate_70_tensor = {intermediate_70_data, intermediate_70_shape, 2};
  float* intermediate_71_data = (float*)allocator.allocate(49152);
  if (!intermediate_71_data) { fprintf(stderr, "Failed to allocate memory for intermediate_71\n"); return; }
  Tensor<float> intermediate_71_tensor = {intermediate_71_data, intermediate_71_shape, 2};
  float* intermediate_72_data = (float*)allocator.allocate(18874368);
  if (!intermediate_72_data) { fprintf(stderr, "Failed to allocate memory for intermediate_72\n"); return; }
  Tensor<float> intermediate_72_tensor = {intermediate_72_data, intermediate_72_shape, 3};
  float* intermediate_73_data = (float*)allocator.allocate(18874368);
  if (!intermediate_73_data) { fprintf(stderr, "Failed to allocate memory for intermediate_73\n"); return; }
  Tensor<float> intermediate_73_tensor = {intermediate_73_data, intermediate_73_shape, 3};
  float* intermediate_74_data = (float*)allocator.allocate(49152);
  if (!intermediate_74_data) { fprintf(stderr, "Failed to allocate memory for intermediate_74\n"); return; }
  Tensor<float> intermediate_74_tensor = {intermediate_74_data, intermediate_74_shape, 2};
  float* intermediate_75_data = (float*)allocator.allocate(49152);
  if (!intermediate_75_data) { fprintf(stderr, "Failed to allocate memory for intermediate_75\n"); return; }
  Tensor<float> intermediate_75_tensor = {intermediate_75_data, intermediate_75_shape, 2};
  float* intermediate_76_data = (float*)allocator.allocate(49152);
  if (!intermediate_76_data) { fprintf(stderr, "Failed to allocate memory for intermediate_76\n"); return; }
  Tensor<float> intermediate_76_tensor = {intermediate_76_data, intermediate_76_shape, 2};
  float* intermediate_77_data = (float*)allocator.allocate(18874368);
  if (!intermediate_77_data) { fprintf(stderr, "Failed to allocate memory for intermediate_77\n"); return; }
  Tensor<float> intermediate_77_tensor = {intermediate_77_data, intermediate_77_shape, 4};
  float* intermediate_78_data = (float*)allocator.allocate(18874368);
  if (!intermediate_78_data) { fprintf(stderr, "Failed to allocate memory for intermediate_78\n"); return; }
  Tensor<float> intermediate_78_tensor = {intermediate_78_data, intermediate_78_shape, 4};
  float* intermediate_79_data = (float*)allocator.allocate(18874368);
  if (!intermediate_79_data) { fprintf(stderr, "Failed to allocate memory for intermediate_79\n"); return; }
  Tensor<float> intermediate_79_tensor = {intermediate_79_data, intermediate_79_shape, 4};
  float* intermediate_80_data = (float*)allocator.allocate(113246208);
  if (!intermediate_80_data) { fprintf(stderr, "Failed to allocate memory for intermediate_80\n"); return; }
  Tensor<float> intermediate_80_tensor = {intermediate_80_data, intermediate_80_shape, 4};
  float* intermediate_81_data = (float*)allocator.allocate(113246208);
  if (!intermediate_81_data) { fprintf(stderr, "Failed to allocate memory for intermediate_81\n"); return; }
  Tensor<float> intermediate_81_tensor = {intermediate_81_data, intermediate_81_shape, 4};
  float* intermediate_82_data = (float*)allocator.allocate(18874368);
  if (!intermediate_82_data) { fprintf(stderr, "Failed to allocate memory for intermediate_82\n"); return; }
  Tensor<float> intermediate_82_tensor = {intermediate_82_data, intermediate_82_shape, 4};
  float* intermediate_83_data = (float*)allocator.allocate(18874368);
  if (!intermediate_83_data) { fprintf(stderr, "Failed to allocate memory for intermediate_83\n"); return; }
  Tensor<float> intermediate_83_tensor = {intermediate_83_data, intermediate_83_shape, 3};
  float* intermediate_84_data = (float*)allocator.allocate(18874368);
  if (!intermediate_84_data) { fprintf(stderr, "Failed to allocate memory for intermediate_84\n"); return; }
  Tensor<float> intermediate_84_tensor = {intermediate_84_data, intermediate_84_shape, 3};
  float* intermediate_85_data = (float*)allocator.allocate(18874368);
  if (!intermediate_85_data) { fprintf(stderr, "Failed to allocate memory for intermediate_85\n"); return; }
  Tensor<float> intermediate_85_tensor = {intermediate_85_data, intermediate_85_shape, 3};
  float* intermediate_86_data = (float*)allocator.allocate(18874368);
  if (!intermediate_86_data) { fprintf(stderr, "Failed to allocate memory for intermediate_86\n"); return; }
  Tensor<float> intermediate_86_tensor = {intermediate_86_data, intermediate_86_shape, 3};
  float* intermediate_87_data = (float*)allocator.allocate(196608);
  if (!intermediate_87_data) { fprintf(stderr, "Failed to allocate memory for intermediate_87\n"); return; }
  Tensor<float> intermediate_87_tensor = {intermediate_87_data, intermediate_87_shape, 2};
  float* intermediate_88_data = (float*)allocator.allocate(196608);
  if (!intermediate_88_data) { fprintf(stderr, "Failed to allocate memory for intermediate_88\n"); return; }
  Tensor<float> intermediate_88_tensor = {intermediate_88_data, intermediate_88_shape, 2};
  float* intermediate_89_data = (float*)allocator.allocate(49152);
  if (!intermediate_89_data) { fprintf(stderr, "Failed to allocate memory for intermediate_89\n"); return; }
  Tensor<float> intermediate_89_tensor = {intermediate_89_data, intermediate_89_shape, 2};
  float* intermediate_90_data = (float*)allocator.allocate(18874368);
  if (!intermediate_90_data) { fprintf(stderr, "Failed to allocate memory for intermediate_90\n"); return; }
  Tensor<float> intermediate_90_tensor = {intermediate_90_data, intermediate_90_shape, 3};
  float* intermediate_91_data = (float*)allocator.allocate(18874368);
  if (!intermediate_91_data) { fprintf(stderr, "Failed to allocate memory for intermediate_91\n"); return; }
  Tensor<float> intermediate_91_tensor = {intermediate_91_data, intermediate_91_shape, 3};
  float* intermediate_92_data = (float*)allocator.allocate(49152);
  if (!intermediate_92_data) { fprintf(stderr, "Failed to allocate memory for intermediate_92\n"); return; }
  Tensor<float> intermediate_92_tensor = {intermediate_92_data, intermediate_92_shape, 2};
  float* intermediate_93_data = (float*)allocator.allocate(49152);
  if (!intermediate_93_data) { fprintf(stderr, "Failed to allocate memory for intermediate_93\n"); return; }
  Tensor<float> intermediate_93_tensor = {intermediate_93_data, intermediate_93_shape, 2};
  float* intermediate_94_data = (float*)allocator.allocate(49152);
  if (!intermediate_94_data) { fprintf(stderr, "Failed to allocate memory for intermediate_94\n"); return; }
  Tensor<float> intermediate_94_tensor = {intermediate_94_data, intermediate_94_shape, 2};
  float* intermediate_95_data = (float*)allocator.allocate(18874368);
  if (!intermediate_95_data) { fprintf(stderr, "Failed to allocate memory for intermediate_95\n"); return; }
  Tensor<float> intermediate_95_tensor = {intermediate_95_data, intermediate_95_shape, 4};
  float* intermediate_96_data = (float*)allocator.allocate(18874368);
  if (!intermediate_96_data) { fprintf(stderr, "Failed to allocate memory for intermediate_96\n"); return; }
  Tensor<float> intermediate_96_tensor = {intermediate_96_data, intermediate_96_shape, 4};
  float* intermediate_97_data = (float*)allocator.allocate(18874368);
  if (!intermediate_97_data) { fprintf(stderr, "Failed to allocate memory for intermediate_97\n"); return; }
  Tensor<float> intermediate_97_tensor = {intermediate_97_data, intermediate_97_shape, 4};
  float* intermediate_98_data = (float*)allocator.allocate(113246208);
  if (!intermediate_98_data) { fprintf(stderr, "Failed to allocate memory for intermediate_98\n"); return; }
  Tensor<float> intermediate_98_tensor = {intermediate_98_data, intermediate_98_shape, 4};
  float* intermediate_99_data = (float*)allocator.allocate(113246208);
  if (!intermediate_99_data) { fprintf(stderr, "Failed to allocate memory for intermediate_99\n"); return; }
  Tensor<float> intermediate_99_tensor = {intermediate_99_data, intermediate_99_shape, 4};
  float* intermediate_100_data = (float*)allocator.allocate(18874368);
  if (!intermediate_100_data) { fprintf(stderr, "Failed to allocate memory for intermediate_100\n"); return; }
  Tensor<float> intermediate_100_tensor = {intermediate_100_data, intermediate_100_shape, 4};
  float* intermediate_101_data = (float*)allocator.allocate(18874368);
  if (!intermediate_101_data) { fprintf(stderr, "Failed to allocate memory for intermediate_101\n"); return; }
  Tensor<float> intermediate_101_tensor = {intermediate_101_data, intermediate_101_shape, 3};
  float* intermediate_102_data = (float*)allocator.allocate(18874368);
  if (!intermediate_102_data) { fprintf(stderr, "Failed to allocate memory for intermediate_102\n"); return; }
  Tensor<float> intermediate_102_tensor = {intermediate_102_data, intermediate_102_shape, 3};
  float* intermediate_103_data = (float*)allocator.allocate(18874368);
  if (!intermediate_103_data) { fprintf(stderr, "Failed to allocate memory for intermediate_103\n"); return; }
  Tensor<float> intermediate_103_tensor = {intermediate_103_data, intermediate_103_shape, 3};
  float* intermediate_104_data = (float*)allocator.allocate(18874368);
  if (!intermediate_104_data) { fprintf(stderr, "Failed to allocate memory for intermediate_104\n"); return; }
  Tensor<float> intermediate_104_tensor = {intermediate_104_data, intermediate_104_shape, 3};
  float* intermediate_105_data = (float*)allocator.allocate(196608);
  if (!intermediate_105_data) { fprintf(stderr, "Failed to allocate memory for intermediate_105\n"); return; }
  Tensor<float> intermediate_105_tensor = {intermediate_105_data, intermediate_105_shape, 2};
  float* intermediate_106_data = (float*)allocator.allocate(196608);
  if (!intermediate_106_data) { fprintf(stderr, "Failed to allocate memory for intermediate_106\n"); return; }
  Tensor<float> intermediate_106_tensor = {intermediate_106_data, intermediate_106_shape, 2};
  float* intermediate_107_data = (float*)allocator.allocate(49152);
  if (!intermediate_107_data) { fprintf(stderr, "Failed to allocate memory for intermediate_107\n"); return; }
  Tensor<float> intermediate_107_tensor = {intermediate_107_data, intermediate_107_shape, 2};
  float* intermediate_108_data = (float*)allocator.allocate(18874368);
  if (!intermediate_108_data) { fprintf(stderr, "Failed to allocate memory for intermediate_108\n"); return; }
  Tensor<float> intermediate_108_tensor = {intermediate_108_data, intermediate_108_shape, 3};
  float* intermediate_109_data = (float*)allocator.allocate(18874368);
  if (!intermediate_109_data) { fprintf(stderr, "Failed to allocate memory for intermediate_109\n"); return; }
  Tensor<float> intermediate_109_tensor = {intermediate_109_data, intermediate_109_shape, 3};
  float* intermediate_110_data = (float*)allocator.allocate(8320);
  if (!intermediate_110_data) { fprintf(stderr, "Failed to allocate memory for intermediate_110\n"); return; }
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
  batched_matmul_transpose_b_tiled<<<dim3(1, 32), dim3(128, 1, 1), 0>>>(intermediate_8_tensor, intermediate_5_tensor, intermediate_6_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(1, 32), dim3(128, 1, 1), 0>>>(intermediate_9_tensor, intermediate_8_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(1, 32), dim3(64, 1, 1), 0>>>(intermediate_10_tensor, intermediate_9_tensor, intermediate_7_tensor);
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
  batched_matmul_transpose_b_tiled<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_26_tensor, intermediate_23_tensor, intermediate_24_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_27_tensor, intermediate_26_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(1, 32), dim3(64, 1, 1), 0>>>(intermediate_28_tensor, intermediate_27_tensor, intermediate_25_tensor);
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
  batched_matmul_transpose_b_tiled<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_44_tensor, intermediate_41_tensor, intermediate_42_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_45_tensor, intermediate_44_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(1, 32), dim3(64, 1, 1), 0>>>(intermediate_46_tensor, intermediate_45_tensor, intermediate_43_tensor);
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
  batched_matmul_transpose_b_tiled<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_62_tensor, intermediate_59_tensor, intermediate_60_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_63_tensor, intermediate_62_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(1, 32), dim3(64, 1, 1), 0>>>(intermediate_64_tensor, intermediate_63_tensor, intermediate_61_tensor);
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
  batched_matmul_transpose_b_tiled<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_80_tensor, intermediate_77_tensor, intermediate_78_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_81_tensor, intermediate_80_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(1, 32), dim3(64, 1, 1), 0>>>(intermediate_82_tensor, intermediate_81_tensor, intermediate_79_tensor);
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
  batched_matmul_transpose_b_tiled<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_98_tensor, intermediate_95_tensor, intermediate_96_tensor);
  CUDA_CHECK(cudaGetLastError());
  fused_scale_softmax_forward<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_99_tensor, intermediate_98_tensor);
  CUDA_CHECK(cudaGetLastError());
  batched_matmul_tiled<<<dim3(1, 32), dim3(64, 1, 1), 0>>>(intermediate_100_tensor, intermediate_99_tensor, intermediate_97_tensor);
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
    