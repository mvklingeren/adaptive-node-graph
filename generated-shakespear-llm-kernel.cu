
#include <cuda_runtime.h>
#include <cfloat>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


// Always enable for debugging purposes
#define TENSOR_BOUNDS_CHECK 1
#define TENSOR_BOUNDS_CHECK_VERBOSE 1

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
      asm("trap;");
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
  
  __device__ inline int total_elements() const {
    int total = 1;
    for (int i = 0; i < dims; i++) {
      total *= shape[i];
    }
    return total;
  }
};


class WorkspaceAllocator {
private:
    char* base_ptr;
    size_t current_offset;
    size_t total_size;
    
    inline size_t align_to_boundary(size_t size) const {
        const size_t alignment = 256;
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
public:
    WorkspaceAllocator(char* workspace, size_t workspace_size) 
        : base_ptr(workspace), current_offset(0), total_size(workspace_size) {}
    
    // Sequential allocation (for fallback cases)
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
    
    // Direct offset allocation (for memory reuse)
    void* allocate_at_offset(size_t offset, size_t size) {
        size_t aligned_size = align_to_boundary(size);
        
        if (offset + aligned_size > total_size) {
            fprintf(stderr, "ERROR: Offset allocation out of bounds. Offset: %zu, Size: %zu (aligned: %zu), Total: %zu\n", 
                    offset, size, aligned_size, total_size);
            return nullptr;
        }
        
        return base_ptr + offset;
    }
    
    size_t get_used_size() const {
        return current_offset;
    }
    
    void reset() {
        current_offset = 0;
    }
};


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

        if (batch_idx < input.shape[0] && output_feature_idx < weights.shape[1]) {
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
    

extern "C" void executeGraph(
    void* input_data,
    const int* input_shape,
    int input_dims,
    const char* input_dtype,
    
    void* output_data,
    const int* output_shape,
    int output_dims,
    const char* output_dtype,
    
    void** param_data,
    const int** param_shapes,
    const int* param_dims,
    const char** param_dtypes,
    int param_count,
    
    char* workspace,
    size_t workspace_size
) {
    if (!workspace) {
        fprintf(stderr, "Error: Null workspace pointer passed to executeGraph\n");
        return;
    }
    
    if (workspace_size < 320864256) {
        fprintf(stderr, "Error: Insufficient workspace size. Required: 320864256 bytes, Provided: %zu bytes\n", workspace_size);
        return;
    }
    
    if (param_count != 100) {
        fprintf(stderr, "Error: Expected 100 parameters, got %d\n", param_count);
        return;
    }
    
    WorkspaceAllocator allocator(workspace, workspace_size);

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

    Tensor<float> param_tensors[100];
    for (int i = 0; i < 100; i++) {
        param_tensors[i] = {
            (float*)param_data[i],
            param_shapes[i],
            param_dims[i]
        };
    }

  // Pool: default, Offset: 0, Size: 6291456 bytes
  float* intermediate_0_data = (float*)(allocator.allocate_at_offset(0, 6291456));
  Tensor<float> intermediate_0_tensor = {intermediate_0_data, intermediate_0_shape, 3};
  // Pool: default, Offset: 6291456, Size: 6291456 bytes
  float* intermediate_1_data = (float*)(allocator.allocate_at_offset(6291456, 6291456));
  Tensor<float> intermediate_1_tensor = {intermediate_1_data, intermediate_1_shape, 3};
  // Pool: default, Offset: 0, Size: 6291456 bytes
  float* intermediate_2_data = (float*)(allocator.allocate_at_offset(0, 6291456));
  Tensor<float> intermediate_2_tensor = {intermediate_2_data, intermediate_2_shape, 3};
  // Pool: default, Offset: 12582912, Size: 6291456 bytes
  float* intermediate_3_data = (float*)(allocator.allocate_at_offset(12582912, 6291456));
  Tensor<float> intermediate_3_tensor = {intermediate_3_data, intermediate_3_shape, 3};
  // Pool: default, Offset: 18874368, Size: 6291456 bytes
  float* intermediate_4_data = (float*)(allocator.allocate_at_offset(18874368, 6291456));
  Tensor<float> intermediate_4_tensor = {intermediate_4_data, intermediate_4_shape, 3};
  // Pool: default, Offset: 25165824, Size: 6291456 bytes
  float* intermediate_5_data = (float*)(allocator.allocate_at_offset(25165824, 6291456));
  Tensor<float> intermediate_5_tensor = {intermediate_5_data, intermediate_5_shape, 4};
  // Pool: default, Offset: 0, Size: 6291456 bytes
  float* intermediate_6_data = (float*)(allocator.allocate_at_offset(0, 6291456));
  Tensor<float> intermediate_6_tensor = {intermediate_6_data, intermediate_6_shape, 4};
  // Pool: default, Offset: 12582912, Size: 6291456 bytes
  float* intermediate_7_data = (float*)(allocator.allocate_at_offset(12582912, 6291456));
  Tensor<float> intermediate_7_tensor = {intermediate_7_data, intermediate_7_shape, 4};
  // Pool: default, Offset: 31457280, Size: 12582912 bytes
  float* intermediate_8_data = (float*)(allocator.allocate_at_offset(31457280, 12582912));
  Tensor<float> intermediate_8_tensor = {intermediate_8_data, intermediate_8_shape, 4};
  // Pool: default, Offset: 18874368, Size: 12582912 bytes
  float* intermediate_9_data = (float*)(allocator.allocate_at_offset(18874368, 12582912));
  Tensor<float> intermediate_9_tensor = {intermediate_9_data, intermediate_9_shape, 4};
  // Pool: default, Offset: 0, Size: 6291456 bytes
  float* intermediate_10_data = (float*)(allocator.allocate_at_offset(0, 6291456));
  Tensor<float> intermediate_10_tensor = {intermediate_10_data, intermediate_10_shape, 4};
  // Pool: default, Offset: 12582912, Size: 6291456 bytes
  float* intermediate_11_data = (float*)(allocator.allocate_at_offset(12582912, 6291456));
  Tensor<float> intermediate_11_tensor = {intermediate_11_data, intermediate_11_shape, 3};
  // Pool: default, Offset: 0, Size: 6291456 bytes
  float* intermediate_12_data = (float*)(allocator.allocate_at_offset(0, 6291456));
  Tensor<float> intermediate_12_tensor = {intermediate_12_data, intermediate_12_shape, 3};
  // Pool: default, Offset: 12582912, Size: 6291456 bytes
  float* intermediate_13_data = (float*)(allocator.allocate_at_offset(12582912, 6291456));
  Tensor<float> intermediate_13_tensor = {intermediate_13_data, intermediate_13_shape, 3};
  // Pool: default, Offset: 0, Size: 6291456 bytes
  float* intermediate_14_data = (float*)(allocator.allocate_at_offset(0, 6291456));
  Tensor<float> intermediate_14_tensor = {intermediate_14_data, intermediate_14_shape, 3};
  // Pool: default, Offset: 12582912, Size: 196608 bytes
  float* intermediate_15_data = (float*)(allocator.allocate_at_offset(12582912, 196608));
  Tensor<float> intermediate_15_tensor = {intermediate_15_data, intermediate_15_shape, 2};
  // Pool: default, Offset: 12779520, Size: 196608 bytes
  float* intermediate_16_data = (float*)(allocator.allocate_at_offset(12779520, 196608));
  Tensor<float> intermediate_16_tensor = {intermediate_16_data, intermediate_16_shape, 2};
  // Pool: default, Offset: 12582912, Size: 49152 bytes
  float* intermediate_17_data = (float*)(allocator.allocate_at_offset(12582912, 49152));
  Tensor<float> intermediate_17_tensor = {intermediate_17_data, intermediate_17_shape, 2};
  // Pool: default, Offset: 12632064, Size: 6291456 bytes
  float* intermediate_18_data = (float*)(allocator.allocate_at_offset(12632064, 6291456));
  Tensor<float> intermediate_18_tensor = {intermediate_18_data, intermediate_18_shape, 3};
  // Pool: default, Offset: 0, Size: 6291456 bytes
  float* intermediate_19_data = (float*)(allocator.allocate_at_offset(0, 6291456));
  Tensor<float> intermediate_19_tensor = {intermediate_19_data, intermediate_19_shape, 3};
  // Pool: default, Offset: 6291456, Size: 49152 bytes
  float* intermediate_20_data = (float*)(allocator.allocate_at_offset(6291456, 49152));
  Tensor<float> intermediate_20_tensor = {intermediate_20_data, intermediate_20_shape, 2};
  // Pool: default, Offset: 6340608, Size: 49152 bytes
  float* intermediate_21_data = (float*)(allocator.allocate_at_offset(6340608, 49152));
  Tensor<float> intermediate_21_tensor = {intermediate_21_data, intermediate_21_shape, 2};
  // Pool: default, Offset: 6389760, Size: 49152 bytes
  float* intermediate_22_data = (float*)(allocator.allocate_at_offset(6389760, 49152));
  Tensor<float> intermediate_22_tensor = {intermediate_22_data, intermediate_22_shape, 2};
  // Pool: default, Offset: 6438912, Size: 18874368 bytes
  float* intermediate_23_data = (float*)(allocator.allocate_at_offset(6438912, 18874368));
  Tensor<float> intermediate_23_tensor = {intermediate_23_data, intermediate_23_shape, 4};
  // Pool: default, Offset: 25313280, Size: 18874368 bytes
  float* intermediate_24_data = (float*)(allocator.allocate_at_offset(25313280, 18874368));
  Tensor<float> intermediate_24_tensor = {intermediate_24_data, intermediate_24_shape, 4};
  // Pool: default, Offset: 44187648, Size: 18874368 bytes
  float* intermediate_25_data = (float*)(allocator.allocate_at_offset(44187648, 18874368));
  Tensor<float> intermediate_25_tensor = {intermediate_25_data, intermediate_25_shape, 4};
  // Pool: default, Offset: 63062016, Size: 113246208 bytes
  float* intermediate_26_data = (float*)(allocator.allocate_at_offset(63062016, 113246208));
  Tensor<float> intermediate_26_tensor = {intermediate_26_data, intermediate_26_shape, 4};
  // Pool: default, Offset: 176308224, Size: 113246208 bytes
  float* intermediate_27_data = (float*)(allocator.allocate_at_offset(176308224, 113246208));
  Tensor<float> intermediate_27_tensor = {intermediate_27_data, intermediate_27_shape, 4};
  // Pool: default, Offset: 6291456, Size: 18874368 bytes
  float* intermediate_28_data = (float*)(allocator.allocate_at_offset(6291456, 18874368));
  Tensor<float> intermediate_28_tensor = {intermediate_28_data, intermediate_28_shape, 4};
  // Pool: default, Offset: 25165824, Size: 18874368 bytes
  float* intermediate_29_data = (float*)(allocator.allocate_at_offset(25165824, 18874368));
  Tensor<float> intermediate_29_tensor = {intermediate_29_data, intermediate_29_shape, 3};
  // Pool: default, Offset: 6291456, Size: 18874368 bytes
  float* intermediate_30_data = (float*)(allocator.allocate_at_offset(6291456, 18874368));
  Tensor<float> intermediate_30_tensor = {intermediate_30_data, intermediate_30_shape, 3};
  // Pool: default, Offset: 25165824, Size: 18874368 bytes
  float* intermediate_31_data = (float*)(allocator.allocate_at_offset(25165824, 18874368));
  Tensor<float> intermediate_31_tensor = {intermediate_31_data, intermediate_31_shape, 3};
  // Pool: default, Offset: 6291456, Size: 18874368 bytes
  float* intermediate_32_data = (float*)(allocator.allocate_at_offset(6291456, 18874368));
  Tensor<float> intermediate_32_tensor = {intermediate_32_data, intermediate_32_shape, 3};
  // Pool: default, Offset: 25165824, Size: 196608 bytes
  float* intermediate_33_data = (float*)(allocator.allocate_at_offset(25165824, 196608));
  Tensor<float> intermediate_33_tensor = {intermediate_33_data, intermediate_33_shape, 2};
  // Pool: default, Offset: 25362432, Size: 196608 bytes
  float* intermediate_34_data = (float*)(allocator.allocate_at_offset(25362432, 196608));
  Tensor<float> intermediate_34_tensor = {intermediate_34_data, intermediate_34_shape, 2};
  // Pool: default, Offset: 25165824, Size: 49152 bytes
  float* intermediate_35_data = (float*)(allocator.allocate_at_offset(25165824, 49152));
  Tensor<float> intermediate_35_tensor = {intermediate_35_data, intermediate_35_shape, 2};
  // Pool: default, Offset: 25214976, Size: 18874368 bytes
  float* intermediate_36_data = (float*)(allocator.allocate_at_offset(25214976, 18874368));
  Tensor<float> intermediate_36_tensor = {intermediate_36_data, intermediate_36_shape, 3};
  // Pool: default, Offset: 6291456, Size: 18874368 bytes
  float* intermediate_37_data = (float*)(allocator.allocate_at_offset(6291456, 18874368));
  Tensor<float> intermediate_37_tensor = {intermediate_37_data, intermediate_37_shape, 3};
  // Pool: default, Offset: 0, Size: 49152 bytes
  float* intermediate_38_data = (float*)(allocator.allocate_at_offset(0, 49152));
  Tensor<float> intermediate_38_tensor = {intermediate_38_data, intermediate_38_shape, 2};
  // Pool: default, Offset: 49152, Size: 49152 bytes
  float* intermediate_39_data = (float*)(allocator.allocate_at_offset(49152, 49152));
  Tensor<float> intermediate_39_tensor = {intermediate_39_data, intermediate_39_shape, 2};
  // Pool: default, Offset: 98304, Size: 49152 bytes
  float* intermediate_40_data = (float*)(allocator.allocate_at_offset(98304, 49152));
  Tensor<float> intermediate_40_tensor = {intermediate_40_data, intermediate_40_shape, 2};
  // Pool: default, Offset: 25165824, Size: 18874368 bytes
  float* intermediate_41_data = (float*)(allocator.allocate_at_offset(25165824, 18874368));
  Tensor<float> intermediate_41_tensor = {intermediate_41_data, intermediate_41_shape, 4};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_42_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_42_tensor = {intermediate_42_data, intermediate_42_shape, 4};
  // Pool: default, Offset: 62914560, Size: 18874368 bytes
  float* intermediate_43_data = (float*)(allocator.allocate_at_offset(62914560, 18874368));
  Tensor<float> intermediate_43_tensor = {intermediate_43_data, intermediate_43_shape, 4};
  // Pool: default, Offset: 81788928, Size: 113246208 bytes
  float* intermediate_44_data = (float*)(allocator.allocate_at_offset(81788928, 113246208));
  Tensor<float> intermediate_44_tensor = {intermediate_44_data, intermediate_44_shape, 4};
  // Pool: default, Offset: 195035136, Size: 113246208 bytes
  float* intermediate_45_data = (float*)(allocator.allocate_at_offset(195035136, 113246208));
  Tensor<float> intermediate_45_tensor = {intermediate_45_data, intermediate_45_shape, 4};
  // Pool: default, Offset: 25165824, Size: 18874368 bytes
  float* intermediate_46_data = (float*)(allocator.allocate_at_offset(25165824, 18874368));
  Tensor<float> intermediate_46_tensor = {intermediate_46_data, intermediate_46_shape, 4};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_47_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_47_tensor = {intermediate_47_data, intermediate_47_shape, 3};
  // Pool: default, Offset: 25165824, Size: 18874368 bytes
  float* intermediate_48_data = (float*)(allocator.allocate_at_offset(25165824, 18874368));
  Tensor<float> intermediate_48_tensor = {intermediate_48_data, intermediate_48_shape, 3};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_49_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_49_tensor = {intermediate_49_data, intermediate_49_shape, 3};
  // Pool: default, Offset: 25165824, Size: 18874368 bytes
  float* intermediate_50_data = (float*)(allocator.allocate_at_offset(25165824, 18874368));
  Tensor<float> intermediate_50_tensor = {intermediate_50_data, intermediate_50_shape, 3};
  // Pool: default, Offset: 0, Size: 196608 bytes
  float* intermediate_51_data = (float*)(allocator.allocate_at_offset(0, 196608));
  Tensor<float> intermediate_51_tensor = {intermediate_51_data, intermediate_51_shape, 2};
  // Pool: default, Offset: 196608, Size: 196608 bytes
  float* intermediate_52_data = (float*)(allocator.allocate_at_offset(196608, 196608));
  Tensor<float> intermediate_52_tensor = {intermediate_52_data, intermediate_52_shape, 2};
  // Pool: default, Offset: 0, Size: 49152 bytes
  float* intermediate_53_data = (float*)(allocator.allocate_at_offset(0, 49152));
  Tensor<float> intermediate_53_tensor = {intermediate_53_data, intermediate_53_shape, 2};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_54_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_54_tensor = {intermediate_54_data, intermediate_54_shape, 3};
  // Pool: default, Offset: 25165824, Size: 18874368 bytes
  float* intermediate_55_data = (float*)(allocator.allocate_at_offset(25165824, 18874368));
  Tensor<float> intermediate_55_tensor = {intermediate_55_data, intermediate_55_shape, 3};
  // Pool: default, Offset: 0, Size: 49152 bytes
  float* intermediate_56_data = (float*)(allocator.allocate_at_offset(0, 49152));
  Tensor<float> intermediate_56_tensor = {intermediate_56_data, intermediate_56_shape, 2};
  // Pool: default, Offset: 49152, Size: 49152 bytes
  float* intermediate_57_data = (float*)(allocator.allocate_at_offset(49152, 49152));
  Tensor<float> intermediate_57_tensor = {intermediate_57_data, intermediate_57_shape, 2};
  // Pool: default, Offset: 98304, Size: 49152 bytes
  float* intermediate_58_data = (float*)(allocator.allocate_at_offset(98304, 49152));
  Tensor<float> intermediate_58_tensor = {intermediate_58_data, intermediate_58_shape, 2};
  // Pool: default, Offset: 147456, Size: 18874368 bytes
  float* intermediate_59_data = (float*)(allocator.allocate_at_offset(147456, 18874368));
  Tensor<float> intermediate_59_tensor = {intermediate_59_data, intermediate_59_shape, 4};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_60_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_60_tensor = {intermediate_60_data, intermediate_60_shape, 4};
  // Pool: default, Offset: 62914560, Size: 18874368 bytes
  float* intermediate_61_data = (float*)(allocator.allocate_at_offset(62914560, 18874368));
  Tensor<float> intermediate_61_tensor = {intermediate_61_data, intermediate_61_shape, 4};
  // Pool: default, Offset: 81788928, Size: 113246208 bytes
  float* intermediate_62_data = (float*)(allocator.allocate_at_offset(81788928, 113246208));
  Tensor<float> intermediate_62_tensor = {intermediate_62_data, intermediate_62_shape, 4};
  // Pool: default, Offset: 195035136, Size: 113246208 bytes
  float* intermediate_63_data = (float*)(allocator.allocate_at_offset(195035136, 113246208));
  Tensor<float> intermediate_63_tensor = {intermediate_63_data, intermediate_63_shape, 4};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_64_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_64_tensor = {intermediate_64_data, intermediate_64_shape, 4};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_65_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_65_tensor = {intermediate_65_data, intermediate_65_shape, 3};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_66_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_66_tensor = {intermediate_66_data, intermediate_66_shape, 3};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_67_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_67_tensor = {intermediate_67_data, intermediate_67_shape, 3};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_68_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_68_tensor = {intermediate_68_data, intermediate_68_shape, 3};
  // Pool: default, Offset: 18874368, Size: 196608 bytes
  float* intermediate_69_data = (float*)(allocator.allocate_at_offset(18874368, 196608));
  Tensor<float> intermediate_69_tensor = {intermediate_69_data, intermediate_69_shape, 2};
  // Pool: default, Offset: 19070976, Size: 196608 bytes
  float* intermediate_70_data = (float*)(allocator.allocate_at_offset(19070976, 196608));
  Tensor<float> intermediate_70_tensor = {intermediate_70_data, intermediate_70_shape, 2};
  // Pool: default, Offset: 18874368, Size: 49152 bytes
  float* intermediate_71_data = (float*)(allocator.allocate_at_offset(18874368, 49152));
  Tensor<float> intermediate_71_tensor = {intermediate_71_data, intermediate_71_shape, 2};
  // Pool: default, Offset: 44040192, Size: 18874368 bytes
  float* intermediate_72_data = (float*)(allocator.allocate_at_offset(44040192, 18874368));
  Tensor<float> intermediate_72_tensor = {intermediate_72_data, intermediate_72_shape, 3};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_73_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_73_tensor = {intermediate_73_data, intermediate_73_shape, 3};
  // Pool: default, Offset: 18874368, Size: 49152 bytes
  float* intermediate_74_data = (float*)(allocator.allocate_at_offset(18874368, 49152));
  Tensor<float> intermediate_74_tensor = {intermediate_74_data, intermediate_74_shape, 2};
  // Pool: default, Offset: 18923520, Size: 49152 bytes
  float* intermediate_75_data = (float*)(allocator.allocate_at_offset(18923520, 49152));
  Tensor<float> intermediate_75_tensor = {intermediate_75_data, intermediate_75_shape, 2};
  // Pool: default, Offset: 18972672, Size: 49152 bytes
  float* intermediate_76_data = (float*)(allocator.allocate_at_offset(18972672, 49152));
  Tensor<float> intermediate_76_tensor = {intermediate_76_data, intermediate_76_shape, 2};
  // Pool: default, Offset: 19021824, Size: 18874368 bytes
  float* intermediate_77_data = (float*)(allocator.allocate_at_offset(19021824, 18874368));
  Tensor<float> intermediate_77_tensor = {intermediate_77_data, intermediate_77_shape, 4};
  // Pool: default, Offset: 37896192, Size: 18874368 bytes
  float* intermediate_78_data = (float*)(allocator.allocate_at_offset(37896192, 18874368));
  Tensor<float> intermediate_78_tensor = {intermediate_78_data, intermediate_78_shape, 4};
  // Pool: default, Offset: 56770560, Size: 18874368 bytes
  float* intermediate_79_data = (float*)(allocator.allocate_at_offset(56770560, 18874368));
  Tensor<float> intermediate_79_tensor = {intermediate_79_data, intermediate_79_shape, 4};
  // Pool: default, Offset: 75644928, Size: 113246208 bytes
  float* intermediate_80_data = (float*)(allocator.allocate_at_offset(75644928, 113246208));
  Tensor<float> intermediate_80_tensor = {intermediate_80_data, intermediate_80_shape, 4};
  // Pool: default, Offset: 188891136, Size: 113246208 bytes
  float* intermediate_81_data = (float*)(allocator.allocate_at_offset(188891136, 113246208));
  Tensor<float> intermediate_81_tensor = {intermediate_81_data, intermediate_81_shape, 4};
  // Pool: default, Offset: 18874368, Size: 18874368 bytes
  float* intermediate_82_data = (float*)(allocator.allocate_at_offset(18874368, 18874368));
  Tensor<float> intermediate_82_tensor = {intermediate_82_data, intermediate_82_shape, 4};
  // Pool: default, Offset: 37748736, Size: 18874368 bytes
  float* intermediate_83_data = (float*)(allocator.allocate_at_offset(37748736, 18874368));
  Tensor<float> intermediate_83_tensor = {intermediate_83_data, intermediate_83_shape, 3};
  // Pool: default, Offset: 18874368, Size: 18874368 bytes
  float* intermediate_84_data = (float*)(allocator.allocate_at_offset(18874368, 18874368));
  Tensor<float> intermediate_84_tensor = {intermediate_84_data, intermediate_84_shape, 3};
  // Pool: default, Offset: 37748736, Size: 18874368 bytes
  float* intermediate_85_data = (float*)(allocator.allocate_at_offset(37748736, 18874368));
  Tensor<float> intermediate_85_tensor = {intermediate_85_data, intermediate_85_shape, 3};
  // Pool: default, Offset: 18874368, Size: 18874368 bytes
  float* intermediate_86_data = (float*)(allocator.allocate_at_offset(18874368, 18874368));
  Tensor<float> intermediate_86_tensor = {intermediate_86_data, intermediate_86_shape, 3};
  // Pool: default, Offset: 37748736, Size: 196608 bytes
  float* intermediate_87_data = (float*)(allocator.allocate_at_offset(37748736, 196608));
  Tensor<float> intermediate_87_tensor = {intermediate_87_data, intermediate_87_shape, 2};
  // Pool: default, Offset: 37945344, Size: 196608 bytes
  float* intermediate_88_data = (float*)(allocator.allocate_at_offset(37945344, 196608));
  Tensor<float> intermediate_88_tensor = {intermediate_88_data, intermediate_88_shape, 2};
  // Pool: default, Offset: 37748736, Size: 49152 bytes
  float* intermediate_89_data = (float*)(allocator.allocate_at_offset(37748736, 49152));
  Tensor<float> intermediate_89_tensor = {intermediate_89_data, intermediate_89_shape, 2};
  // Pool: default, Offset: 37797888, Size: 18874368 bytes
  float* intermediate_90_data = (float*)(allocator.allocate_at_offset(37797888, 18874368));
  Tensor<float> intermediate_90_tensor = {intermediate_90_data, intermediate_90_shape, 3};
  // Pool: default, Offset: 18874368, Size: 18874368 bytes
  float* intermediate_91_data = (float*)(allocator.allocate_at_offset(18874368, 18874368));
  Tensor<float> intermediate_91_tensor = {intermediate_91_data, intermediate_91_shape, 3};
  // Pool: default, Offset: 0, Size: 49152 bytes
  float* intermediate_92_data = (float*)(allocator.allocate_at_offset(0, 49152));
  Tensor<float> intermediate_92_tensor = {intermediate_92_data, intermediate_92_shape, 2};
  // Pool: default, Offset: 49152, Size: 49152 bytes
  float* intermediate_93_data = (float*)(allocator.allocate_at_offset(49152, 49152));
  Tensor<float> intermediate_93_tensor = {intermediate_93_data, intermediate_93_shape, 2};
  // Pool: default, Offset: 98304, Size: 49152 bytes
  float* intermediate_94_data = (float*)(allocator.allocate_at_offset(98304, 49152));
  Tensor<float> intermediate_94_tensor = {intermediate_94_data, intermediate_94_shape, 2};
  // Pool: default, Offset: 37748736, Size: 18874368 bytes
  float* intermediate_95_data = (float*)(allocator.allocate_at_offset(37748736, 18874368));
  Tensor<float> intermediate_95_tensor = {intermediate_95_data, intermediate_95_shape, 4};
  // Pool: default, Offset: 56623104, Size: 18874368 bytes
  float* intermediate_96_data = (float*)(allocator.allocate_at_offset(56623104, 18874368));
  Tensor<float> intermediate_96_tensor = {intermediate_96_data, intermediate_96_shape, 4};
  // Pool: default, Offset: 75497472, Size: 18874368 bytes
  float* intermediate_97_data = (float*)(allocator.allocate_at_offset(75497472, 18874368));
  Tensor<float> intermediate_97_tensor = {intermediate_97_data, intermediate_97_shape, 4};
  // Pool: default, Offset: 94371840, Size: 113246208 bytes
  float* intermediate_98_data = (float*)(allocator.allocate_at_offset(94371840, 113246208));
  Tensor<float> intermediate_98_tensor = {intermediate_98_data, intermediate_98_shape, 4};
  // Pool: default, Offset: 207618048, Size: 113246208 bytes
  float* intermediate_99_data = (float*)(allocator.allocate_at_offset(207618048, 113246208));
  Tensor<float> intermediate_99_tensor = {intermediate_99_data, intermediate_99_shape, 4};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_100_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_100_tensor = {intermediate_100_data, intermediate_100_shape, 4};
  // Pool: default, Offset: 37748736, Size: 18874368 bytes
  float* intermediate_101_data = (float*)(allocator.allocate_at_offset(37748736, 18874368));
  Tensor<float> intermediate_101_tensor = {intermediate_101_data, intermediate_101_shape, 3};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_102_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_102_tensor = {intermediate_102_data, intermediate_102_shape, 3};
  // Pool: default, Offset: 37748736, Size: 18874368 bytes
  float* intermediate_103_data = (float*)(allocator.allocate_at_offset(37748736, 18874368));
  Tensor<float> intermediate_103_tensor = {intermediate_103_data, intermediate_103_shape, 3};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_104_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_104_tensor = {intermediate_104_data, intermediate_104_shape, 3};
  // Pool: default, Offset: 37748736, Size: 196608 bytes
  float* intermediate_105_data = (float*)(allocator.allocate_at_offset(37748736, 196608));
  Tensor<float> intermediate_105_tensor = {intermediate_105_data, intermediate_105_shape, 2};
  // Pool: default, Offset: 37945344, Size: 196608 bytes
  float* intermediate_106_data = (float*)(allocator.allocate_at_offset(37945344, 196608));
  Tensor<float> intermediate_106_tensor = {intermediate_106_data, intermediate_106_shape, 2};
  // Pool: default, Offset: 37748736, Size: 49152 bytes
  float* intermediate_107_data = (float*)(allocator.allocate_at_offset(37748736, 49152));
  Tensor<float> intermediate_107_tensor = {intermediate_107_data, intermediate_107_shape, 2};
  // Pool: default, Offset: 37797888, Size: 18874368 bytes
  float* intermediate_108_data = (float*)(allocator.allocate_at_offset(37797888, 18874368));
  Tensor<float> intermediate_108_tensor = {intermediate_108_data, intermediate_108_shape, 3};
  // Pool: default, Offset: 0, Size: 18874368 bytes
  float* intermediate_109_data = (float*)(allocator.allocate_at_offset(0, 18874368));
  Tensor<float> intermediate_109_tensor = {intermediate_109_data, intermediate_109_shape, 3};
  // Pool: default, Offset: 18874368, Size: 8320 bytes
  float* intermediate_110_data = (float*)(allocator.allocate_at_offset(18874368, 8320));
  Tensor<float> intermediate_110_tensor = {intermediate_110_data, intermediate_110_shape, 2};
  Tensor<int> input = {(int*)input_data, input_shape, input_dims};
  Tensor<float> output = {(float*)output_data, output_shape, output_dims};

  embedding_forward<<<dim3(128, 32), dim3(384, 1, 1), 0>>>(intermediate_0_tensor, input, param_tensors[0]);
  CUDA_CHECK(cudaGetLastError());
  positional_encoding_forward<<<dim3(6144, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_1_tensor, intermediate_0_tensor, param_tensors[1]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_2_tensor, intermediate_1_tensor, param_tensors[2], param_tensors[3]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_3_tensor, intermediate_1_tensor, param_tensors[4], param_tensors[5]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_4_tensor, intermediate_1_tensor, param_tensors[6], param_tensors[7]);
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
  dense_forward_3d<<<dim3(2, 128, 32), dim3(256, 1, 1), 0>>>(intermediate_12_tensor, intermediate_11_tensor, param_tensors[8], param_tensors[9]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(6144, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_13_tensor, intermediate_1_tensor, intermediate_12_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(128, 32), dim3(384, 1, 1), 1536>>>(intermediate_14_tensor, intermediate_13_tensor, param_tensors[10], param_tensors[11]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_15_tensor, intermediate_14_tensor, param_tensors[12], param_tensors[13]);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_16_tensor, intermediate_15_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_17_tensor, intermediate_16_tensor, param_tensors[14], param_tensors[15]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(6144, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_18_tensor, intermediate_14_tensor, intermediate_17_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(128, 32), dim3(384, 1, 1), 1536>>>(intermediate_19_tensor, intermediate_18_tensor, param_tensors[16], param_tensors[17]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_20_tensor, intermediate_19_tensor, param_tensors[18], param_tensors[19]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_21_tensor, intermediate_19_tensor, param_tensors[20], param_tensors[21]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_22_tensor, intermediate_19_tensor, param_tensors[22], param_tensors[23]);
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
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_30_tensor, intermediate_29_tensor, param_tensors[24], param_tensors[25]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_31_tensor, intermediate_19_tensor, intermediate_30_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_32_tensor, intermediate_31_tensor, param_tensors[26], param_tensors[27]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_33_tensor, intermediate_32_tensor, param_tensors[28], param_tensors[29]);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_34_tensor, intermediate_33_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_35_tensor, intermediate_34_tensor, param_tensors[30], param_tensors[31]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_36_tensor, intermediate_32_tensor, intermediate_35_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_37_tensor, intermediate_36_tensor, param_tensors[32], param_tensors[33]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_38_tensor, intermediate_37_tensor, param_tensors[34], param_tensors[35]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_39_tensor, intermediate_37_tensor, param_tensors[36], param_tensors[37]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_40_tensor, intermediate_37_tensor, param_tensors[38], param_tensors[39]);
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
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_48_tensor, intermediate_47_tensor, param_tensors[40], param_tensors[41]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_49_tensor, intermediate_37_tensor, intermediate_48_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_50_tensor, intermediate_49_tensor, param_tensors[42], param_tensors[43]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_51_tensor, intermediate_50_tensor, param_tensors[44], param_tensors[45]);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_52_tensor, intermediate_51_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_53_tensor, intermediate_52_tensor, param_tensors[46], param_tensors[47]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_54_tensor, intermediate_50_tensor, intermediate_53_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_55_tensor, intermediate_54_tensor, param_tensors[48], param_tensors[49]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_56_tensor, intermediate_55_tensor, param_tensors[50], param_tensors[51]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_57_tensor, intermediate_55_tensor, param_tensors[52], param_tensors[53]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_58_tensor, intermediate_55_tensor, param_tensors[54], param_tensors[55]);
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
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_66_tensor, intermediate_65_tensor, param_tensors[56], param_tensors[57]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_67_tensor, intermediate_55_tensor, intermediate_66_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_68_tensor, intermediate_67_tensor, param_tensors[58], param_tensors[59]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_69_tensor, intermediate_68_tensor, param_tensors[60], param_tensors[61]);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_70_tensor, intermediate_69_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_71_tensor, intermediate_70_tensor, param_tensors[62], param_tensors[63]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_72_tensor, intermediate_68_tensor, intermediate_71_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_73_tensor, intermediate_72_tensor, param_tensors[64], param_tensors[65]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_74_tensor, intermediate_73_tensor, param_tensors[66], param_tensors[67]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_75_tensor, intermediate_73_tensor, param_tensors[68], param_tensors[69]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_76_tensor, intermediate_73_tensor, param_tensors[70], param_tensors[71]);
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
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_84_tensor, intermediate_83_tensor, param_tensors[72], param_tensors[73]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_85_tensor, intermediate_73_tensor, intermediate_84_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_86_tensor, intermediate_85_tensor, param_tensors[74], param_tensors[75]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_87_tensor, intermediate_86_tensor, param_tensors[76], param_tensors[77]);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_88_tensor, intermediate_87_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_89_tensor, intermediate_88_tensor, param_tensors[78], param_tensors[79]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_90_tensor, intermediate_86_tensor, intermediate_89_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_91_tensor, intermediate_90_tensor, param_tensors[80], param_tensors[81]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_92_tensor, intermediate_91_tensor, param_tensors[82], param_tensors[83]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_93_tensor, intermediate_91_tensor, param_tensors[84], param_tensors[85]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_94_tensor, intermediate_91_tensor, param_tensors[86], param_tensors[87]);
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
  dense_forward_3d<<<dim3(2, 384, 32), dim3(256, 1, 1), 0>>>(intermediate_102_tensor, intermediate_101_tensor, param_tensors[88], param_tensors[89]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_103_tensor, intermediate_91_tensor, intermediate_102_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_104_tensor, intermediate_103_tensor, param_tensors[90], param_tensors[91]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(6, 32), dim3(256, 1, 1), 0>>>(intermediate_105_tensor, intermediate_104_tensor, param_tensors[92], param_tensors[93]);
  CUDA_CHECK(cudaGetLastError());
  relu_forward<<<dim3(192, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_106_tensor, intermediate_105_tensor);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(2, 32), dim3(256, 1, 1), 0>>>(intermediate_107_tensor, intermediate_106_tensor, param_tensors[94], param_tensors[95]);
  CUDA_CHECK(cudaGetLastError());
  add_forward<<<dim3(18432, 1, 1), dim3(256, 1, 1), 0>>>(intermediate_108_tensor, intermediate_104_tensor, intermediate_107_tensor);
  CUDA_CHECK(cudaGetLastError());
  layer_norm_forward<<<dim3(384, 32), dim3(384, 1, 1), 1536>>>(intermediate_109_tensor, intermediate_108_tensor, param_tensors[96], param_tensors[97]);
  CUDA_CHECK(cudaGetLastError());
  dense_forward_2d<<<dim3(1, 32), dim3(256, 1, 1), 0>>>(intermediate_110_tensor, intermediate_109_tensor, param_tensors[98], param_tensors[99]);
  CUDA_CHECK(cudaGetLastError());
  softmax_forward<<<dim3(32, 1, 1), dim3(96, 1, 1), 384>>>(output, intermediate_110_tensor);
  CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t used_memory = allocator.get_used_size();
    if (used_memory > 0) {
    }
}

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
    const int input_shape[] = {input_batch_size, input_seq_len};
    const int output_shape[] = {output_batch_size, output_vocab_size};
    
        const int param_shape_0[] = {65, 384}; // param_0_embeddings
    const int param_shape_1[] = {192}; // param_1_frequencies
    const int param_shape_2[] = {384, 384}; // param_2_weights
    const int param_shape_3[] = {384}; // param_3_bias
    const int param_shape_4[] = {384, 384}; // param_4_weights
    const int param_shape_5[] = {384}; // param_5_bias
    const int param_shape_6[] = {384, 384}; // param_6_weights
    const int param_shape_7[] = {384}; // param_7_bias
    const int param_shape_8[] = {384, 384}; // param_8_weights
    const int param_shape_9[] = {384}; // param_9_bias
    const int param_shape_10[] = {384}; // param_10_gamma
    const int param_shape_11[] = {384}; // param_11_beta
    const int param_shape_12[] = {384, 1536}; // param_12_weights
    const int param_shape_13[] = {1536}; // param_13_bias
    const int param_shape_14[] = {1536, 384}; // param_14_weights
    const int param_shape_15[] = {384}; // param_15_bias
    const int param_shape_16[] = {384}; // param_16_gamma
    const int param_shape_17[] = {384}; // param_17_beta
    const int param_shape_18[] = {384, 384}; // param_18_weights
    const int param_shape_19[] = {384}; // param_19_bias
    const int param_shape_20[] = {384, 384}; // param_20_weights
    const int param_shape_21[] = {384}; // param_21_bias
    const int param_shape_22[] = {384, 384}; // param_22_weights
    const int param_shape_23[] = {384}; // param_23_bias
    const int param_shape_24[] = {384, 384}; // param_24_weights
    const int param_shape_25[] = {384}; // param_25_bias
    const int param_shape_26[] = {384}; // param_26_gamma
    const int param_shape_27[] = {384}; // param_27_beta
    const int param_shape_28[] = {384, 1536}; // param_28_weights
    const int param_shape_29[] = {1536}; // param_29_bias
    const int param_shape_30[] = {1536, 384}; // param_30_weights
    const int param_shape_31[] = {384}; // param_31_bias
    const int param_shape_32[] = {384}; // param_32_gamma
    const int param_shape_33[] = {384}; // param_33_beta
    const int param_shape_34[] = {384, 384}; // param_34_weights
    const int param_shape_35[] = {384}; // param_35_bias
    const int param_shape_36[] = {384, 384}; // param_36_weights
    const int param_shape_37[] = {384}; // param_37_bias
    const int param_shape_38[] = {384, 384}; // param_38_weights
    const int param_shape_39[] = {384}; // param_39_bias
    const int param_shape_40[] = {384, 384}; // param_40_weights
    const int param_shape_41[] = {384}; // param_41_bias
    const int param_shape_42[] = {384}; // param_42_gamma
    const int param_shape_43[] = {384}; // param_43_beta
    const int param_shape_44[] = {384, 1536}; // param_44_weights
    const int param_shape_45[] = {1536}; // param_45_bias
    const int param_shape_46[] = {1536, 384}; // param_46_weights
    const int param_shape_47[] = {384}; // param_47_bias
    const int param_shape_48[] = {384}; // param_48_gamma
    const int param_shape_49[] = {384}; // param_49_beta
    const int param_shape_50[] = {384, 384}; // param_50_weights
    const int param_shape_51[] = {384}; // param_51_bias
    const int param_shape_52[] = {384, 384}; // param_52_weights
    const int param_shape_53[] = {384}; // param_53_bias
    const int param_shape_54[] = {384, 384}; // param_54_weights
    const int param_shape_55[] = {384}; // param_55_bias
    const int param_shape_56[] = {384, 384}; // param_56_weights
    const int param_shape_57[] = {384}; // param_57_bias
    const int param_shape_58[] = {384}; // param_58_gamma
    const int param_shape_59[] = {384}; // param_59_beta
    const int param_shape_60[] = {384, 1536}; // param_60_weights
    const int param_shape_61[] = {1536}; // param_61_bias
    const int param_shape_62[] = {1536, 384}; // param_62_weights
    const int param_shape_63[] = {384}; // param_63_bias
    const int param_shape_64[] = {384}; // param_64_gamma
    const int param_shape_65[] = {384}; // param_65_beta
    const int param_shape_66[] = {384, 384}; // param_66_weights
    const int param_shape_67[] = {384}; // param_67_bias
    const int param_shape_68[] = {384, 384}; // param_68_weights
    const int param_shape_69[] = {384}; // param_69_bias
    const int param_shape_70[] = {384, 384}; // param_70_weights
    const int param_shape_71[] = {384}; // param_71_bias
    const int param_shape_72[] = {384, 384}; // param_72_weights
    const int param_shape_73[] = {384}; // param_73_bias
    const int param_shape_74[] = {384}; // param_74_gamma
    const int param_shape_75[] = {384}; // param_75_beta
    const int param_shape_76[] = {384, 1536}; // param_76_weights
    const int param_shape_77[] = {1536}; // param_77_bias
    const int param_shape_78[] = {1536, 384}; // param_78_weights
    const int param_shape_79[] = {384}; // param_79_bias
    const int param_shape_80[] = {384}; // param_80_gamma
    const int param_shape_81[] = {384}; // param_81_beta
    const int param_shape_82[] = {384, 384}; // param_82_weights
    const int param_shape_83[] = {384}; // param_83_bias
    const int param_shape_84[] = {384, 384}; // param_84_weights
    const int param_shape_85[] = {384}; // param_85_bias
    const int param_shape_86[] = {384, 384}; // param_86_weights
    const int param_shape_87[] = {384}; // param_87_bias
    const int param_shape_88[] = {384, 384}; // param_88_weights
    const int param_shape_89[] = {384}; // param_89_bias
    const int param_shape_90[] = {384}; // param_90_gamma
    const int param_shape_91[] = {384}; // param_91_beta
    const int param_shape_92[] = {384, 1536}; // param_92_weights
    const int param_shape_93[] = {1536}; // param_93_bias
    const int param_shape_94[] = {1536, 384}; // param_94_weights
    const int param_shape_95[] = {384}; // param_95_bias
    const int param_shape_96[] = {384}; // param_96_gamma
    const int param_shape_97[] = {384}; // param_97_beta
    const int param_shape_98[] = {384, 65}; // param_98_weights
    const int param_shape_99[] = {65}; // param_99_bias
    
    const int* param_shapes[100] = {
        param_shape_0,
        param_shape_1,
        param_shape_2,
        param_shape_3,
        param_shape_4,
        param_shape_5,
        param_shape_6,
        param_shape_7,
        param_shape_8,
        param_shape_9,
        param_shape_10,
        param_shape_11,
        param_shape_12,
        param_shape_13,
        param_shape_14,
        param_shape_15,
        param_shape_16,
        param_shape_17,
        param_shape_18,
        param_shape_19,
        param_shape_20,
        param_shape_21,
        param_shape_22,
        param_shape_23,
        param_shape_24,
        param_shape_25,
        param_shape_26,
        param_shape_27,
        param_shape_28,
        param_shape_29,
        param_shape_30,
        param_shape_31,
        param_shape_32,
        param_shape_33,
        param_shape_34,
        param_shape_35,
        param_shape_36,
        param_shape_37,
        param_shape_38,
        param_shape_39,
        param_shape_40,
        param_shape_41,
        param_shape_42,
        param_shape_43,
        param_shape_44,
        param_shape_45,
        param_shape_46,
        param_shape_47,
        param_shape_48,
        param_shape_49,
        param_shape_50,
        param_shape_51,
        param_shape_52,
        param_shape_53,
        param_shape_54,
        param_shape_55,
        param_shape_56,
        param_shape_57,
        param_shape_58,
        param_shape_59,
        param_shape_60,
        param_shape_61,
        param_shape_62,
        param_shape_63,
        param_shape_64,
        param_shape_65,
        param_shape_66,
        param_shape_67,
        param_shape_68,
        param_shape_69,
        param_shape_70,
        param_shape_71,
        param_shape_72,
        param_shape_73,
        param_shape_74,
        param_shape_75,
        param_shape_76,
        param_shape_77,
        param_shape_78,
        param_shape_79,
        param_shape_80,
        param_shape_81,
        param_shape_82,
        param_shape_83,
        param_shape_84,
        param_shape_85,
        param_shape_86,
        param_shape_87,
        param_shape_88,
        param_shape_89,
        param_shape_90,
        param_shape_91,
        param_shape_92,
        param_shape_93,
        param_shape_94,
        param_shape_95,
        param_shape_96,
        param_shape_97,
        param_shape_98,
        param_shape_99
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
