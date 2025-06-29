
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
  float* input_data,
  const int* input_shape,
  int input_dims,
  float* output_data,
  const int* output_shape,
  int output_dims,
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


  // --- Tensor Struct Instantiation ---
  Tensor<float> input = {(float*)input_data, input_shape, input_dims};
  Tensor<float> output = {output_data, output_shape, output_dims};

  // --- Kernel Launch Sequence ---
  softmax_forward<<<dim3(1, 1, 1), dim3(32, 1, 1), 128>>>(output, input);
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
    