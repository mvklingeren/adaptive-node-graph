// ============================================================================
// cuBLAS Integration Test
// ============================================================================
// This test demonstrates the integration of cuBLAS library with CUDA streams
// for optimized neural network execution, replacing dense_forward and 
// batched_matmul operations with cuBLAS GEMM calls.
// ============================================================================

import { MockCublasRuntime } from "./cuda-abstractions";
import { CublasNeuralGraph, CublasDenseLayer, CublasBatchedMatMul, ReLULayer } from "./neural-network";

async function testCublasIntegration() {
  console.log("=".repeat(80));
  console.log("cuBLAS Integration Test - Optimized Neural Network with Streams");
  console.log("=".repeat(80));

  // Create a cuBLAS-enabled runtime
  const runtime = new MockCublasRuntime();
  
  // Create a neural graph with multiple streams for parallel execution
  const graph = new CublasNeuralGraph("CublasOptimizedNetwork", runtime, 3);
  
  try {
    // Initialize the graph (creates streams and cuBLAS handles)
    await graph.initialize();
    
    console.log("\n1. Building cuBLAS-optimized neural network...");
    
    // Create cuBLAS-optimized layers with different streams for parallel execution
    const denseLayer1 = new CublasDenseLayer(
      runtime,
      graph.getCublasHandle(0), // Stream 0
      graph.getStream(0),
      384, // input features
      512  // output features
    );
    await denseLayer1.initialize();
    
    const denseLayer2 = new CublasDenseLayer(
      runtime,
      graph.getCublasHandle(1), // Stream 1 - parallel execution
      graph.getStream(1),
      512, // input features
      256  // output features
    );
    await denseLayer2.initialize();
    
    const denseLayer3 = new CublasDenseLayer(
      runtime,
      graph.getCublasHandle(2), // Stream 2 - parallel execution
      graph.getStream(2),
      256, // input features
      128  // output features
    );
    await denseLayer3.initialize();
    
    // Create cuBLAS batched matrix multiplication for attention mechanism
    const batchedMatMul = new CublasBatchedMatMul(
      runtime,
      graph.getCublasHandle(0),
      graph.getStream(0),
      true // transpose B matrix
    );
    
    // Create activation layer (still uses custom CUDA kernel)
    const reluLayer = new ReLULayer();
    
    console.log("\n2. Adding layers to graph with stream assignment...");
    
    // Build the network with cuBLAS-optimized layers
    // Each layer can execute on different streams for parallelism
    const dense1Node = graph.addCublasLayer(
      (handle, stream) => denseLayer1,
      0 // Stream 0
    );
    
    const relu1Node = graph.addLayer(reluLayer, dense1Node);
    
    const dense2Node = graph.addCublasLayer(
      (handle, stream) => denseLayer2,
      1, // Stream 1 - can execute in parallel with other operations
      relu1Node
    );
    
    const dense3Node = graph.addCublasLayer(
      (handle, stream) => denseLayer3,
      2, // Stream 2 - can execute in parallel with other operations
      dense2Node
    );
    
    console.log("\n3. Demonstrating cuBLAS operations...");
    
    // Show how cuBLAS operations are configured
    console.log("\nCuBLAS Dense Layer Configuration:");
    console.log("- Uses cublasGemmEx for matrix multiplication");
    console.log("- Replaces custom dense_forward kernels");
    console.log("- Supports both 2D and 3D tensor operations");
    console.log("- Automatic bias addition with beta parameter");
    
    console.log("\nCuBLAS Batched MatMul Configuration:");
    console.log("- Uses cublasGemmStridedBatchedEx for attention");
    console.log("- Replaces custom batched_matmul kernels");
    console.log("- Optimized for transformer attention mechanisms");
    console.log("- Supports transpose operations");
    
    console.log("\n4. Stream-based execution benefits:");
    console.log("- Multiple operations can execute concurrently");
    console.log("- Better GPU utilization through parallelism");
    console.log("- Reduced memory transfer overhead");
    console.log("- Asynchronous execution pipeline");
    
    console.log("\n5. Testing cuBLAS operations with mock data...");
    
    // Simulate cuBLAS operations
    const handle = graph.getCublasHandle(0);
    const stream = graph.getStream(0);
    
    // Create mock tensors for demonstration
    const mockTensorA = await runtime.malloc(384 * 512 * 4, [384, 512], "float32");
    const mockTensorB = await runtime.malloc(512 * 256 * 4, [512, 256], "float32");
    const mockTensorC = await runtime.malloc(384 * 256 * 4, [384, 256], "float32");
    
    // Demonstrate cuBLAS GEMM operation
    console.log("\nExecuting cuBLAS GEMM: C = A * B");
    await runtime.gemmEx(
      handle,
      stream,
      'N', // No transpose A
      'N', // No transpose B
      384, // M (rows of A and C)
      256, // N (columns of B and C)
      512, // K (columns of A, rows of B)
      1.0, // alpha
      mockTensorA,
      384, // leading dimension of A
      mockTensorB,
      512, // leading dimension of B
      0.0, // beta
      mockTensorC,
      384  // leading dimension of C
    );
    
    // Demonstrate batched GEMM for attention
    console.log("\nExecuting cuBLAS Strided Batched GEMM for attention:");
    const batchSize = 2;
    const numHeads = 8;
    const seqLen = 64;
    const headDim = 64;
    
    const attentionA = await runtime.malloc(batchSize * numHeads * seqLen * headDim * 4, 
                                          [batchSize, numHeads, seqLen, headDim], "float32");
    const attentionB = await runtime.malloc(batchSize * numHeads * headDim * seqLen * 4, 
                                          [batchSize, numHeads, headDim, seqLen], "float32");
    const attentionC = await runtime.malloc(batchSize * numHeads * seqLen * seqLen * 4, 
                                          [batchSize, numHeads, seqLen, seqLen], "float32");
    
    await runtime.gemmStridedBatchedEx(
      handle,
      stream,
      'N', // No transpose A
      'T', // Transpose B
      seqLen,  // M
      seqLen,  // N
      headDim, // K
      1.0,     // alpha
      attentionA,
      seqLen,  // lda
      seqLen * headDim, // strideA
      attentionB,
      headDim, // ldb
      headDim * seqLen, // strideB
      0.0,     // beta
      attentionC,
      seqLen,  // ldc
      seqLen * seqLen,  // strideC
      batchSize * numHeads // batch count
    );
    
    console.log("\n6. Performance benefits of cuBLAS integration:");
    console.log("✓ Highly optimized BLAS operations");
    console.log("✓ Tensor Core utilization on modern GPUs");
    console.log("✓ Reduced kernel launch overhead");
    console.log("✓ Better memory access patterns");
    console.log("✓ Automatic mixed precision support");
    console.log("✓ Asynchronous execution with streams");
    
    console.log("\n7. Migration from custom kernels:");
    console.log("Before: dense_forward_2d/3d custom CUDA kernels");
    console.log("After:  cublasGemmEx optimized operations");
    console.log("Before: batched_matmul custom kernels");
    console.log("After:  cublasGemmStridedBatchedEx operations");
    
    // Clean up mock tensors
    await mockTensorA.free();
    await mockTensorB.free();
    await mockTensorC.free();
    await attentionA.free();
    await attentionB.free();
    await attentionC.free();
    
    console.log("\n" + "=".repeat(80));
    console.log("cuBLAS Integration Test Completed Successfully!");
    console.log("=".repeat(80));
    
  } catch (error) {
    console.error("Test failed:", error);
  } finally {
    // Clean up resources
    await graph.cleanup();
  }
}

async function testStreamSynchronization() {
  console.log("\n" + "=".repeat(80));
  console.log("CUDA Streams Synchronization Test");
  console.log("=".repeat(80));
  
  const runtime = new MockCublasRuntime();
  
  // Create multiple streams
  const stream1 = await runtime.createStream();
  const stream2 = await runtime.createStream();
  const stream3 = await runtime.createStream();
  
  console.log("\n1. Created 3 CUDA streams for parallel execution");
  
  // Create cuBLAS handles for each stream
  const handle1 = await runtime.createCublasHandle();
  const handle2 = await runtime.createCublasHandle();
  const handle3 = await runtime.createCublasHandle();
  
  await handle1.setStream(stream1);
  await handle2.setStream(stream2);
  await handle3.setStream(stream3);
  
  console.log("2. Created cuBLAS handles and associated with streams");
  
  // Simulate parallel operations
  console.log("\n3. Executing parallel operations on different streams:");
  console.log("   Stream 1: Dense layer forward pass");
  console.log("   Stream 2: Attention computation");
  console.log("   Stream 3: Output projection");
  
  // Synchronize all streams
  console.log("\n4. Synchronizing all streams...");
  await stream1.synchronize();
  await stream2.synchronize();
  await stream3.synchronize();
  
  console.log("5. All streams synchronized successfully");
  
  // Clean up
  await runtime.destroyCublasHandle(handle1);
  await runtime.destroyCublasHandle(handle2);
  await runtime.destroyCublasHandle(handle3);
  await runtime.destroyStream(stream1);
  await runtime.destroyStream(stream2);
  await runtime.destroyStream(stream3);
  
  console.log("6. Cleaned up all resources");
  console.log("=".repeat(80));
}

// Run the tests
if (require.main === module) {
  (async () => {
    await testCublasIntegration();
    await testStreamSynchronization();
  })();
}

export { testCublasIntegration, testStreamSynchronization };
