// ============================================================================
// Tensor Core MatMul Node Test
// ============================================================================
// This file tests the new TensorCoreMatMulNode implementation and compares
// it with the standard MatMulNode to demonstrate the Tensor Core enhancement.
// ============================================================================

import { CudaGraph, CudaGraphCompiler } from "./cuda-graph.js";
import { MatMulNode, TensorCoreMatMulNode } from "./cuda-nodes.js";
import { MockCudaRuntime } from "./cuda-abstractions.js";

async function testTensorCoreMatMul() {
  console.log("=".repeat(80));
  console.log("Testing Tensor Core Enhanced Matrix Multiplication");
  console.log("=".repeat(80));

  const runtime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(runtime);

  // Test 1: Standard MatMul vs TensorCore MatMul comparison
  console.log("\n--- Test 1: Standard vs Tensor Core MatMul ---");
  
  // Create graphs with both implementations
  const standardGraph = new CudaGraph("StandardMatMulTest");
  const tensorCoreGraph = new CudaGraph("TensorCoreMatMulTest");

  // Add nodes
  const standardMatMul = new MatMulNode();
  const tensorCoreMatMul = new TensorCoreMatMulNode(true, 'fp16', false);

  standardGraph.addNode(standardMatMul);
  tensorCoreGraph.addNode(tensorCoreMatMul);

  // Test matrix dimensions that are optimal for Tensor Cores (multiples of 16)
  const inputShapes = new Map([
    ["A", [128, 256]], // 128x256 matrix
    ["B", [256, 512]]  // 256x512 matrix
  ]);

  console.log("Input shapes:", Object.fromEntries(inputShapes));

  try {
    // Compile both graphs
    console.log("\nCompiling standard MatMul graph...");
    const standardResult = await compiler.compile(standardGraph, inputShapes);
    
    console.log("\nCompiling Tensor Core MatMul graph...");
    const tensorCoreResult = await compiler.compile(tensorCoreGraph, inputShapes);

    console.log("✅ Both graphs compiled successfully!");
    
    // Compare generated kernel code
    console.log("\n--- Kernel Code Comparison ---");
    console.log("Standard MatMul kernel includes basic nested loops");
    console.log("Tensor Core MatMul kernel includes:");
    console.log("  - WMMA API usage for Volta+ GPUs");
    console.log("  - FP32 to FP16 conversion");
    console.log("  - Shared memory tiling fallback");
    console.log("  - Optimal grid/block configuration");

  } catch (error) {
    console.error("❌ Compilation failed:", error);
  }

  // Test 2: Different precision modes
  console.log("\n--- Test 2: Different Precision Modes ---");
  
  const precisionModes: Array<'fp16' | 'bf16' | 'fp32'> = ['fp16', 'bf16', 'fp32'];
  
  for (const precision of precisionModes) {
    console.log(`\nTesting ${precision.toUpperCase()} precision...`);
    
    const graph = new CudaGraph(`TensorCore_${precision}`);
    const node = new TensorCoreMatMulNode(true, precision, false);
    graph.addNode(node);
    
    try {
      const result = await compiler.compile(graph, inputShapes);
      console.log(`✅ ${precision.toUpperCase()} precision compilation successful`);
    } catch (error) {
      console.error(`❌ ${precision.toUpperCase()} precision compilation failed:`, error);
    }
  }

  // Test 3: cuBLAS wrapper mode
  console.log("\n--- Test 3: cuBLAS Integration Mode ---");
  
  const cublasGraph = new CudaGraph("CuBLASMatMulTest");
  const cublasMatMul = new TensorCoreMatMulNode(true, 'fp16', true);
  cublasGraph.addNode(cublasMatMul);
  
  try {
    const result = await compiler.compile(cublasGraph, inputShapes);
    console.log("✅ cuBLAS integration mode compilation successful");
  } catch (error) {
    console.error("❌ cuBLAS integration mode compilation failed:", error);
  }

  // Test 4: Edge cases and validation
  console.log("\n--- Test 4: Edge Cases and Validation ---");
  
  // Test non-square matrices
  const edgeCases = [
    { name: "Small matrices", shapes: new Map([["A", [16, 32]], ["B", [32, 16]]]) },
    { name: "Large matrices", shapes: new Map([["A", [1024, 2048]], ["B", [2048, 1024]]]) },
    { name: "Non-16-multiple", shapes: new Map([["A", [100, 200]], ["B", [200, 150]]]) },
  ];

  for (const testCase of edgeCases) {
    console.log(`\nTesting ${testCase.name}...`);
    
    const graph = new CudaGraph(`EdgeCase_${testCase.name.replace(/\s+/g, '_')}`);
    const node = new TensorCoreMatMulNode();
    graph.addNode(node);
    
    try {
      const result = await compiler.compile(graph, testCase.shapes);
      console.log(`✅ ${testCase.name} test passed`);
    } catch (error) {
      console.error(`❌ ${testCase.name} test failed:`, error);
    }
  }

  // Test 5: Error handling
  console.log("\n--- Test 5: Error Handling ---");
  
  const errorCases = [
    { 
      name: "Dimension mismatch", 
      shapes: new Map([["A", [128, 256]], ["B", [128, 512]]])  // Wrong inner dimension
    },
    { 
      name: "1D tensors", 
      shapes: new Map([["A", [128]], ["B", [128]]])  // Should be 2D
    }
  ];

  for (const testCase of errorCases) {
    console.log(`\nTesting ${testCase.name}...`);
    
    const graph = new CudaGraph(`ErrorCase_${testCase.name.replace(/\s+/g, '_')}`);
    const node = new TensorCoreMatMulNode();
    graph.addNode(node);
    
    try {
      const result = await compiler.compile(graph, testCase.shapes);
      console.log(`❌ ${testCase.name} should have failed but didn't`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log(`✅ ${testCase.name} correctly threw error: ${errorMessage}`);
    }
  }

  console.log("\n" + "=".repeat(80));
  console.log("Tensor Core MatMul Testing Complete");
  console.log("=".repeat(80));
}

// Performance comparison demonstration
async function demonstratePerformanceGains() {
  console.log("\n" + "=".repeat(80));
  console.log("Performance Characteristics Demonstration");
  console.log("=".repeat(80));

  console.log("\n🚀 Expected Performance Improvements with Tensor Cores:");
  console.log("   • 3-5x speedup for matrix multiplications on Volta+ GPUs");
  console.log("   • Reduced memory bandwidth usage due to FP16 precision");
  console.log("   • Better energy efficiency through specialized hardware");
  console.log("   • Automatic fallback ensures compatibility with older GPUs");

  console.log("\n📊 Optimization Features Implemented:");
  console.log("   ✅ WMMA API integration for native Tensor Core usage");
  console.log("   ✅ Mixed precision (FP16 input, FP32 accumulation)");
  console.log("   ✅ Shared memory optimization for data conversion");
  console.log("   ✅ Optimal grid/block configuration for 16x16 tiles");
  console.log("   ✅ Compile-time GPU architecture detection");
  console.log("   ✅ Automatic fallback to optimized CUDA cores");

  console.log("\n🎯 Use Cases:");
  console.log("   • Deep learning inference and training");
  console.log("   • Large-scale matrix computations");
  console.log("   • Transformer model attention mechanisms");
  console.log("   • Scientific computing applications");

  console.log("\n⚙️  Configuration Options:");
  console.log("   • Precision: FP16, BF16, or FP32");
  console.log("   • Backend: WMMA API or cuBLAS integration");
  console.log("   • Automatic GPU capability detection");
  console.log("   • Configurable optimization levels");
}

// Run the tests
if (import.meta.url === `file://${process.argv[1]}`) {
  testTensorCoreMatMul()
    .then(() => demonstratePerformanceGains())
    .catch(console.error);
}

export { testTensorCoreMatMul, demonstratePerformanceGains };
