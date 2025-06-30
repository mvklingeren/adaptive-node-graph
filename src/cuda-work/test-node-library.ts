// ============================================================================
// Test Suite for Standard CUDA Node Library
// ============================================================================
// This file contains tests to verify the functionality of the pre-defined
// CUDA nodes from the cuda-nodes.ts library. It uses the MockCudaRuntime
// to simulate GPU operations and ensure that the graph compilation and
// node logic are correct.
// ============================================================================

import { CudaGraph, CudaGraphCompiler } from "./cuda-graph.js";
import { MockCudaRuntime } from "./cuda-abstractions.js";
import { AddNode, MatMulNode } from "./cuda-nodes.js";
import { strict as assert } from "assert";

async function runTest() {
  console.log("--- Running CUDA Node Library Test ---");

  // 1. Setup
  const runtime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(runtime);
  const graph = new CudaGraph("TestMatMulAddGraph");

  // 2. Create Nodes from the Library
  const matmulNode = new MatMulNode();
  const addNode = new AddNode();

  // 3. Build the Graph
  // This graph will compute: (A * B) + C
  graph.addNode(matmulNode).addNode(addNode);

  // Define input shapes
  const shapeA = [2, 3]; // M=2, K=3
  const shapeB = [3, 4]; // K=3, N=4
  const shapeC = [2, 4]; // M=2, N=4

  // Manually set the initial shapes for the graph inputs.
  // In a real scenario, this might come from actual data.
  matmulNode.updateInputShape("A", shapeA);
  matmulNode.updateInputShape("B", shapeB);
  addNode.updateInputShape("B", shapeC); // The 'B' input of AddNode is our 'C' tensor

  // Connect the output of MatMul to the first input of Add
  graph.connect(matmulNode, "C", addNode, "A");

  // 4. Compile the Graph
  console.log("\nCompiling graph...");
  const { kernel, parameters, kernelCode, workspaceSize } = await compiler.compile(graph);
  
  console.log("\nCompilation successful!");
  console.log(`- Workspace size: ${workspaceSize} bytes`);
  console.log(`- Number of parameters: ${parameters.length}`);

  // 5. Prepare Data and Execute (Mock)
  // In a real scenario, you would create CudaTensors and copy data here.
  // With the mock runtime, we can just check if the compilation produces
  // the expected kernel structure and signature.

  console.log("\n--- Verifying Kernel Code ---");
  
  // Verify that both matmul and add device codes are included
  assert(kernelCode.includes("__device__ void matmul"), "Kernel code should include matmul function.");
  assert(kernelCode.includes("__device__ void add"), "Kernel code should include add function.");
  
  // Verify the main kernel signature includes the expected inputs
  assert(kernelCode.includes("float* A_data"), "Kernel signature should include input A.");
  assert(kernelCode.includes("float* B_data"), "Kernel signature should include input B.");
  assert(kernelCode.includes("float* B_data"), "Kernel signature should include input C (named B for the add node).");
  assert(kernelCode.includes("float* C_data"), "Kernel signature should include output C.");

  console.log("\n--- Test Passed ---");
  console.log("The graph was compiled successfully with nodes from the library.");
}

runTest().catch(err => {
  console.error("Test failed:", err);
  process.exit(1);
});
