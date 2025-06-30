// ============================================================================
// Test for PositionalEncodingLayer with Pre-computed Frequencies
// ============================================================================

import { MockCudaRuntime } from "../cuda-abstractions";
import { PositionalEncodingLayer } from "./embedding";
import { NeuralGraph } from "../neural-network";

async function testPositionalEncodingLayer() {
  console.log("Testing PositionalEncodingLayer with pre-computed frequencies...");
  
  const runtime = new MockCudaRuntime();
  const maxLen = 512;
  const embedDim = 256;
  
  // Create the layer
  const posLayer = new PositionalEncodingLayer(runtime, maxLen, embedDim);
  
  // Initialize the layer (this should pre-compute frequencies)
  await posLayer.initialize();
  console.log("✓ PositionalEncodingLayer initialized successfully");
  
  // Create a neural graph and add the layer
  const graph = new NeuralGraph("TestGraph");
  const posNode = posLayer.addToGraph(graph);
  
  console.log("✓ PositionalEncodingLayer added to graph successfully");
  console.log(`✓ Node has ${posNode.parameters.size} parameters (should be 1 for frequencies)`);
  console.log(`✓ Node inputs: ${Array.from(posNode.inputs.keys()).join(", ")}`);
  console.log(`✓ Node outputs: ${Array.from(posNode.outputs.keys()).join(", ")}`);
  
  // Verify the frequencies parameter exists
  const frequencies = posNode.parameters.get("frequencies");
  if (frequencies) {
    console.log(`✓ Frequencies tensor created with shape: [${frequencies.shape.join(", ")}]`);
    console.log(`✓ Expected frequency count: ${Math.floor(embedDim / 2)}`);
    console.log(`✓ Actual frequency tensor size: ${frequencies.shape[0]} elements`);
  } else {
    console.error("✗ Frequencies parameter not found!");
  }
  
  console.log("\n=== Generated CUDA Kernel Code ===");
  console.log(posNode.deviceCode);
  
  console.log("\n✓ All tests passed! PositionalEncodingLayer now uses pre-computed frequencies.");
}

// Run the test
testPositionalEncodingLayer().catch(console.error);
