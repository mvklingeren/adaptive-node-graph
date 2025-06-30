// ============================================================================
// Test for Embedding Layer Dynamic Shape Fix
// ============================================================================

import { MockCudaRuntime } from "../cuda-abstractions";
import { EmbeddingLayer, PositionalEncodingLayer } from "./embedding";
import { NeuralGraph } from "../neural-network";

async function testEmbeddingDynamicShapes() {
  console.log("Testing Embedding Layer Dynamic Shape Resolution...");
  
  const runtime = new MockCudaRuntime();

  // Test parameters
  const vocabSize = 1000;
  const embedDim = 384;
  const maxLen = 512; // Maximum allowed sequence length
  
  // Create embedding layer
  const embeddingLayer = new EmbeddingLayer(runtime, vocabSize, embedDim, maxLen);
  await embeddingLayer.initialize();
  
  // Create positional encoding layer
  const posEncodingLayer = new PositionalEncodingLayer(runtime, maxLen, embedDim);
  await posEncodingLayer.initialize();

  // Test with different sequence lengths
  const testCases = [
    { batchSize: 32, seqLen: 128, description: "Short sequence (128)" },
    { batchSize: 32, seqLen: 256, description: "Medium sequence (256)" },
    { batchSize: 32, seqLen: 384, description: "Long sequence (384)" },
    { batchSize: 16, seqLen: 512, description: "Maximum sequence (512)" },
  ];

  for (const testCase of testCases) {
    console.log(`\nTesting ${testCase.description}:`);
    
    try {
      // Create a neural graph for this test
      const graph = new NeuralGraph("EmbeddingTest");
      
      // Add embedding layer to graph
      const embeddingNode = embeddingLayer.addToGraph(graph);
      
      // Add positional encoding layer
      const posEncodingNode = posEncodingLayer.addToGraph(graph, embeddingNode);
      
      // Simulate shape resolution with concrete input shape
      const mockInputs = new Map([
        ["input", { shape: [testCase.batchSize, testCase.seqLen] }]
      ]);
      
      // Test embedding layer shape resolution
      const embeddingShapeResolver = (embeddingNode as any).shapeResolver;
      const embeddingOutputShapes = embeddingShapeResolver(mockInputs);
      
      if (embeddingOutputShapes.has("output")) {
        const outputShape = embeddingOutputShapes.get("output").shape;
        console.log(`  Embedding output shape: [${outputShape.join(", ")}]`);
        
        // Verify the shape is correct
        const expectedShape = [testCase.batchSize, testCase.seqLen, embedDim];
        const isCorrect = outputShape.length === expectedShape.length &&
                         outputShape.every((dim: number, i: number) => dim === expectedShape[i]);
        
        if (isCorrect) {
          console.log(`  ✅ Shape resolution correct`);
        } else {
          console.log(`  ❌ Shape resolution incorrect. Expected: [${expectedShape.join(", ")}]`);
        }
      } else {
        console.log(`  ❌ No output shape resolved`);
      }
      
      // Test positional encoding layer shape resolution
      const posEncodingInputs = new Map([
        ["input", { shape: [testCase.batchSize, testCase.seqLen, embedDim] }]
      ]);
      
      const posEncodingShapeResolver = (posEncodingNode as any).shapeResolver;
      const posEncodingOutputShapes = posEncodingShapeResolver(posEncodingInputs);
      
      if (posEncodingOutputShapes.has("output")) {
        const outputShape = posEncodingOutputShapes.get("output").shape;
        console.log(`  Positional encoding output shape: [${outputShape.join(", ")}]`);
        
        // Verify the shape matches input (pass-through)
        const expectedShape = [testCase.batchSize, testCase.seqLen, embedDim];
        const isCorrect = outputShape.length === expectedShape.length &&
                         outputShape.every((dim: number, i: number) => dim === expectedShape[i]);
        
        if (isCorrect) {
          console.log(`  ✅ Positional encoding shape correct`);
        } else {
          console.log(`  ❌ Positional encoding shape incorrect. Expected: [${expectedShape.join(", ")}]`);
        }
      }
      
    } catch (error) {
      console.log(`  ❌ Error: ${(error as Error).message}`);
    }
  }

  // Test sequence length validation (should throw error)
  console.log(`\nTesting sequence length validation (should fail):`);
  try {
    const graph = new NeuralGraph("ValidationTest");
    const embeddingNode = embeddingLayer.addToGraph(graph);
    
    const invalidInputs = new Map([
      ["input", { shape: [32, 600] }] // Exceeds maxLen of 512
    ]);
    
    const embeddingShapeResolver = (embeddingNode as any).shapeResolver;
    embeddingShapeResolver(invalidInputs);
    
    console.log(`  ❌ Validation failed - should have thrown error`);
  } catch (error) {
    console.log(`  ✅ Validation working: ${(error as Error).message}`);
  }

  // MockCudaRuntime doesn't have a cleanup method, so we skip it
  console.log("\nEmbedding layer dynamic shape test completed!");
}

// Run the test
if (require.main === module) {
  testEmbeddingDynamicShapes().catch(console.error);
}

export { testEmbeddingDynamicShapes };
