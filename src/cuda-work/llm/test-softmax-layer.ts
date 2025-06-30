// ============================================================================
// Test for the Softmax Layer
// ============================================================================

import { MockCudaRuntime } from "../cuda-abstractions";
import { CudaGraphCompiler } from "../cuda-graph";
import { NeuralGraph } from "../neural-network";
import { SoftmaxLayer } from "./llm-layers";
import * as fs from "fs";
import * as path from "path";

async function main() {
  console.log("Initializing CUDA Runtime and Compiler...");
  const runtime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(runtime);

  console.log("\nBuilding a graph with a single Softmax Layer...");
  const neuralGraph = new NeuralGraph("SoftmaxTestGraph");
  
  // The compiler needs an input shape. We define a placeholder for the graph's input.
  // The SoftmaxLayer itself will create the CudaNode with the correct inputs.
  const softmaxLayer = new SoftmaxLayer();
  neuralGraph.addLayer(softmaxLayer); // This will be the graph's input layer

  console.log("Graph built successfully.");

  const inputShapes = new Map([["input", [1, 10]]]);

  console.log("\nCompiling graph to a single CUDA kernel...");
  try {
    const { kernel, parameters, kernelCode, workspaceSize } =
      await compiler.compile(neuralGraph, inputShapes);

    console.log("\nCompilation complete.");
    console.log(`- Generated Kernel ID: ${kernel.id}`);
    console.log(`- Number of parameters to pass: ${parameters.length}`);
    console.log(`- Required workspace size: ${workspaceSize} bytes`);

    const outputPath = path.join(process.cwd(), "generated-softmax-kernel.cu");
    fs.writeFileSync(outputPath, kernelCode);
    console.log(`\nKernel code written to ${outputPath}`);
    console.log("\n✅ Verification successful! The softmax kernel compiled without errors.");

  } catch (e) {
    console.error("\n❌ Verification failed! The softmax kernel failed to compile.");
    console.error(e);
  }
}

main().catch(console.error);
