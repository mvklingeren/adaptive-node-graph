// ============================================================================
// Test and Example Usage for the Language Model
// ============================================================================

import { MockCudaRuntime } from "../cuda-abstractions";
import { CudaGraphCompiler } from "../cuda-graph";
import { LanguageModel } from "./language-model";
import * as fs from "fs";
import * as path from "path";

async function main() {
  console.log("Initializing CUDA Runtime and Compiler...");
  const runtime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(runtime);

  console.log("\nBuilding a Language Model...");
  const vocabSize = 1000;
  const embedDim = 128;
  const numHeads = 4;
  const numLayers = 2;
  const ffnHiddenDim = 512;
  const maxLen = 256;

  const model = new LanguageModel(
    runtime,
    vocabSize,
    embedDim,
    numHeads,
    numLayers,
    ffnHiddenDim,
    maxLen
  );

  console.log("Language Model built successfully.");

  console.log("\nInitializing model parameters...");
  await model.initialize();

  console.log("\nBuilding model graph...");
  model.build();

  const batchSize = 1;
  const inputShapes = new Map([["input", [batchSize, maxLen]]]);

  console.log("\nCompiling graph to a single CUDA kernel...");
  const { kernel, parameters, kernelCode, workspaceSize } =
    await compiler.compile(model, inputShapes);

  console.log("\nCompilation complete.");
  console.log(`- Generated Kernel ID: ${kernel.id}`);
  console.log(`- Number of parameters to pass: ${parameters.length}`);
  console.log(`- Required workspace size: ${workspaceSize} bytes`);

  const outputPath = path.join(process.cwd(), "generated-llm-kernel.cu");
  fs.writeFileSync(outputPath, kernelCode);
  console.log(`\nKernel code written to ${outputPath}`);

  console.log(
    "\nThis demonstrates the successful construction and compilation of a basic language model."
  );
}

main().catch(console.error);
