// ============================================================================
// Test and Example Usage for the Neural Graph System
// ============================================================================
// This file demonstrates how to use the CudaGraph and NeuralNetwork
// components to build, compile, and (theoretically) execute a neural
// network on the GPU.
// ============================================================================

import { MockCudaRuntime } from "./cuda-abstractions";
import { CudaGraphCompiler } from "./cuda-graph";
import { NeuralGraph, DenseLayer, ReLULayer } from "./neural-network";
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  console.log("Initializing CUDA Runtime and Compiler...");
  // Use the MockCudaRuntime for development without a native addon.
  const runtime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(runtime);

  console.log("\nBuilding a simple Neural Graph...");
  // 1. Create a new NeuralGraph.
  const model = new NeuralGraph("SimpleMLP");

  // 2. Create the layers.
  const denseLayer1 = new DenseLayer(runtime, 784, 256);
  const reluLayer1 = new ReLULayer();
  const denseLayer2 = new DenseLayer(runtime, 256, 10);

  // 3. Initialize the stateful layers (allocates weights on GPU).
  console.log("Initializing layer parameters on the mock GPU...");
  await denseLayer1.initialize();
  await denseLayer2.initialize();

  // 4. Add layers to the model. They are automatically connected in sequence.
  model.addLayer(denseLayer1);
  model.addLayer(reluLayer1);
  model.addLayer(denseLayer2);
  console.log("Neural Graph built successfully.");

  // 5. Compile the entire graph into a single CUDA kernel.
  console.log("\nCompiling graph to a single CUDA kernel...");
  // The compiler now infers the graph's inputs and outputs automatically.
  const { kernel, parameters, kernelCode } = await compiler.compile(model);

  console.log("\nCompilation complete.");
  console.log(`- Generated Kernel ID: ${kernel.id}`);
  console.log(`- Number of parameters to pass: ${parameters.length}`);
  
  // The `compiler.compile` method already logs the generated kernel code via
  // the MockCudaRuntime, so we can see the output in the console.

  // 6. Write the generated kernel code to a file for inspection and compilation.
  const outputPath = path.join(process.cwd(), 'generated-kernel.cu');
  fs.writeFileSync(outputPath, kernelCode);
  console.log(`\nKernel code written to ${outputPath}`);

  // 7. Prepare for execution (simulation).
  console.log("\nSimulating execution...");
  const batchSize = 64;
  const inputData = new Float32Array(batchSize * 784).fill(1.0);
  const outputData = new Float32Array(batchSize * 10);

  // Allocate GPU memory for input and output, now with shape information.
  const d_input = await runtime.malloc(inputData.byteLength, [batchSize, 784], "float32");
  const d_output = await runtime.malloc(outputData.byteLength, [batchSize, 10], "float32");

  // Copy input data to the GPU.
  await runtime.memcpyHostToDevice(d_input, Buffer.from(inputData.buffer));

  // 8. Launch the kernel.
  // The arguments must now be ordered correctly: graph inputs, then graph
  // outputs, then the collected parameters. Our simple sequential model has
  // one input ('input') and one output ('output').
  const allArgs = [d_input, d_output, ...parameters];
  
  console.log(`Launching kernel with ${allArgs.length} total arguments.`);
  await kernel.launch(
    { x: Math.ceil(batchSize / 256), y: 1, z: 1 }, // Grid dimensions
    { x: 256, y: 1, z: 1 }, // Block dimensions
    0, // Shared memory
    allArgs
  );

  console.log("\nKernel launch simulated.");
  console.log("The full forward pass is represented by a single kernel launch.");
  console.log("This demonstrates the power of graph compilation for performance.");
}

main().catch(console.error);
