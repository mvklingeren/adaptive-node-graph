// ============================================================================
// Test and Example Usage for the Neural Graph System
// ============================================================================
// This file demonstrates how to use the CudaGraph and NeuralNetwork
// components to build, compile, and (theoretically) execute a neural
// network on the GPU.
// ============================================================================

import { MockCudaRuntime } from "./cuda-abstractions.js";
import { CudaGraphCompiler } from "./cuda-graph.js";
import { NeuralGraph, DenseLayer, ReLULayer } from "./neural-network.js";
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
  
  const batchSize = 64;
  const inputShapes = new Map([
    ["input", [batchSize, 784]],
  ]);
  
  // The compiler now returns the required workspace size for intermediate tensors.
  const { kernel, parameters, kernelCode, workspaceSize } = await compiler.compile(model, inputShapes);

  console.log("\nCompilation complete.");
  console.log(`- Generated Kernel ID: ${kernel.id}`);
  console.log(`- Number of parameters to pass: ${parameters.length}`);
  console.log(`- Required workspace size: ${workspaceSize} bytes`);
  
  // The `compiler.compile` method already logs the generated kernel code via
  // the MockCudaRuntime, so we can see the output in the console.

  // 6. Write the generated kernel code to a file for inspection and compilation.
  const outputPath = path.join(process.cwd(), 'generated-kernel.cu');
  fs.writeFileSync(outputPath, kernelCode);
  console.log(`\nKernel code written to ${outputPath}`);

  // 7. Prepare for execution (simulation).
  console.log("\nSimulating execution...");
  const inputData = new Float32Array(batchSize * 784).fill(1.0);
  const outputData = new Float32Array(batchSize * 10);

  // Allocate GPU memory for input and output, now with shape information.
  const d_input = await runtime.malloc(inputData.byteLength, [batchSize, 784], "float32");
  const d_output = await runtime.malloc(outputData.byteLength, [batchSize, 10], "float32");

  // Copy input data to the GPU.
  await runtime.memcpyHostToDevice(d_input, Buffer.from(inputData.buffer));

  // 8. Launch the kernel.
  // The new kernel expects a flattened list of arguments for each tensor:
  // (data_pointer, shape_pointer, dimensions). We need to prepare this.
  const allArgs: any[] = [];

  // Helper to create GPU buffers for tensor metadata
  async function prepareTensorArgs(tensor: any) {
      const shapeBuffer = Buffer.from(new Int32Array(tensor.shape).buffer);
      const d_shape = await runtime.malloc(shapeBuffer.byteLength, tensor.shape, "int32");
      await runtime.memcpyHostToDevice(d_shape, shapeBuffer);
      // In a real implementation, the kernel launch would handle these raw values.
      // For the mock, we pass the tensor objects themselves for clarity in logging.
      // A real C++ bridge would flatten this to [tensor.data, d_shape, tensor.shape.length]
      return [tensor, d_shape, tensor.shape.length];
  }

  // The argument order must match the kernel signature: inputs, outputs, then params.
  const inputArgs = await prepareTensorArgs(d_input);
  const outputArgs = await prepareTensorArgs(d_output);
  
  allArgs.push(...inputArgs);
  allArgs.push(...outputArgs);

  for (const param of parameters) {
      const paramArgs = await prepareTensorArgs(param);
      allArgs.push(...paramArgs);
  }

  console.log(`Launching kernel with ${allArgs.length} total flattened arguments.`);
  
  // The mock launch function doesn't use the flattened args, but a real one would.
  // We pass the original tensors for the mock's logging to remain readable.
  const mockLaunchArgs = [d_input, d_output, ...parameters];

  if (workspaceSize > 0) {
    console.log(`Allocating ${workspaceSize} byte workspace.`);
    const d_workspace = await runtime.malloc(workspaceSize);
    mockLaunchArgs.push(d_workspace);
  }

  await kernel.launch(
    { x: Math.ceil(batchSize / 256), y: 1, z: 1 }, // Grid dimensions
    { x: 256, y: 1, z: 1 }, // Block dimensions
    0, // Shared memory
    mockLaunchArgs
  );

  console.log("\nKernel launch simulated.");
  console.log("The full forward pass is represented by a single kernel launch.");
  console.log("This demonstrates the power of graph compilation for performance.");
}

main().catch(console.error);
