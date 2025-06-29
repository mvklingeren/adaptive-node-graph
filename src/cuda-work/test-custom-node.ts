// ============================================================================
// Test and Example Usage for Custom CudaNodes
// ============================================================================
// This file demonstrates how to create and use custom CudaNodes directly,
// without the higher-level abstractions of the NeuralNetwork layers.
// ============================================================================

import { MockCudaRuntime } from "./cuda-abstractions.js";
import { CudaGraph, CudaNode, CudaGraphCompiler } from "./cuda-graph.js";
import * as fs from 'fs';
import * as path from 'path';

async function main() {
  console.log("Initializing CUDA Runtime and Compiler...");
  const runtime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(runtime);

  // --- Example 1: A simple "add" node ---
  console.log("\n--- Example 1: Simple 'add' graph ---");
  const addGraph = new CudaGraph("AddGraph");
  const addDeviceCode = `
    __global__ void add_kernel(Tensor<float> out, const Tensor<float> a, const Tensor<float> b) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < out.shape[0]) {
        out(i) = a(i) + b(i);
      }
    }
  `;
  const addNode = new CudaNode(addDeviceCode, "add_kernel")
    .addInput('a', [1024], 'float32')
    .addInput('b', [1024], 'float32')
    .addOutput('out', [1024], 'float32');
  addGraph.addNode(addNode);

  const { kernelCode: addKernelCode } = await compiler.compile(addGraph, new Map());
  const addOutputPath = path.join(process.cwd(), 'generated-add-kernel.cu');
  fs.writeFileSync(addOutputPath, addKernelCode);
  console.log(`'add' kernel code written to ${addOutputPath}`);


  // --- Example 2: A graph with two connected nodes (add -> scale) ---
  console.log("\n--- Example 2: 'add' -> 'scale' graph ---");
  const addScaleGraph = new CudaGraph("AddScaleGraph");
  const scaleDeviceCode = `
    __global__ void scale_kernel(Tensor<float> out, const Tensor<float> in, const Tensor<float> scale_factor) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < out.shape[0]) {
        out(i) = in(i) * scale_factor(0);
      }
    }
  `;
  const addNode2 = new CudaNode(addDeviceCode, "add_kernel")
    .addInput('a', [1024], 'float32')
    .addInput('b', [1024], 'float32')
    .addOutput('out', [1024], 'float32');
  
  const scaleFactor = await runtime.malloc(4, [1], 'float32');
  const scaleNode = new CudaNode(scaleDeviceCode, "scale_kernel")
    .addInput('in', [1024], 'float32')
    .addOutput('out', [1024], 'float32')
    .addParameter('scale_factor', scaleFactor);

  addScaleGraph.addNode(addNode2).addNode(scaleNode);
  addScaleGraph.connect(addNode2, 'out', scaleNode, 'in');

  const { kernelCode: addScaleKernelCode } = await compiler.compile(addScaleGraph, new Map());
  const addScaleOutputPath = path.join(process.cwd(), 'generated-add-scale-kernel.cu');
  fs.writeFileSync(addScaleOutputPath, addScaleKernelCode);
  console.log(`'add-scale' kernel code written to ${addScaleOutputPath}`);


  // --- Example 3: A graph with a fan-out structure ---
  console.log("\n--- Example 3: Fan-out graph ---");
  const fanOutGraph = new CudaGraph("FanOutGraph");
  const addNode3 = new CudaNode(addDeviceCode, "add_kernel")
    .addInput('a', [1024], 'float32')
    .addInput('b', [1024], 'float32')
    .addOutput('sum', [1024], 'float32');
  
  const scaleNode2 = new CudaNode(scaleDeviceCode, "scale_kernel")
    .addInput('in', [1024], 'float32')
    .addOutput('out', [1024], 'float32')
    .addParameter('scale_factor', scaleFactor);
  
  const scaleNode3 = new CudaNode(scaleDeviceCode, "scale_kernel")
    .addInput('in', [1024], 'float32')
    .addOutput('out', [1024], 'float32')
    .addParameter('scale_factor', scaleFactor);

  fanOutGraph.addNode(addNode3).addNode(scaleNode2).addNode(scaleNode3);
  fanOutGraph.connect(addNode3, 'sum', scaleNode2, 'in');
  fanOutGraph.connect(addNode3, 'sum', scaleNode3, 'in');

  const { kernelCode: fanOutKernelCode } = await compiler.compile(fanOutGraph, new Map());
  const fanOutOutputPath = path.join(process.cwd(), 'generated-fan-out-kernel.cu');
  fs.writeFileSync(fanOutOutputPath, fanOutKernelCode);
  console.log(`'fan-out' kernel code written to ${fanOutOutputPath}`);


  // --- Example 4: A graph with a fan-in structure ---
  console.log("\n--- Example 4: Fan-in graph ---");
  const fanInGraph = new CudaGraph("FanInGraph");
  const scaleNode4 = new CudaNode(scaleDeviceCode, "scale_kernel")
    .addInput('in', [1024], 'float32')
    .addOutput('out1', [1024], 'float32')
    .addParameter('scale_factor', scaleFactor);
  
  const scaleNode5 = new CudaNode(scaleDeviceCode, "scale_kernel")
    .addInput('in', [1024], 'float32')
    .addOutput('out2', [1024], 'float32')
    .addParameter('scale_factor', scaleFactor);

  const addNode4 = new CudaNode(addDeviceCode, "add_kernel")
    .addInput('a', [1024], 'float32')
    .addInput('b', [1024], 'float32')
    .addOutput('sum', [1024], 'float32');

  fanInGraph.addNode(scaleNode4).addNode(scaleNode5).addNode(addNode4);
  fanInGraph.connect(scaleNode4, 'out1', addNode4, 'a');
  fanInGraph.connect(scaleNode5, 'out2', addNode4, 'b');

  const { kernelCode: fanInKernelCode } = await compiler.compile(fanInGraph, new Map());
  const fanInOutputPath = path.join(process.cwd(), 'generated-fan-in-kernel.cu');
  fs.writeFileSync(fanInOutputPath, fanInKernelCode);
  console.log(`'fan-in' kernel code written to ${fanInOutputPath}`);


  // --- Example 5: Dynamic Shape Resolution ---
  console.log("\n--- Example 5: Dynamic Shape Resolution ---");
  const dynamicGraph = new CudaGraph("DynamicGraph");
  const dynamicNode = new CudaNode(addDeviceCode, "add_kernel")
    .addInput('a', [-1, 10], 'float32') // Batch size is dynamic
    .addInput('b', [-1, 10], 'float32')
    .addOutput('out', [-1, 10], 'float32')
    .setShapeResolver(inputs => {
      const aShape = inputs.get('a')!.shape;
      // Output shape is the same as input 'a'
      return new Map([['out', { shape: aShape }]]);
    });
  dynamicGraph.addNode(dynamicNode);

  // Set the concrete shape before compiling
  const dynamicInputShapes = new Map([
    ['a', [64, 10]],
    ['b', [64, 10]]
  ]);
  
  const { kernelCode: dynamicKernelCode } = await compiler.compile(dynamicGraph, dynamicInputShapes);
  const dynamicOutputPath = path.join(process.cwd(), 'generated-dynamic-kernel.cu');
  fs.writeFileSync(dynamicOutputPath, dynamicKernelCode);
  console.log(`'dynamic' kernel code written to ${dynamicOutputPath}`);


  // --- Example 6: More Complex Dynamic Shape Resolution ---
  console.log("\n--- Example 6: Complex Dynamic Shape Resolution ---");
  const complexDynamicGraph = new CudaGraph("ComplexDynamicGraph");
  const matmulDeviceCode = `
    __global__ void matmul_kernel(Tensor<float> out, const Tensor<float> a, const Tensor<float> b) {
        // Simplified matmul
    }
  `;
  const matmulNode = new CudaNode(matmulDeviceCode, "matmul_kernel")
    .addInput('a', [-1, 784], 'float32') // [batch, in_features]
    .addInput('b', [784, 10], 'float32') // [in_features, out_features]
    .addOutput('out', [-1, 10], 'float32') // [batch, out_features]
    .setShapeResolver(inputs => {
      const aShape = inputs.get('a')!.shape;
      const bShape = inputs.get('b')!.shape;
      const batchSize = aShape[0];
      const outFeatures = bShape[1];
      return new Map([['out', { shape: [batchSize, outFeatures] }]]);
    });
  complexDynamicGraph.addNode(matmulNode);

  const complexDynamicInputShapes = new Map([
    ['a', [128, 784]] // Provide concrete batch size
  ]);
  
  const { kernelCode: complexDynamicKernelCode } = await compiler.compile(complexDynamicGraph, complexDynamicInputShapes);
  const complexDynamicOutputPath = path.join(process.cwd(), 'generated-complex-dynamic-kernel.cu');
  fs.writeFileSync(complexDynamicOutputPath, complexDynamicKernelCode);
  console.log(`'complex-dynamic' kernel code written to ${complexDynamicOutputPath}`);
}

main().catch(console.error);
