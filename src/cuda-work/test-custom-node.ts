import { CustomCudaNode } from "./cuda-nodes.js";
import { CudaGraph, CudaGraphCompiler, CudaNode } from "./cuda-graph.js";
import { MockCudaRuntime } from "./cuda-abstractions.js";

async function testCustomCudaNode() {
  console.log("--- Running CustomCudaNode Test ---");

  const runtime = new MockCudaRuntime();

  const tsCode = `
    function customAdd(A: number, B: number, C: number): void {
      let i: number = 0; // Simplified for testing
      C[i] = A[i] + B[i];
    }
  `;

  try {
    const customNode = new CustomCudaNode(tsCode);

    // Manually set the output for now, since the automatic detection is simplified
    customNode.outputs.clear();
    customNode.addOutput("C", [-1], "float32");

    const graph = new CudaGraph("CustomNodeTest");
    graph.addNode(customNode);

    // To make the graph valid, we need to connect the inputs.
    // For this test, we'll create dummy nodes to serve as inputs.
    const inputNodeA = new CudaNode("","").addOutput("A", [1], "float32");
    const inputNodeB = new CudaNode("","").addOutput("B", [1], "float32");

    graph.addNode(inputNodeA);
    graph.addNode(inputNodeB);
    graph.connect(inputNodeA, "A", customNode, "A");
    graph.connect(inputNodeB, "B", customNode, "B");

    const compiler = new CudaGraphCompiler(runtime);
    const { kernelCode } = await compiler.compile(graph);

    console.log("Compilation successful. Generated kernel code:");
    console.log(kernelCode);

    if (kernelCode.includes("customAdd") && kernelCode.includes("C(i) = A(i) + B(i);")) {
      console.log("Test PASSED: Kernel code contains the expected custom function.");
    } else {
      console.error("Test FAILED: Kernel code does not contain the expected custom function.");
    }
  } catch (error) {
    console.error("Test FAILED:", error);
  }
}

async function testIfStatement() {
  console.log("\n--- Running If Statement Test ---");
  const runtime = new MockCudaRuntime();
  const tsCode = `
    function customIf(A: number, B: number, C: number): void {
      let i: number = 0;
      if (A[i] > B[i]) {
        C[i] = A[i];
      } else {
        C[i] = B[i];
      }
    }
  `;

  try {
    const customNode = new CustomCudaNode(tsCode);
    customNode.outputs.clear();
    customNode.addOutput("C", [-1], "float32");

    const graph = new CudaGraph("IfTest");
    graph.addNode(customNode);

    const inputNodeA = new CudaNode("","").addOutput("A", [1], "float32");
    const inputNodeB = new CudaNode("","").addOutput("B", [1], "float32");
    graph.addNode(inputNodeA);
    graph.addNode(inputNodeB);
    graph.connect(inputNodeA, "A", customNode, "A");
    graph.connect(inputNodeB, "B", customNode, "B");

    const compiler = new CudaGraphCompiler(runtime);
    const { kernelCode } = await compiler.compile(graph);

    console.log("Generated kernel code for if statement test:");
    console.log(kernelCode);

    if (kernelCode.includes("if (A(i) > B(i))")) {
      console.log("Test PASSED: Kernel code contains the if statement.");
    } else {
      console.error("Test FAILED: Kernel code does not contain the if statement.");
    }
  } catch (error) {
    console.error("Test FAILED:", error);
  }
}

async function testForLoop() {
  console.log("\n--- Running For Loop Test ---");
  const runtime = new MockCudaRuntime();
  const tsCode = `
    function customFor(A: number, C: number): void {
      for (let i: number = 0; i < 10; i++) {
        C[i] = A[i] * 2.0;
      }
    }
  `;

  try {
    const customNode = new CustomCudaNode(tsCode);
    customNode.outputs.clear();
    customNode.addOutput("C", [-1], "float32");

    const graph = new CudaGraph("ForLoopTest");
    graph.addNode(customNode);

    const inputNodeA = new CudaNode("","").addOutput("A", [10], "float32");
    graph.addNode(inputNodeA);
    graph.connect(inputNodeA, "A", customNode, "A");

    const compiler = new CudaGraphCompiler(runtime);
    const { kernelCode } = await compiler.compile(graph);

    console.log("Generated kernel code for for loop test:");
    console.log(kernelCode);

    if (kernelCode.includes("for (float i = 0; i < 10; i++)")) {
      console.log("Test PASSED: Kernel code contains the for loop.");
    } else {
      console.error("Test FAILED: Kernel code does not contain the for loop.");
    }
  } catch (error) {
    console.error("Test FAILED:", error);
  }
}

async function testTypeInference() {
  console.log("\n--- Running Type Inference Test ---");
  const runtime = new MockCudaRuntime();
  const tsCode = `
    function customTypeInference(A: number, C: number): void {
      let x = 5;
      let y = 5.0;
      let z = true;
      C[0] = x + y;
    }
  `;

  try {
    const customNode = new CustomCudaNode(tsCode);
    customNode.outputs.clear();
    customNode.addOutput("C", [-1], "float32");

    const graph = new CudaGraph("TypeInferenceTest");
    graph.addNode(customNode);

    const inputNodeA = new CudaNode("","").addOutput("A", [1], "float32");
    graph.addNode(inputNodeA);
    graph.connect(inputNodeA, "A", customNode, "A");

    const compiler = new CudaGraphCompiler(runtime);
    const { kernelCode } = await compiler.compile(graph);

    console.log("Generated kernel code for type inference test:");
    console.log(kernelCode);

    if (
      kernelCode.includes("int x = 5;") &&
      kernelCode.includes("float y = 5.0;") &&
      kernelCode.includes("bool z = true;")
    ) {
      console.log("Test PASSED: Kernel code contains the correctly inferred types.");
    } else {
      console.error("Test FAILED: Kernel code does not contain the correctly inferred types.");
    }
  } catch (error) {
    console.error("Test FAILED:", error);
  }
}

async function testWhileLoop() {
  console.log("\n--- Running While Loop Test ---");
  const runtime = new MockCudaRuntime();
  const tsCode = `
    function customWhile(A: number, C: number): void {
      let i = 0;
      while (i < 10) {
        C[i] = A[i];
        i++;
      }
    }
  `;

  try {
    const customNode = new CustomCudaNode(tsCode);
    customNode.outputs.clear();
    customNode.addOutput("C", [-1], "float32");

    const graph = new CudaGraph("WhileLoopTest");
    graph.addNode(customNode);

    const inputNodeA = new CudaNode("","").addOutput("A", [10], "float32");
    graph.addNode(inputNodeA);
    graph.connect(inputNodeA, "A", customNode, "A");

    const compiler = new CudaGraphCompiler(runtime);
    const { kernelCode } = await compiler.compile(graph);

    console.log("Generated kernel code for while loop test:");
    console.log(kernelCode);

    if (kernelCode.includes("while (i < 10)")) {
      console.log("Test PASSED: Kernel code contains the while loop.");
    } else {
      console.error("Test FAILED: Kernel code does not contain the while loop.");
    }
  } catch (error) {
    console.error("Test FAILED:", error);
  }
}

async function testKernelTypeConfiguration() {
  console.log("\n--- Running Kernel Type Configuration Test ---");
  const runtime = new MockCudaRuntime();
  const tsCode = `
    // @cuda global
    function customGlobalKernel(A: number, C: number): void {
      C[0] = A[0];
    }
  `;

  try {
    const customNode = new CustomCudaNode(tsCode);
    customNode.outputs.clear();
    customNode.addOutput("C", [-1], "float32");

    const graph = new CudaGraph("KernelTypeTest");
    graph.addNode(customNode);

    const inputNodeA = new CudaNode("","").addOutput("A", [1], "float32");
    graph.addNode(inputNodeA);
    graph.connect(inputNodeA, "A", customNode, "A");

    const compiler = new CudaGraphCompiler(runtime);
    const { kernelCode } = await compiler.compile(graph);

    console.log("Generated kernel code for kernel type configuration test:");
    console.log(kernelCode);

    if (kernelCode.includes("__global__ void customGlobalKernel")) {
      console.log("Test PASSED: Kernel code contains the __global__ kernel.");
    } else {
      console.error("Test FAILED: Kernel code does not contain the __global__ kernel.");
    }
  } catch (error) {
    console.error("Test FAILED:", error);
  }
}

async function runTests() {
  await testCustomCudaNode();
  await testIfStatement();
  await testForLoop();
  await testTypeInference();
  await testWhileLoop();
  await testKernelTypeConfiguration();
}

runTests();
