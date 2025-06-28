import { CustomCudaNode } from "./cuda-nodes.js";
import { CudaGraph, CudaGraphCompiler, CudaNode } from "./cuda-graph.js";
import { MockCudaRuntime } from "./cuda-abstractions.js";

async function testCustomCudaNode() {
  console.log("--- Running CustomCudaNode Test ---");

  const runtime = new MockCudaRuntime();

  const tsCode = `
    function customAdd(A: number, B: number, C: number): void {
      const i = 0; // Simplified for testing
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

testCustomCudaNode();
