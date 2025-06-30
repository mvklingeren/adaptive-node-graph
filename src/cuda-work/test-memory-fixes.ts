import { CudaGraphCompiler } from "./graph/compiler.js";
import { CudaGraph, CudaNode } from "./graph/core.js";
import { CudaRuntime, CudaKernel } from "./cuda-abstractions.js";

// Mock runtime for testing
class MockCudaRuntime implements CudaRuntime {
  malloc(size: number): Promise<any> { return Promise.resolve({}); }
  memcpyHostToDevice(dst: any, src: any, size: number): Promise<void> { return Promise.resolve(); }
  memcpyDeviceToHost(dst: any, src: any, size: number): Promise<void> { return Promise.resolve(); }
  memcpyDeviceToDevice(dst: any, src: any, size: number): Promise<void> { return Promise.resolve(); }
  free(ptr: any): Promise<void> { return Promise.resolve(); }
  synchronize(): Promise<void> { return Promise.resolve(); }
  getDeviceProperties(): Promise<any> { return Promise.resolve({}); }
  setDevice(device: number): Promise<void> { return Promise.resolve(); }
  getDevice(): Promise<number> { return Promise.resolve(0); }
  memset(ptr: any, value: number, size: number): Promise<void> { return Promise.resolve(); }
  getDeviceInfo(): Promise<any> { return Promise.resolve({}); }
  createStream(): Promise<any> { return Promise.resolve({}); }
  destroyStream(stream: any): Promise<void> { return Promise.resolve(); }
  setCurrentStream(stream: any): Promise<void> { return Promise.resolve(); }

  async compile(code: string, filename: string): Promise<CudaKernel> {
    console.log(`\n=== Generated CUDA Code Analysis (${filename}) ===`);
    
    // Extract and analyze the WorkspaceAllocator class
    const allocatorMatch = code.match(/class WorkspaceAllocator \{[\s\S]*?\};/);
    if (allocatorMatch) {
      console.log("✅ Enhanced WorkspaceAllocator found with:");
      
      // Check for allocate_at_offset method
      if (code.includes("allocate_at_offset")) {
        console.log("  ✅ allocate_at_offset method implemented");
      } else {
        console.log("  ❌ allocate_at_offset method missing");
      }
      
      // Check for proper error handling
      if (code.includes("Offset allocation out of bounds")) {
        console.log("  ✅ Enhanced error handling for offset allocation");
      } else {
        console.log("  ❌ Missing enhanced error handling");
      }
    } else {
      console.log("❌ WorkspaceAllocator class not found");
    }
    
    // Check for proper memory reuse in tensor instantiation
    const allocateAtOffsetUsage = (code.match(/allocate_at_offset/g) || []).length;
    console.log(`\n=== Memory Reuse Analysis ===`);
    console.log(`allocate_at_offset calls: ${allocateAtOffsetUsage}`);
    
    if (allocateAtOffsetUsage > 0) {
      console.log("✅ Memory reuse is being utilized");
    } else {
      console.log("⚠️  No memory reuse detected (may be expected for simple graphs)");
    }
    
    // Check for offset comments
    const offsetComments = (code.match(/\/\/ Pool: .*, Offset: \d+, Size: \d+ bytes/g) || []).length;
    console.log(`Offset-based allocation comments: ${offsetComments}`);
    
    if (offsetComments > 0) {
      console.log("✅ Proper offset tracking in generated code");
    }
    
    // Extract workspace size
    const workspaceSizeMatch = code.match(/workspace_size < (\d+)/);
    if (workspaceSizeMatch) {
      const workspaceSize = parseInt(workspaceSizeMatch[1]);
      console.log(`\n=== Workspace Size ===`);
      console.log(`Required workspace: ${workspaceSize} bytes (${(workspaceSize / 1024 / 1024).toFixed(2)} MB)`);
    }
    
    return { execute: () => Promise.resolve() } as any;
  }
}

// Simple test node for memory allocation testing
class TestNode extends CudaNode {
  constructor(id: string, outputShape: number[], hasInput: boolean = false) {
    super(id, `test_${id}`);
    if (hasInput) {
      this.inputs.set("input", { shape: outputShape, dtype: "float32" });
    }
    this.outputs.set("output", { shape: outputShape, dtype: "float32" });
    // Use type assertion to set the readonly deviceCode property
    (this as any).deviceCode = `
      /**
       * @cuda global
       */
      __global__ void test_${id}_forward(Tensor<float> output) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < output.total_elements()) {
          output.data[idx] = 1.0f;
        }
      }
    `;
  }

  resolveShapes(): void {
    // Shapes are already set in constructor
  }

  getKernelCall(
    outputTensors: Map<string, string>,
    inputTensors: Map<string, string>,
    getParamTensor: (nodeId: string, paramName: string) => string
  ): string {
    const outputTensor = outputTensors.get("output");
    return `${outputTensor}`;
  }
}

async function testMemoryManagement() {
  console.log("🧪 Testing Enhanced Memory Management Fixes");
  console.log("=" .repeat(50));
  
  const mockRuntime = new MockCudaRuntime();
  const compiler = new CudaGraphCompiler(mockRuntime, {
    verboseMemoryPlanning: true,
    enableMemoryPooling: true,
    memoryAlignment: 256
  });
  
  // Create a test graph with multiple nodes that should benefit from memory reuse
  const graph = new CudaGraph("memory-test");
  
  // Add nodes with different lifetimes to test memory reuse
  const node1 = new TestNode("node1", [32, 128, 384], false); // Large tensor, no input
  const node2 = new TestNode("node2", [32, 128, 384], true);  // Same size, has input
  const node3 = new TestNode("node3", [32, 64, 192], false);  // Smaller tensor, no input
  const node4 = new TestNode("node4", [32, 128, 384], true);  // Can reuse node1's memory, has input
  
  graph.addNode(node1);
  graph.addNode(node2);
  graph.addNode(node3);
  graph.addNode(node4);
  
  // Create connections to establish lifetimes
  // node1 -> node2 -> node4 (sequential)
  // node3 runs in parallel
  graph.connect(node1, "output", node2, "input");
  graph.connect(node2, "output", node4, "input");
  
  const inputShapes = new Map([
    ["input", [32, 128]]
  ]);
  
  try {
    const result = await compiler.compile(graph, inputShapes);
    
    console.log(`\n=== Compilation Results ===`);
    console.log(`✅ Compilation successful`);
    console.log(`Workspace size: ${result.workspaceSize} bytes (${(result.workspaceSize / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`Parameters: ${result.parameters.length}`);
    
    // Calculate theoretical memory usage without reuse
    const tensorSize = 32 * 128 * 384 * 4; // float32
    const smallTensorSize = 32 * 64 * 192 * 4;
    const totalWithoutReuse = (tensorSize * 3) + smallTensorSize; // 4 tensors, 3 large + 1 small
    
    console.log(`\n=== Memory Efficiency Analysis ===`);
    console.log(`Theoretical memory without reuse: ${totalWithoutReuse} bytes (${(totalWithoutReuse / 1024 / 1024).toFixed(2)} MB)`);
    
    if (result.workspaceSize < totalWithoutReuse) {
      const savings = totalWithoutReuse - result.workspaceSize;
      const efficiency = ((savings / totalWithoutReuse) * 100).toFixed(1);
      console.log(`✅ Memory savings: ${savings} bytes (${(savings / 1024 / 1024).toFixed(2)} MB)`);
      console.log(`✅ Memory efficiency: ${efficiency}% reduction`);
    } else {
      console.log(`⚠️  No memory savings detected (may be due to alignment overhead)`);
    }
    
  } catch (error) {
    console.error(`❌ Compilation failed:`, error);
  }
}

// Run the test
testMemoryManagement().catch(console.error);
