// ============================================================================
// Test Memory System Fix
// ============================================================================

import { MemoryManager } from "./index.js";

function testMemorySystemFix() {
  console.log("🧪 Testing Memory System Fix");
  console.log("=".repeat(50));

  // Mock graph structure to test the fix
  const mockGraph = {
    getGraphInputs: () =>
      new Map([
        [
          "input",
          {
            node: {
              id: "node1",
              inputs: new Map([["input", { shape: [32, 10], dtype: "int32" }]]),
            },
            port: "input",
          },
        ],
      ]),
    nodes: new Map([
      [
        "node1",
        {
          id: "node1",
          name: "embedding_forward",
          outputs: new Map([
            ["output", { shape: [32, 10, 384], dtype: "float32" }],
          ]),
        },
      ],
    ]),
    connections: [],
  };

  const mockExecutionOrder = [
    {
      id: "node1",
      name: "embedding_forward",
      outputs: new Map([
        ["output", { shape: [32, 10, 384], dtype: "float32" }],
      ]),
    },
  ];

  const config = {
    enableMemoryPooling: true,
    memoryAlignment: 256,
    verboseMemoryPlanning: true,
    enableInPlaceOptimization: true,
  };

  try {
    const memoryManager = new MemoryManager(config);
    const result = memoryManager.planMemory(mockGraph, mockExecutionOrder);

    console.log(`✅ Memory planning successful`);
    console.log(`   Workspace size: ${result.workspaceSize} bytes`);
    console.log(`   Tensor registry size: ${result.tensorRegistry.size}`);

    // Check if the tensor registry contains the expected keys
    const expectedKey = "node1:input"; // This was failing before
    if (result.tensorRegistry.has(expectedKey)) {
      console.log(`✅ Found expected tensor registry key: ${expectedKey}`);
    } else {
      console.log(`❌ Missing expected tensor registry key: ${expectedKey}`);
      console.log(
        `   Available keys:`,
        Array.from(result.tensorRegistry.keys())
      );
    }
  } catch (error) {
    console.error(`❌ Memory planning failed:`, error);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  testMemorySystemFix();
}

export { testMemorySystemFix };
