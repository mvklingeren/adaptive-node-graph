// ============================================================================
// Memory Management System for CUDA Graph Compiler
// ============================================================================

export * from "./lifetime-analyzer.js";
export * from "./allocator.js";
export * from "./pool-manager.js";
export * from "./types.js";

import { MemoryLifetimeAnalyzer } from "./lifetime-analyzer.js";
import { MemoryAllocator } from "./allocator.js";
import { MemoryPoolManager } from "./pool-manager.js";
import { MemoryPlanningConfig } from "./types.js";

/**
 * Unified memory management facade that coordinates all memory planning operations
 */
export class MemoryManager {
  private lifetimeAnalyzer: MemoryLifetimeAnalyzer;
  private allocator: MemoryAllocator;
  private poolManager: MemoryPoolManager;

  constructor(private config: MemoryPlanningConfig) {
    this.lifetimeAnalyzer = new MemoryLifetimeAnalyzer(config);
    this.allocator = new MemoryAllocator(config);
    this.poolManager = new MemoryPoolManager(config);
  }

  /**
   * Main entry point for memory planning - replaces the monolithic planMemory method
   */
  planMemory(graph: any, executionOrder: any[]) {
    // Step 1: Analyze tensor lifetimes
    const lifetimes = this.lifetimeAnalyzer.computeLifetimes(
      graph,
      executionOrder
    );

    // Step 2: Create memory pools for better organization
    const pools = this.poolManager.createPools(lifetimes);

    // Step 3: Perform optimal memory allocation
    const allocation = this.allocator.allocateWithOptimalPacking(
      lifetimes,
      pools
    );

    // Step 4: Build complete tensor registry from intermediate registry and IO registry
    const tensorRegistry = new Map(allocation.registry);
    for (const [key, value] of lifetimes.ioTensorRegistry.entries()) {
      tensorRegistry.set(key, value);
    }

    return {
      intermediateTensors: allocation.tensors,
      workspaceSize: allocation.workspaceSize,
      tensorRegistry: tensorRegistry,
      memoryPools: pools,
    };
  }
}
