// ============================================================================
// Memory Pool Manager
// ============================================================================

import { TensorLifetimes, MemoryPool, MemoryPlanningConfig } from "./types.js";

export class MemoryPoolManager {
  constructor(private config: MemoryPlanningConfig) {}

  /**
   * Creates memory pools for better organization and potential optimizations
   */
  createPools(lifetimes: TensorLifetimes): Map<string, MemoryPool> {
    const pools = new Map<string, MemoryPool>();

    if (!this.config.enableMemoryPooling) {
      return pools;
    }

    // Categorize tensors by size
    const categorizedTensors = this.categorizeTensorsBySize(lifetimes.tensors);

    // Create pools for each size category
    for (const [category, tensorList] of categorizedTensors.entries()) {
      if (tensorList.length > 1) {
        // Only create pools if there are multiple tensors
        const pool = this.createPool(category, tensorList);
        pools.set(pool.id, pool);

        // Assign pool IDs to tensors
        for (const tensorKey of tensorList) {
          const tensor = lifetimes.tensors.get(tensorKey);
          if (tensor) {
            tensor.poolId = pool.id;
          }
        }
      }
    }

    if (this.config.verboseMemoryPlanning) {
      this.logPoolStatistics(pools);
    }

    return pools;
  }

  private categorizeTensorsBySize(
    tensors: Map<string, any>
  ): Map<string, string[]> {
    const categories = new Map<string, string[]>();

    for (const [tensorKey, tensor] of tensors.entries()) {
      const category = this.categorizeTensorSize(tensor.size);

      if (!categories.has(category)) {
        categories.set(category, []);
      }
      categories.get(category)!.push(tensorKey);
    }

    return categories;
  }

  private categorizeTensorSize(size: number): string {
    const KB = 1024;
    const MB = 1024 * 1024;

    if (size < 16 * KB) return "small"; // < 16KB
    if (size < 256 * KB) return "medium"; // 16KB - 256KB
    if (size < 4 * MB) return "large"; // 256KB - 4MB
    return "xlarge"; // > 4MB
  }

  private createPool(category: string, tensorList: string[]): MemoryPool {
    // Calculate total size for the pool
    const totalSize = tensorList.reduce((sum, tensorKey) => {
      // For now, we'll calculate this based on tensor count and category
      // In a real implementation, you'd use actual tensor sizes
      return sum + this.getEstimatedTensorSize(category);
    }, 0);

    return {
      id: `pool_${category}_${Date.now()}`,
      size: totalSize,
      tensors: [...tensorList],
      offset: -1, // Will be set during allocation
      category: category as "small" | "medium" | "large" | "xlarge",
    };
  }

  private getEstimatedTensorSize(category: string): number {
    const KB = 1024;
    const MB = 1024 * 1024;

    switch (category) {
      case "small":
        return 8 * KB;
      case "medium":
        return 128 * KB;
      case "large":
        return 1 * MB;
      case "xlarge":
        return 8 * MB;
      default:
        return 128 * KB;
    }
  }

  private logPoolStatistics(pools: Map<string, MemoryPool>): void {
    if (pools.size === 0) {
      console.log("[Memory Pools] No pools created");
      return;
    }

    console.log(`\n[Memory Pools] Created ${pools.size} memory pools:`);

    for (const [poolId, pool] of pools.entries()) {
      console.log(`  Pool ${poolId}:`);
      console.log(`    Category: ${pool.category}`);
      console.log(`    Tensors: ${pool.tensors.length}`);
      console.log(`    Estimated size: ${(pool.size / 1024).toFixed(1)}KB`);
    }

    // Summary statistics
    const poolsByCategory = new Map<string, number>();
    let totalPooledTensors = 0;

    for (const pool of pools.values()) {
      poolsByCategory.set(
        pool.category,
        (poolsByCategory.get(pool.category) || 0) + 1
      );
      totalPooledTensors += pool.tensors.length;
    }

    console.log(`\n[Memory Pools] Summary:`);
    console.log(`  Total pooled tensors: ${totalPooledTensors}`);
    console.log(`  Pools by category:`, Object.fromEntries(poolsByCategory));
  }
}
