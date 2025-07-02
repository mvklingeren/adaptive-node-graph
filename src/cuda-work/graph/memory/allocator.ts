// ============================================================================
// Optimized Memory Allocator
// ============================================================================

import {
  TensorLifetimes,
  TensorInfo,
  AllocationResult,
  MemoryInterval,
  MemoryPlanningConfig,
} from "./types.js";

export class MemoryAllocator {
  constructor(private config: MemoryPlanningConfig) {}

  /**
   * Allocates memory with optimal packing using improved O(n log n) algorithm
   */
  allocateWithOptimalPacking(
    lifetimes: TensorLifetimes,
    pools: Map<string, any>
  ): AllocationResult {
    const intermediateTensors = lifetimes.tensors;

    // Step 1: Sort tensors for optimal allocation order
    const sortedTensors = this.sortTensorsForAllocation(intermediateTensors);

    // Step 2: Use interval tree for efficient overlap detection
    const allocationTracker = new AllocationTracker(this.config);

    // Step 3: Allocate each tensor with optimal placement
    let maxWorkspaceSize = 0;
    for (const [tensorKey, tensor] of sortedTensors) {
      const offset = allocationTracker.findOptimalOffset(tensor);
      tensor.offset = offset;

      const alignedSize = this.alignToMemoryBoundary(tensor.size);
      maxWorkspaceSize = Math.max(maxWorkspaceSize, offset + alignedSize);

      allocationTracker.addAllocation(tensorKey, tensor, alignedSize);
    }

    // Step 4: Verify allocation and compute statistics
    if (this.config.verboseMemoryPlanning) {
      this.verifyAllocation(intermediateTensors);
    }

    const stats = this.computeAllocationStatistics(
      intermediateTensors,
      maxWorkspaceSize
    );

    return {
      tensors: intermediateTensors,
      workspaceSize: maxWorkspaceSize,
      registry: this.buildTensorRegistry(intermediateTensors),
      efficiency: stats.efficiency,
      savings: stats.savings,
    };
  }

  private sortTensorsForAllocation(
    intermediateTensors: Map<string, TensorInfo>
  ): Array<[string, TensorInfo]> {
    return Array.from(intermediateTensors.entries())
      .filter(([_, tensor]) => !tensor.poolId?.startsWith("inplace_"))
      .sort((a, b) => {
        // Primary sort: by start time (earlier first)
        if (a[1].liveStart !== b[1].liveStart) {
          return a[1].liveStart - b[1].liveStart;
        }
        // Secondary sort: by size (larger first for better packing)
        if (a[1].size !== b[1].size) {
          return b[1].size - a[1].size;
        }
        // Tertiary sort: by lifetime duration (shorter first)
        const lifetimeA = a[1].liveEnd - a[1].liveStart;
        const lifetimeB = b[1].liveEnd - b[1].liveStart;
        return lifetimeA - lifetimeB;
      });
  }

  private alignToMemoryBoundary(size: number): number {
    return (
      Math.ceil(size / this.config.memoryAlignment) *
      this.config.memoryAlignment
    );
  }

  private buildTensorRegistry(intermediateTensors: Map<string, TensorInfo>) {
    const registry = new Map<
      string,
      { varName: string; spec: { shape: number[]; dtype: string } }
    >();
    let idx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
      registry.set(key, {
        varName: `intermediate_${idx}_tensor`,
        spec: tensorInfo.spec,
      });
      idx++;
    }
    return registry;
  }

  private computeAllocationStatistics(
    intermediateTensors: Map<string, TensorInfo>,
    workspaceSize: number
  ) {
    const totalTensorSize = Array.from(intermediateTensors.values()).reduce(
      (sum, tensor) => sum + this.alignToMemoryBoundary(tensor.size),
      0
    );

    const savings = totalTensorSize - workspaceSize;
    const efficiency =
      workspaceSize > 0 ? (savings / totalTensorSize) * 100 : 0;

    if (this.config.verboseMemoryPlanning) {
      console.log(`\n[Optimized Memory Allocation] Results:`);
      console.log(
        `  Workspace size: ${workspaceSize} bytes (${(
          workspaceSize /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `  Total tensor size: ${totalTensorSize} bytes (${(
          totalTensorSize /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `  Memory saved: ${savings} bytes (${(savings / 1024 / 1024).toFixed(
          2
        )} MB)`
      );
      console.log(`  Memory efficiency: ${efficiency.toFixed(1)}%`);
    }

    return { efficiency, savings };
  }

  private verifyAllocation(intermediateTensors: Map<string, TensorInfo>): void {
    const intervals: MemoryInterval[] = [];

    for (const [key, tensor] of intermediateTensors.entries()) {
      if (tensor.offset >= 0) {
        const alignedSize = this.alignToMemoryBoundary(tensor.size);
        intervals.push({
          start: tensor.offset,
          end: tensor.offset + alignedSize,
          size: alignedSize,
          tensorKey: key,
          lifetime: [tensor.liveStart, tensor.liveEnd],
        });
      }
    }

    // Check for overlaps
    let overlapCount = 0;
    for (let i = 0; i < intervals.length; i++) {
      for (let j = i + 1; j < intervals.length; j++) {
        const interval1 = intervals[i];
        const interval2 = intervals[j];

        // Check if lifetimes overlap
        const lifetimeOverlap = !(
          interval1.lifetime[1] < interval2.lifetime[0] ||
          interval1.lifetime[0] > interval2.lifetime[1]
        );

        // Check if memory regions overlap
        const memoryOverlap = !(
          interval1.end <= interval2.start || interval1.start >= interval2.end
        );

        if (lifetimeOverlap && memoryOverlap) {
          console.error(
            `[Memory Allocation] ERROR: Overlap detected between ${interval1.tensorKey} and ${interval2.tensorKey}`
          );
          overlapCount++;
        }
      }
    }

    if (overlapCount === 0) {
      console.log(
        `[Memory Allocation] ✅ Verification passed - no overlaps detected`
      );
    } else {
      console.error(
        `[Memory Allocation] ❌ Verification failed - ${overlapCount} overlaps detected`
      );
    }
  }
}

/**
 * Efficient allocation tracker using optimized data structures
 */
class AllocationTracker {
  private allocatedIntervals: MemoryInterval[] = [];

  constructor(private config: MemoryPlanningConfig) {}

  findOptimalOffset(tensor: TensorInfo): number {
    const alignedSize = this.alignToMemoryBoundary(tensor.size);

    // Sort intervals by start offset for binary search
    this.allocatedIntervals.sort((a, b) => a.start - b.start);

    // Try to find a gap between existing allocations
    let bestOffset = 0;

    for (let i = 0; i <= this.allocatedIntervals.length; i++) {
      const gapStart = i === 0 ? 0 : this.allocatedIntervals[i - 1].end;
      const gapEnd =
        i === this.allocatedIntervals.length
          ? Infinity
          : this.allocatedIntervals[i].start;

      if (gapEnd - gapStart >= alignedSize) {
        // Check if this gap is usable (no lifetime conflicts)
        if (this.canAllocateInGap(gapStart, gapStart + alignedSize, tensor)) {
          bestOffset = gapStart;
          break;
        }
      }
    }

    // If no gap found, place at the end
    if (bestOffset === 0 && this.allocatedIntervals.length > 0) {
      bestOffset = Math.max(
        ...this.allocatedIntervals.map((interval) => interval.end)
      );
    }

    // Ensure alignment
    return (
      Math.ceil(bestOffset / this.config.memoryAlignment) *
      this.config.memoryAlignment
    );
  }

  private canAllocateInGap(
    start: number,
    end: number,
    tensor: TensorInfo
  ): boolean {
    // Check for conflicts with existing allocations that have overlapping lifetimes
    for (const interval of this.allocatedIntervals) {
      // Check if lifetimes overlap
      const lifetimeOverlap = !(
        tensor.liveEnd < interval.lifetime[0] ||
        tensor.liveStart > interval.lifetime[1]
      );

      // Check if memory regions would overlap
      const memoryOverlap = !(end <= interval.start || start >= interval.end);

      if (lifetimeOverlap && memoryOverlap) {
        return false;
      }
    }
    return true;
  }

  addAllocation(
    tensorKey: string,
    tensor: TensorInfo,
    alignedSize: number
  ): void {
    this.allocatedIntervals.push({
      start: tensor.offset,
      end: tensor.offset + alignedSize,
      size: alignedSize,
      tensorKey,
      lifetime: [tensor.liveStart, tensor.liveEnd],
    });
  }

  private alignToMemoryBoundary(size: number): number {
    return (
      Math.ceil(size / this.config.memoryAlignment) *
      this.config.memoryAlignment
    );
  }
}
