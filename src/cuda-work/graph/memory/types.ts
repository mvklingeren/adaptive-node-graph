// ============================================================================
// Memory Management Types
// ============================================================================

export interface TensorInfo {
  spec: { shape: number[]; dtype: string };
  size: number;
  liveStart: number;
  liveEnd: number;
  offset: number;
  poolId?: string;
}

export interface TensorLifetimes {
  tensors: Map<string, TensorInfo>;
  totalTensors: number;
  maxLifetime: number;
}

export interface MemoryPool {
  id: string;
  size: number;
  tensors: string[];
  offset: number;
  category: "small" | "medium" | "large" | "xlarge";
}

export interface AllocationResult {
  tensors: Map<string, TensorInfo>;
  workspaceSize: number;
  registry: Map<
    string,
    { varName: string; spec: { shape: number[]; dtype: string } }
  >;
  efficiency: number;
  savings: number;
}

export interface MemoryPlanningConfig {
  enableMemoryPooling: boolean;
  memoryAlignment: number;
  verboseMemoryPlanning: boolean;
  enableInPlaceOptimization: boolean;
  maxPoolSize?: number;
}

export interface MemoryInterval {
  start: number;
  end: number;
  size: number;
  tensorKey: string;
  lifetime: [number, number];
}

export interface InPlaceCandidate {
  tensorKey: string;
  sourceKey: string;
  canReuse: boolean;
  reason?: string;
}
