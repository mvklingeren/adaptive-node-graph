// ============================================================================
// Memory Lifetime Analyzer
// ============================================================================

import { TensorLifetimes, TensorInfo, MemoryPlanningConfig } from "./types.js";

export class MemoryLifetimeAnalyzer {
  constructor(private config: MemoryPlanningConfig) {}

  /**
   * Computes enhanced tensor lifetimes with improved algorithm
   */
  computeLifetimes(
    graph: any,
    executionOrder: any[]
  ): TensorLifetimes & { ioTensorRegistry: Map<string, any> } {
    const nodeIndexMap = this.buildNodeIndexMap(executionOrder);
    const { intermediateTensors, ioTensorRegistry } =
      this.identifyIntermediateTensors(graph, executionOrder, nodeIndexMap);

    this.initializeLifetimes(intermediateTensors, nodeIndexMap);
    this.propagateConsumerLifetimes(graph, intermediateTensors, nodeIndexMap);
    this.handleGraphOutputs(graph, intermediateTensors, executionOrder);

    if (this.config.verboseMemoryPlanning) {
      this.logLifetimeStatistics(intermediateTensors);
    }

    return {
      tensors: intermediateTensors,
      totalTensors: intermediateTensors.size,
      maxLifetime: this.calculateMaxLifetime(intermediateTensors),
      ioTensorRegistry,
    };
  }

  private buildNodeIndexMap(executionOrder: any[]): Map<string, number> {
    const nodeIndexMap = new Map<string, number>();
    executionOrder.forEach((node, i) => {
      nodeIndexMap.set(node.id, i);
    });
    return nodeIndexMap;
  }

  private identifyIntermediateTensors(
    graph: any,
    executionOrder: any[],
    nodeIndexMap: Map<string, number>
  ): {
    intermediateTensors: Map<string, TensorInfo>;
    ioTensorRegistry: Map<string, any>;
  } {
    const intermediateTensors = new Map<string, TensorInfo>();
    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);

    // Create tensor registry for inputs and outputs
    const ioTensorRegistry = new Map();
    for (const [inputPort, { node, port }] of graphInputs.entries()) {
      ioTensorRegistry.set(`${node.id}:${port}`, {
        varName: inputPort,
        spec: node.inputs.get(port)!,
      });
    }
    for (const [outputPort, { node, port }] of graphOutputs.entries()) {
      ioTensorRegistry.set(`${node.id}:${outputPort}`, {
        varName: outputPort,
        spec: node.outputs.get(port)!,
      });
    }

    // Identify intermediate tensors
    executionOrder.forEach((node, i) => {
      for (const [outputPort, spec] of node.outputs) {
        const key = `${node.id}:${outputPort}`;
        if (spec.shape.includes(-1)) {
          throw new Error(
            `Unresolved dynamic shape ${JSON.stringify(
              spec.shape
            )} for output '${outputPort}' of node '${node.name}'`
          );
        }
        if (!ioTensorRegistry.has(key)) {
          const size = spec.shape.reduce((a, b) => a * b, 1) * 4; // float32
          intermediateTensors.set(key, {
            spec,
            size,
            liveStart: i,
            liveEnd: i,
            offset: -1,
          });
        }
      }
    });

    return { intermediateTensors, ioTensorRegistry };
  }

  private initializeLifetimes(
    intermediateTensors: Map<string, TensorInfo>,
    nodeIndexMap: Map<string, number>
  ): void {
    for (const [tensorKey, tensorInfo] of intermediateTensors.entries()) {
      const [nodeId] = tensorKey.split(":");
      const producerIndex = nodeIndexMap.get(nodeId);
      if (producerIndex !== undefined) {
        tensorInfo.liveStart = producerIndex;
        tensorInfo.liveEnd = producerIndex;
      }
    }
  }

  private propagateConsumerLifetimes(
    graph: any,
    intermediateTensors: Map<string, TensorInfo>,
    nodeIndexMap: Map<string, number>
  ): void {
    // Build tensor consumer map
    const tensorConsumers = new Map<string, Set<string>>();
    for (const conn of graph.connections) {
      const sourceKey = `${conn.fromNode.id}:${conn.fromPort}`;
      const consumerId = conn.toNode.id;

      if (!tensorConsumers.has(sourceKey)) {
        tensorConsumers.set(sourceKey, new Set());
      }
      tensorConsumers.get(sourceKey)!.add(consumerId);
    }

    // Update liveEnd based on last consumer
    for (const [tensorKey, consumerIds] of tensorConsumers.entries()) {
      const tensorInfo = intermediateTensors.get(tensorKey);
      if (!tensorInfo) continue;

      let maxConsumerIndex = tensorInfo.liveStart;
      for (const consumerId of consumerIds) {
        const consumerIndex = nodeIndexMap.get(consumerId);
        if (consumerIndex !== undefined) {
          maxConsumerIndex = Math.max(maxConsumerIndex, consumerIndex);
        }
      }
      tensorInfo.liveEnd = maxConsumerIndex;
    }
  }

  private handleGraphOutputs(
    graph: any,
    intermediateTensors: Map<string, TensorInfo>,
    executionOrder: any[]
  ): void {
    const graphOutputs = this.getGraphOutputs(graph);
    for (const [outputName, outputInfo] of graphOutputs) {
      const outputKey = `${outputInfo.node.id}:${outputInfo.port}`;
      const tensorInfo = intermediateTensors.get(outputKey);
      if (tensorInfo) {
        tensorInfo.liveEnd = executionOrder.length - 1;
      }
    }
  }

  private getGraphOutputs(
    graph: any
  ): Map<string, { node: any; port: string }> {
    const graphOutputs = new Map<string, { node: any; port: string }>();
    const allNodeOutputs = new Set<string>();

    for (const conn of graph.connections) {
      allNodeOutputs.add(`${conn.fromNode.id}:${conn.fromPort}`);
    }

    for (const node of graph.nodes.values()) {
      for (const [outputPort] of node.outputs) {
        if (!allNodeOutputs.has(`${node.id}:${outputPort}`)) {
          graphOutputs.set(outputPort, { node, port: outputPort });
        }
      }
    }

    return graphOutputs;
  }

  private calculateMaxLifetime(
    intermediateTensors: Map<string, TensorInfo>
  ): number {
    let maxLifetime = 0;
    for (const tensor of intermediateTensors.values()) {
      maxLifetime = Math.max(maxLifetime, tensor.liveEnd - tensor.liveStart);
    }
    return maxLifetime;
  }

  private logLifetimeStatistics(
    intermediateTensors: Map<string, TensorInfo>
  ): void {
    console.log("\n[Memory Planning] Tensor lifetime analysis:");

    const lifetimeHistogram = new Map<number, number>();
    let totalLifetime = 0;

    for (const [key, info] of intermediateTensors.entries()) {
      const lifetime = info.liveEnd - info.liveStart;
      totalLifetime += lifetime;
      lifetimeHistogram.set(
        lifetime,
        (lifetimeHistogram.get(lifetime) || 0) + 1
      );

      console.log(
        `  ${key}: live [${info.liveStart}-${
          info.liveEnd
        }] (${lifetime} steps, ${(info.size / 1024).toFixed(1)}KB)`
      );
    }

    const avgLifetime = totalLifetime / intermediateTensors.size;
    console.log(
      `\n[Memory Planning] Average tensor lifetime: ${avgLifetime.toFixed(
        1
      )} steps`
    );
    console.log(
      `[Memory Planning] Lifetime distribution:`,
      Object.fromEntries(lifetimeHistogram)
    );
  }
}
