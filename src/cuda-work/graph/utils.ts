// ============================================================================
// Graph Utilities: Helpers for Visualization, Debugging, and Building
// ============================================================================

import { CudaGraph, CudaNode } from "./core.js";

// ============================================================================
// GraphDebugger
// ============================================================================

export class GraphDebugger {
  static printGraphStructure(graph: CudaGraph): void {
    console.log(`\n=== Graph: ${graph.name} (${graph.id}) ===`);
    console.log(`Nodes: ${graph.nodes.size}`);
    console.log(`Connections: ${graph.connections.size}`);

    console.log("\nNodes:");
    for (const node of graph.nodes.values()) {
      console.log(`  ${node.name} (${node.id})`);
      console.log(`    Inputs: ${Array.from(node.inputs.keys()).join(", ")}`);
      console.log(`    Outputs: ${Array.from(node.outputs.keys()).join(", ")}`);
      console.log(
        `    Parameters: ${Array.from(node.parameters.keys()).join(", ")}`
      );
    }

    console.log("\nConnections:");
    for (const conn of graph.connections) {
      console.log(
        `  ${conn.fromNode.name}:${conn.fromPort} -> ${conn.toNode.name}:${conn.toPort}`
      );
    }

    console.log("\nExecution Order:");
    const executionOrder = graph.getExecutionOrder();
    executionOrder.forEach((node, i) => {
      console.log(`  ${i}: ${node.name}`);
    });
  }

  static generateDotGraph(graph: CudaGraph): string {
    let dot = `digraph "${graph.name}" {\n`;
    dot += `  rankdir=TB;\n`;
    dot += `  node [shape=box, style=rounded];\n\n`;

    for (const node of graph.nodes.values()) {
      const inputs = Array.from(node.inputs.keys()).join(",");
      const outputs = Array.from(node.outputs.keys()).join(",");
      const params = Array.from(node.parameters.keys()).join(",");

      let label = `${node.name}\\n`;
      if (inputs) label += `in: ${inputs}\\n`;
      if (outputs) label += `out: ${outputs}\\n`;
      if (params) label += `params: ${params}`;

      dot += `  "${node.id}" [label="${label}"];\n`;
    }

    dot += "\n";

    for (const conn of graph.connections) {
      dot += `  "${conn.fromNode.id}" -> "${conn.toNode.id}" [label="${conn.fromPort}→${conn.toPort}"];\n`;
    }

    dot += "}\n";
    return dot;
  }

  static validateGraph(graph: CudaGraph): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    const connectedNodes = new Set<string>();
    for (const conn of graph.connections) {
      connectedNodes.add(conn.fromNode.id);
      connectedNodes.add(conn.toNode.id);
    }

    for (const node of graph.nodes.values()) {
      if (!connectedNodes.has(node.id) && node.inputs.size > 0) {
        errors.push(
          `Node ${node.name} (${node.id}) has inputs but is not connected`
        );
      }
    }

    for (const node of graph.nodes.values()) {
      for (const [port, spec] of node.outputs) {
        if (spec.shape.includes(-1)) {
          errors.push(
            `Node ${
              node.name
            } output ${port} has unresolved shape: [${spec.shape.join(", ")}]`
          );
        }
      }
    }

    for (const conn of graph.connections) {
      const fromSpec = conn.fromNode.outputs.get(conn.fromPort);
      const toSpec = conn.toNode.inputs.get(conn.toPort);

      if (fromSpec && toSpec) {
        if (fromSpec.dtype !== toSpec.dtype) {
          errors.push(
            `Type mismatch: ${conn.fromNode.name}:${conn.fromPort} (${fromSpec.dtype}) -> ${conn.toNode.name}:${conn.toPort} (${toSpec.dtype})`
          );
        }

        if (fromSpec.shape.length !== toSpec.shape.length) {
          errors.push(
            `Shape rank mismatch: ${conn.fromNode.name}:${conn.fromPort} (rank ${fromSpec.shape.length}) -> ${conn.toNode.name}:${conn.toPort} (rank ${toSpec.shape.length})`
          );
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}

// ============================================================================
// MemoryPlanVisualizer
// ============================================================================

export class MemoryPlanVisualizer {
  static visualizeMemoryPlan(
    intermediateTensors: Map<string, any>,
    workspaceSize: number
  ): string {
    let visualization = "\n=== Memory Allocation Plan ===\n";
    visualization += `Total Workspace: ${workspaceSize} bytes (${(workspaceSize / 1024 / 1024).toFixed(2)} MB)\n\n`;
    
    const sortedTensors = Array.from(intermediateTensors.entries())
      .sort((a, b) => a[1].offset - b[1].offset);
    
    const scale = 80;
    const maxOffset = workspaceSize;
    
    visualization += "Memory Layout:\n";
    visualization += "0" + " ".repeat(scale - 10) + `${(maxOffset / 1024 / 1024).toFixed(1)}MB\n`;
    visualization += "|" + "-".repeat(scale - 2) + "|\n";
    
    for (const [key, tensor] of sortedTensors) {
      const startPos = Math.floor((tensor.offset / maxOffset) * scale);
      const endPos = Math.floor(((tensor.offset + tensor.size) / maxOffset) * scale);
      const width = Math.max(1, endPos - startPos);
      
      let line = " ".repeat(startPos) + "█".repeat(width);
      const label = ` ${key} (${(tensor.size / 1024).toFixed(1)}KB)`;
      
      visualization += line + label + "\n";
    }
    
    visualization += "|" + "-".repeat(scale - 2) + "|\n\n";
    
    visualization += "Tensor Details:\n";
    visualization += "Name".padEnd(40) + "Offset".padEnd(12) + "Size".padEnd(12) + "Lifetime\n";
    visualization += "-".repeat(76) + "\n";
    
    for (const [key, tensor] of sortedTensors) {
      const name = key.padEnd(40);
      const offset = tensor.offset.toString().padEnd(12);
      const size = `${(tensor.size / 1024).toFixed(1)}KB`.padEnd(12);
      const lifetime = `[${tensor.liveStart}-${tensor.liveEnd}]`;
      
      visualization += `${name}${offset}${size}${lifetime}\n`;
    }
    
    return visualization;
  }
}

// ============================================================================
// GraphBuilder
// ============================================================================

export class GraphBuilder {
  private graph: CudaGraph;
  
  constructor(name: string) {
    this.graph = new CudaGraph(name);
  }
  
  addNode(node: CudaNode): GraphBuilder {
    this.graph.addNode(node);
    return this;
  }
  
  connect(
    fromNode: CudaNode | string,
    fromPort: string,
    toNode: CudaNode | string,
    toPort: string
  ): GraphBuilder {
    const from = typeof fromNode === 'string' 
      ? this.findNodeByName(fromNode) 
      : fromNode;
    const to = typeof toNode === 'string' 
      ? this.findNodeByName(toNode) 
      : toNode;
      
    if (!from || !to) {
      throw new Error(`Cannot find nodes for connection`);
    }
    
    this.graph.connect(from, fromPort, to, toPort);
    return this;
  }
  
  private findNodeByName(name: string): CudaNode | null {
    for (const node of this.graph.nodes.values()) {
      if (node.name === name) {
        return node;
      }
    }
    return null;
  }
  
  build(): CudaGraph {
    return this.graph;
  }
}
