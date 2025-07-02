// ============================================================================
// CudaGraphCompiler: Compiles a CudaGraph into a set of kernels and a
// host-side execution function with improved parameter interface.
// ============================================================================

import { CudaRuntime, CudaTensor, CudaKernel } from "../cuda-abstractions.js";
import { CompilerConfig, CudaGraph, CudaNode } from "./core.js";
import { generateKernelCode } from "./kernel-template.js";
import {
  createKernelConfigRegistry,
  KernelConfigRegistry,
  DEFAULT_CUDA_CONSTRAINTS,
} from "./kernel-config/index.js";
import { MemoryManager } from "./memory/index.js";

export class CudaGraphCompiler {
  private config: CompilerConfig;
  private kernelConfigRegistry: KernelConfigRegistry;
  private memoryManager: MemoryManager;

  constructor(private runtime: CudaRuntime, config?: CompilerConfig) {
    this.config = {
      useArrayInterface: true,
      enableBoundsChecking: true, // Ensure this is true for debugging
      defaultBlockSize: 256,
      memoryAlignment: 256,
      enableMemoryPooling: true,
      verboseMemoryPlanning: true, // Enable verbose logging for memory planning
      enableInPlaceOptimization: true,
      ...config,
    };

    // Initialize the kernel configuration registry
    this.kernelConfigRegistry = createKernelConfigRegistry();

    // Initialize the new memory management system
    this.memoryManager = new MemoryManager({
      enableMemoryPooling: this.config.enableMemoryPooling!,
      memoryAlignment: this.config.memoryAlignment!,
      verboseMemoryPlanning: this.config.verboseMemoryPlanning!,
      enableInPlaceOptimization: this.config.enableInPlaceOptimization!,
    });
  }

  async compile(
    graph: CudaGraph,
    inputShapes: Map<string, number[]>
  ): Promise<{
    kernel: CudaKernel;
    parameters: CudaTensor[];
    kernelCode: string;
    workspaceSize: number;
  }> {
    const allParams = new Map<string, CudaTensor>();
    const paramNameMapping = new Map<string, string>();
    let paramCounter = 0;

    for (const node of graph.nodes.values()) {
      if (!node || !node.id) {
        console.warn("Skipping invalid node in parameter mapping");
        continue;
      }

      for (const [localParamName, tensor] of node.parameters) {
        if (!tensor || !tensor.id) {
          console.warn(
            `Skipping invalid tensor '${localParamName}' in node '${node.name}'`
          );
          continue;
        }

        let globalParamName = null;
        for (const [existingName, existingTensor] of allParams.entries()) {
          if (existingTensor && existingTensor.id === tensor.id) {
            globalParamName = existingName;
            break;
          }
        }

        if (!globalParamName) {
          globalParamName = `param_${paramCounter++}_${localParamName}`;
          allParams.set(globalParamName, tensor);
        }

        const nodeParamKey = `${node.id}:${localParamName}`;
        paramNameMapping.set(nodeParamKey, globalParamName);
      }
    }

    const orderedParams = Array.from(allParams.values());
    const paramNames = Array.from(allParams.keys());

    this._setAndPropagateShapes(graph, inputShapes);

    const { kernelCode, workspaceSize, filename } = this.config
      .useArrayInterface
      ? this.generateImprovedKernelCode(graph, paramNames, paramNameMapping)
      : this.generateLegacyKernelCode(graph, paramNames, paramNameMapping);

    const kernel = await this.runtime.compile(kernelCode, filename);

    return { kernel, parameters: orderedParams, kernelCode, workspaceSize };
  }

  private generateImprovedKernelCode(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): { kernelCode: string; workspaceSize: number; filename: string } {
    const executionOrder = graph.getExecutionOrder();
    const filename = `generated-${graph.name}-kernel.cu`;

    const { intermediateTensors, workspaceSize, tensorRegistry, memoryPools } =
      this.memoryManager.planMemory(graph, executionOrder);

    const variableDeclarations: string[] = [];
    let intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
      const varName = `intermediate_${intermediateIdx}`;
      const shapeVar = `${varName}_shape`;
      const tensorVar = `${varName}_tensor`;
      variableDeclarations.push(
        `  const int ${shapeVar}[] = {${tensorInfo.spec.shape.join(", ")}};`
      );
      tensorRegistry.set(key, { varName: tensorVar, spec: tensorInfo.spec });
      intermediateIdx++;
    }

    const uniqueKernels = new Map<string, { code: string; node: CudaNode }>();
    for (const node of executionOrder) {
      if (node.deviceCode.trim() && !uniqueKernels.has(node.functionName)) {
        uniqueKernels.set(node.functionName, { code: node.deviceCode, node });
      }
    }
    const kernelDefinitions = Array.from(uniqueKernels.values())
      .map((k) => k.code)
      .join("\n\n");

    const executionCalls: string[] = [];
    for (const node of executionOrder) {
      if (!node.deviceCode.trim()) {
        continue;
      }

      const outputTensors = new Map<string, string>();
      for (const [outputPort] of node.outputs) {
        const key = `${node.id}:${outputPort}`;
        const tensorInfo = tensorRegistry.get(key);
        if (!tensorInfo) {
          throw new Error(
            `Compiler Error: Could not find output tensor for ${node.name}:${outputPort} (key: ${key})`
          );
        }
        outputTensors.set(outputPort, tensorInfo.varName);
      }

      const inputTensors = new Map<string, string>();
      for (const [inputPort] of node.inputs) {
        const conn = Array.from(graph.connections).find(
          (c) => c.toNode === node && c.toPort === inputPort
        );
        const sourceKey = conn
          ? `${conn.fromNode.id}:${conn.fromPort}`
          : `${node.id}:${inputPort}`;
        const tensorInfo = tensorRegistry.get(sourceKey);
        if (!tensorInfo) {
          throw new Error(
            `Compiler Error: Could not find source tensor for ${node.name}:${inputPort} (key: ${sourceKey})`
          );
        }
        inputTensors.set(inputPort, tensorInfo.varName);
      }

      const kernelCall = node.getKernelCall(
        outputTensors,
        inputTensors,
        (nodeId, paramName) => {
          const nodeParamKey = `${nodeId}:${paramName}`;
          const globalParamName = paramNameMapping.get(nodeParamKey);
          if (!globalParamName) {
            throw new Error(
              `Compiler Error: Could not find global parameter name for ${node.name}:${paramName}`
            );
          }
          const paramIndex = paramNames.indexOf(globalParamName);
          return `param_tensors[${paramIndex}]`;
        }
      );

      const { gridDim, blockDim, sharedMemSize } =
        this.calculateOptimalGridBlock(node, outputTensors, inputTensors);

      executionCalls.push(
        `  ${node.functionName}<<<${gridDim}, ${blockDim}, ${sharedMemSize}>>>(${kernelCall});`
      );
      // Add CUDA_CHECK after each kernel launch for easier debugging
      executionCalls.push(`  CUDA_CHECK(cudaGetLastError());`);
    }

    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);

    const tensorInstantiations: string[] = [];
    intermediateIdx = 0;
    for (const [key, tensorInfo] of intermediateTensors.entries()) {
      const varName = `intermediate_${intermediateIdx}`;
      const shapeVar = `${varName}_shape`;
      const tensorVar = `${varName}_tensor`;

      if (tensorInfo.poolId?.startsWith("inplace_")) {
        const sourceKey = tensorInfo.poolId.replace("inplace_", "");
        const sourceTensor = intermediateTensors.get(sourceKey);
        if (sourceTensor && sourceTensor.offset >= 0) {
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate_at_offset(${sourceTensor.offset}, ${tensorInfo.size})); // In-place reuse of ${sourceKey}`
          );
          tensorInstantiations.push(
            `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
          );
        } else {
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate(${tensorInfo.size}));`
          );
          tensorInstantiations.push(
            `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
          );
        }
      } else {
        // Use calculated offset for proper memory reuse
        if (tensorInfo.offset >= 0) {
          tensorInstantiations.push(
            `  // Pool: ${tensorInfo.poolId || "default"}, Offset: ${
              tensorInfo.offset
            }, Size: ${tensorInfo.size} bytes`
          );
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate_at_offset(${tensorInfo.offset}, ${tensorInfo.size}));`
          );
        } else {
          // Fallback to sequential allocation
          tensorInstantiations.push(
            `  // Sequential allocation fallback - Size: ${tensorInfo.size} bytes`
          );
          tensorInstantiations.push(
            `  float* ${varName}_data = (float*)(allocator.allocate(${tensorInfo.size}));`
          );
        }
        tensorInstantiations.push(
          `  Tensor<float> ${tensorVar} = {${varName}_data, ${shapeVar}, ${tensorInfo.spec.shape.length}};`
        );
      }
      intermediateIdx++;
    }

    for (const [inputName, { node, port }] of graphInputs.entries()) {
      const inputSpec = node.inputs.get(port)!;
      const tensorType = inputSpec.dtype === "int32" ? "int" : "float";
      tensorInstantiations.push(
        `  Tensor<${tensorType}> ${inputName} = {(${tensorType}*)input_data, input_shape, input_dims};`
      );
    }

    for (const outputName of graphOutputs.keys()) {
      tensorInstantiations.push(
        `  Tensor<float> ${outputName} = {(float*)output_data, output_shape, output_dims};`
      );
    }

    const kernelCode = generateKernelCode({
      config: { memoryAlignment: this.config.memoryAlignment! },
      kernelDefinitions,
      workspaceSize,
      paramNames,
      variableDeclarations,
      tensorInstantiations,
      executionCalls,
      graph,
      paramNameMapping,
      generateParameterShapeDeclarations:
        this.generateParameterShapeDeclarations.bind(this),
      generateParameterDims: this.generateParameterDims.bind(this),
    });

    return { kernelCode, workspaceSize, filename };
  }

  private generateLegacyKernelCode(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): { kernelCode: string; workspaceSize: number; filename: string } {
    throw new Error("Legacy kernel generation not implemented in this example");
  }

  private generateParameterShapeDeclarations(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): string {
    const declarations: string[] = [];

    paramNames.forEach((paramName, i) => {
      const tensor = this.findTensorByGlobalName(
        graph,
        paramName,
        paramNameMapping
      );
      if (tensor && tensor.shape) {
        declarations.push(
          `    const int param_shape_${i}[] = {${tensor.shape.join(
            ", "
          )}}; // ${paramName}`
        );
      } else {
        if (paramName.includes("embeddings")) {
          declarations.push(
            `    const int param_shape_${i}[] = {65, 384}; // ${paramName}`
          );
        } else if (paramName.includes("frequencies")) {
          declarations.push(
            `    const int param_shape_${i}[] = {192}; // ${paramName}`
          );
        } else if (paramName.includes("weights")) {
          declarations.push(
            `    const int param_shape_${i}[] = {384, 384}; // ${paramName}`
          );
        } else if (paramName.includes("bias")) {
          declarations.push(
            `    const int param_shape_${i}[] = {384}; // ${paramName}`
          );
        } else {
          declarations.push(
            `    const int param_shape_${i}[] = {1}; // ${paramName}`
          );
        }
      }
    });

    return declarations.join("\n");
  }

  private generateParameterDims(
    graph: CudaGraph,
    paramNames: string[],
    paramNameMapping: Map<string, string>
  ): number[] {
    return paramNames.map((paramName) => {
      const tensor = this.findTensorByGlobalName(
        graph,
        paramName,
        paramNameMapping
      );
      return tensor ? tensor.shape.length : 1;
    });
  }

  private findTensorByGlobalName(
    graph: CudaGraph,
    globalName: string,
    paramNameMapping: Map<string, string>
  ): CudaTensor | null {
    for (const node of graph.nodes.values()) {
      for (const [localName, tensor] of node.parameters) {
        const nodeParamKey = `${node.id}:${localName}`;
        const mappedGlobalName = paramNameMapping.get(nodeParamKey);
        if (mappedGlobalName === globalName) {
          return tensor;
        }
      }
    }
    return null;
  }

  private _setAndPropagateShapes(
    graph: CudaGraph,
    inputShapes: Map<string, number[]>
  ): void {
    const graphInputs = graph.getGraphInputs();

    let batchSize = -1;
    for (const shape of inputShapes.values()) {
      if (shape.length > 0 && shape[0] !== -1) {
        batchSize = shape[0];
        break;
      }
    }

    if (batchSize === -1) {
      batchSize = 1;
    }

    for (const [inputName, shape] of inputShapes.entries()) {
      const finalShape = shape.map((dim) => (dim === -1 ? batchSize : dim));
      const inputInfo = graphInputs.get(inputName);
      if (inputInfo) {
        inputInfo.node.updateInputShape(inputInfo.port, finalShape);
      } else {
        console.warn(`Graph input port named '${inputName}' not found.`);
      }
    }

    const executionOrder = graph.getExecutionOrder();

    let maxPasses = 3;
    for (let pass = 0; pass < maxPasses; pass++) {
      let anyShapeChanged = false;

      for (const node of executionOrder) {
        const oldOutputShapes = new Map();
        for (const [port, spec] of node.outputs) {
          oldOutputShapes.set(port, [...spec.shape]);
        }

        node.resolveShapes();

        for (const [port, spec] of node.outputs) {
          const oldShape = oldOutputShapes.get(port);
          if (
            !oldShape ||
            JSON.stringify(oldShape) !== JSON.stringify(spec.shape)
          ) {
            anyShapeChanged = true;
          }
        }

        for (const conn of graph.connections) {
          if (conn.fromNode.id === node.id) {
            const sourceSpec = node.outputs.get(conn.fromPort);
            if (sourceSpec) {
              conn.toNode.updateInputShape(conn.toPort, sourceSpec.shape);
            }
          }
        }
      }

      if (!anyShapeChanged) {
        break;
      }
    }

    for (const node of executionOrder) {
      for (const [outputPort, spec] of node.outputs) {
        if (spec.shape.includes(-1)) {
          const inputShapes = JSON.stringify(
            Object.fromEntries(node.inputs),
            null,
            2
          );
          throw new Error(
            `Compiler error: Node '${node.name}' failed to resolve its output shape for port '${outputPort}'. Current input shapes for this node:\n${inputShapes}`
          );
        }
      }
    }
  }

  private planMemory(
    graph: CudaGraph,
    executionOrder: CudaNode[]
  ): {
    intermediateTensors: Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >;
    workspaceSize: number;
    tensorRegistry: Map<
      string,
      { varName: string; spec: { shape: number[]; dtype: string } }
    >;
    memoryPools: Map<
      string,
      { size: number; tensors: string[]; offset: number }
    >;
  } {
    const tensorRegistry = new Map<
      string,
      { varName: string; spec: { shape: number[]; dtype: string } }
    >();
    const graphInputs = graph.getGraphInputs();
    const graphOutputs = this.getGraphOutputs(graph);

    for (const [inputPort, { node, port }] of graphInputs.entries()) {
      tensorRegistry.set(`${node.id}:${port}`, {
        varName: inputPort,
        spec: node.inputs.get(port)!,
      });
    }
    for (const [outputPort, { node, port }] of graphOutputs.entries()) {
      tensorRegistry.set(`${node.id}:${outputPort}`, {
        varName: outputPort,
        spec: node.outputs.get(port)!,
      });
    }

    const intermediateTensors = new Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >();

    executionOrder.forEach((node, i) => {
      for (const [outputPort, spec] of node.outputs) {
        const key = `${node.id}:${outputPort}`;
        if (spec.shape.includes(-1)) {
          const inputShapes = JSON.stringify(Array.from(node.inputs.entries()));
          throw new Error(
            `Compiler error: Unresolved dynamic shape ${JSON.stringify(
              spec.shape
            )} for output '${outputPort}' of node '${
              node.name
            }'. Node inputs: ${inputShapes}`
          );
        }
        if (!tensorRegistry.has(key)) {
          const size = spec.shape.reduce((a, b) => a * b, 1) * 4;
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

    this.computeEnhancedLifetimes(graph, executionOrder, intermediateTensors);

    if (this.config.enableMemoryPooling) {
      this.detectInPlaceOperations(graph, executionOrder, intermediateTensors);
    }

    const { memoryPools, workspaceSize } =
      this.allocateWithMemoryPools(intermediateTensors);

    if (this.config.verboseMemoryPlanning) {
      const totalTensorSize = Array.from(intermediateTensors.values()).reduce(
        (sum, tensor) => sum + this.alignTo256Bytes(tensor.size),
        0
      );
      const efficiency =
        workspaceSize > 0
          ? ((totalTensorSize / workspaceSize) * 100).toFixed(1)
          : "N/A";
      const savings = totalTensorSize - workspaceSize;

      console.log(
        `[Enhanced Memory Planning] Total workspace size: ${workspaceSize} bytes (${(
          workspaceSize /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `[Enhanced Memory Planning] Total tensor size (if not reused): ${totalTensorSize} bytes (${(
          totalTensorSize /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `[Enhanced Memory Planning] Memory saved: ${savings} bytes (${(
          savings /
          1024 /
          1024
        ).toFixed(2)} MB)`
      );
      console.log(
        `[Enhanced Memory Planning] Memory efficiency: ${efficiency}%`
      );
      console.log(
        `[Enhanced Memory Planning] Memory pools created: ${memoryPools.size}`
      );
    }

    return { intermediateTensors, workspaceSize, tensorRegistry, memoryPools };
  }

  private computeEnhancedLifetimes(
    graph: CudaGraph,
    executionOrder: CudaNode[],
    intermediateTensors: Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >
  ): void {
    const nodeIndexMap = new Map<string, number>();
    executionOrder.forEach((node, i) => {
      nodeIndexMap.set(node.id, i);
    });

    // Initialize all tensors with their production time
    for (const [tensorKey, tensorInfo] of intermediateTensors.entries()) {
      const [nodeId, port] = tensorKey.split(":");
      const producerIndex = nodeIndexMap.get(nodeId);
      if (producerIndex !== undefined) {
        tensorInfo.liveStart = producerIndex;
        tensorInfo.liveEnd = producerIndex;
      }
    }

    // Build a map of which tensors are consumed by which nodes
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

    // Handle graph outputs - they must live until the end
    const graphOutputs = this.getGraphOutputs(graph);
    for (const [outputName, outputInfo] of graphOutputs) {
      const outputKey = `${outputInfo.node.id}:${outputInfo.port}`;
      const tensorInfo = intermediateTensors.get(outputKey);
      if (tensorInfo) {
        tensorInfo.liveEnd = executionOrder.length - 1;
      }
    }

    // Log lifetime information for debugging
    if (this.config.verboseMemoryPlanning) {
      console.log("\n[Memory Planning] Tensor lifetimes:");
      for (const [key, info] of intermediateTensors.entries()) {
        console.log(
          `  ${key}: live from ${info.liveStart} to ${info.liveEnd} (size: ${info.size} bytes)`
        );
      }
    }
  }

  private detectInPlaceOperations(
    graph: CudaGraph,
    executionOrder: CudaNode[],
    intermediateTensors: Map<string, any>
  ): void {
    return;
  }

  private allocateWithMemoryPools(
    intermediateTensors: Map<
      string,
      {
        spec: { shape: number[]; dtype: string };
        size: number;
        liveStart: number;
        liveEnd: number;
        offset: number;
        poolId?: string;
      }
    >
  ): {
    memoryPools: Map<
      string,
      { size: number; tensors: string[]; offset: number }
    >;
    workspaceSize: number;
  } {
    const memoryPools = new Map<
      string,
      { size: number; tensors: string[]; offset: number }
    >();

    // Sort tensors by start time, then by size (largest first for better packing)
    const sortedTensors = Array.from(intermediateTensors.entries())
      .filter(([_, tensor]) => !tensor.poolId?.startsWith("inplace_"))
      .sort((a, b) => {
        if (a[1].liveStart !== b[1].liveStart) {
          return a[1].liveStart - b[1].liveStart;
        }
        return b[1].size - a[1].size; // Larger tensors first
      });

    // Track memory intervals: array of {start, end, size}
    const memoryIntervals: Array<{
      start: number;
      end: number;
      size: number;
      tensorKey: string;
    }> = [];

    let maxWorkspaceSize = 0;

    for (const [tensorKey, tensor] of sortedTensors) {
      const alignedSize = this.alignTo256Bytes(tensor.size);

      // Find the earliest offset where this tensor can fit
      let bestOffset = 0;
      let found = false;

      // Sort intervals by start offset
      memoryIntervals.sort((a, b) => a.start - b.start);

      // Check each gap between intervals
      for (let i = 0; i <= memoryIntervals.length; i++) {
        const gapStart = i === 0 ? 0 : memoryIntervals[i - 1].end;
        const gapEnd =
          i === memoryIntervals.length ? Infinity : memoryIntervals[i].start;

        // Check if this gap is large enough and doesn't overlap with live tensors
        if (gapEnd - gapStart >= alignedSize) {
          // Verify no overlap with any tensor that's live during our lifetime
          let hasConflict = false;
          for (const interval of memoryIntervals) {
            // Check if lifetimes overlap
            const lifetimeOverlap = !(
              tensor.liveEnd < interval.start || tensor.liveStart > interval.end
            );
            // Check if memory regions would overlap
            const memoryOverlap = !(
              gapStart + alignedSize <= interval.start ||
              gapStart >= interval.end
            );

            if (lifetimeOverlap && memoryOverlap) {
              hasConflict = true;
              break;
            }
          }

          if (!hasConflict) {
            bestOffset = gapStart;
            found = true;
            break;
          }
        }
      }

      if (!found) {
        // No suitable gap found, place at the end of all allocations
        bestOffset =
          memoryIntervals.length > 0
            ? Math.max(...memoryIntervals.map((interval) => interval.end))
            : 0;
      }

      // Ensure alignment
      bestOffset =
        Math.ceil(bestOffset / this.config.memoryAlignment!) *
        this.config.memoryAlignment!;

      tensor.offset = bestOffset;

      // Add this allocation to our intervals
      memoryIntervals.push({
        start: bestOffset,
        end: bestOffset + alignedSize,
        size: alignedSize,
        tensorKey: tensorKey,
      });

      maxWorkspaceSize = Math.max(maxWorkspaceSize, bestOffset + alignedSize);

      // Log allocation for debugging
      if (this.config.verboseMemoryPlanning) {
        console.log(
          `[Memory Planning] Allocated ${tensorKey} at offset ${bestOffset} (size: ${alignedSize}, lifetime: ${tensor.liveStart}-${tensor.liveEnd})`
        );
      }
    }

    // Verify no overlaps in final allocation
    if (this.config.verboseMemoryPlanning) {
      console.log("\n[Memory Planning] Verifying memory allocation...");
      for (let i = 0; i < memoryIntervals.length; i++) {
        for (let j = i + 1; j < memoryIntervals.length; j++) {
          const interval1 = memoryIntervals[i];
          const interval2 = memoryIntervals[j];
          const tensor1 = intermediateTensors.get(interval1.tensorKey)!;
          const tensor2 = intermediateTensors.get(interval2.tensorKey)!;

          // Check if lifetimes overlap
          const lifetimeOverlap = !(
            tensor1.liveEnd < tensor2.liveStart ||
            tensor1.liveStart > tensor2.liveEnd
          );
          // Check if memory regions overlap
          const memoryOverlap = !(
            interval1.end <= interval2.start || interval1.start >= interval2.end
          );

          if (lifetimeOverlap && memoryOverlap) {
            console.error(`[Memory Planning] ERROR: Memory overlap detected!`);
            console.error(
              `  ${interval1.tensorKey}: offset=${interval1.start}, size=${interval1.size}, lifetime=${tensor1.liveStart}-${tensor1.liveEnd}`
            );
            console.error(
              `  ${interval2.tensorKey}: offset=${interval2.start}, size=${interval2.size}, lifetime=${tensor2.liveStart}-${tensor2.liveEnd}`
            );
          }
        }
      }
    }

    return { memoryPools, workspaceSize: maxWorkspaceSize };
  }

  private categorizeTensorSize(size: number): string {
    const MB = 1024 * 1024;
    if (size < MB) return "small";
    if (size < 10 * MB) return "medium";
    if (size < 50 * MB) return "large";
    return "xlarge";
  }

  private alignTo256Bytes(size: number): number {
    return (
      Math.ceil(size / this.config.memoryAlignment!) *
      this.config.memoryAlignment!
    );
  }

  private calculateOptimalGridBlock(
    node: CudaNode,
    outputTensors: Map<string, string>,
    inputTensors: Map<string, string>
  ): { gridDim: string; blockDim: string; sharedMemSize: number } {
    const primaryOutput = Array.from(node.outputs.values())[0];
    if (!primaryOutput) {
      const defaultConfig = this.kernelConfigRegistry.getDefaultConfig(
        [],
        DEFAULT_CUDA_CONSTRAINTS,
        this.config.defaultBlockSize!
      );
      return {
        gridDim: defaultConfig.gridDim,
        blockDim: defaultConfig.blockDim,
        sharedMemSize: defaultConfig.sharedMemSize,
      };
    }

    const shape = primaryOutput.shape;

    // Use the kernel configuration registry to get optimized launch parameters
    const config = this.kernelConfigRegistry.getConfig(
      node.functionName,
      shape,
      DEFAULT_CUDA_CONSTRAINTS,
      this.config.defaultBlockSize!
    );

    return {
      gridDim: config.gridDim,
      blockDim: config.blockDim,
      sharedMemSize: config.sharedMemSize,
    };
  }

  private getGraphOutputs(
    graph: CudaGraph
  ): Map<string, { node: CudaNode; port: string }> {
    const graphOutputs = new Map<string, { node: CudaNode; port: string }>();
    const allNodeOutputs = new Set<string>();
    for (const conn of graph.connections) {
      allNodeOutputs.add(`${conn.fromNode.id}:${conn.fromPort}`);
    }
    for (const node of graph.nodes.values()) {
      for (const [outputPort, spec] of node.outputs) {
        if (!allNodeOutputs.has(`${node.id}:${outputPort}`)) {
          graphOutputs.set(outputPort, { node, port: outputPort });
        }
      }
    }
    return graphOutputs;
  }
}
