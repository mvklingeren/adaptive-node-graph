// test-basics.ts
// Contains all the basic tests from the /examples directory

import {
  Graph,
  AdaptiveNode,
  OscillatorNode,
  createProcessor,
  createRouterNode,
  createLoadBalancerNode,
  createAddNode,
  createFloat32MultiplyNode,
  createParallelNode,
} from "./core";

// ============================================================================
// 1. Audio Processing Demo
// ============================================================================

async function audioProcessingDemo() {
  console.log("=== Audio Processing Demo ===");
  // Envelope generator
  const envelope = createProcessor<number, number>((time) => {
    // Simple ADSR envelope
    if (time < 0.1) return time / 0.1; // Attack
    if (time < 0.2) return 1; // Decay to sustain
    if (time < 0.8) return 0.7; // Sustain
    return 0.7 * (1 - (time - 0.8) / 0.2); // Release
  }, "envelope");

  // Filter node
  const filter = new AdaptiveNode<Float32Array, Float32Array>((samples) => {
    // Simple low-pass filter
    const filtered = new Float32Array(samples.length);
    filtered[0] = samples[0];

    for (let i = 1; i < samples.length; i++) {
      filtered[i] = filtered[i - 1] * 0.9 + samples[i] * 0.1;
    }

    return filtered;
  }).setLabel("lowpass");

  // Build synthesizer graph
  const graph = new Graph();
  const osc = new OscillatorNode();
  const gain = createProcessor<[Float32Array, number], Float32Array>(
    ([samples, gainValue]) => samples.map((s) => s * gainValue),
    "gain"
  );

  graph.addNode(osc);
  graph.addNode(envelope);
  graph.addNode(filter);
  graph.addNode(gain);

  // Generate 1 second of audio
  const params = {
    frequency: 440,
    amplitude: 0.5,
    sampleRate: 44100,
    length: 44100,
    waveform: "sawtooth" as const,
  };

  const samples = await osc.process(params);
  const filtered = await filter.process(samples!);

  // Apply envelope
  const envelopeValues = await Promise.all(
    Array.from({ length: 44100 }, (_, i) => envelope.process(i / 44100))
  );
  const output = filtered?.map((sample, i) => sample * envelopeValues[i]!);
  console.log("Audio processing demo complete.");
}

// ============================================================================
// 2. Basic Math Demo
// ============================================================================

async function basicMathDemo() {
  console.log("\n=== Basic Math Demo ===");
  const graph = new Graph();

  // Create nodes
  const input = createProcessor<number, number>((x) => x, "input");
  const add5 = createProcessor<number, number>((x) => x + 5, "add5");
  const multiply2 = createProcessor<number, number>((x) => x * 2, "multiply2");
  const output = createProcessor<number, void>(
    (x) => console.log("Result:", x),
    "output"
  );

  // Build graph: (input + 5) * 2
  graph.addNode(input).addNode(add5).addNode(multiply2).addNode(output);

  graph.connect(input, add5);
  graph.connect(add5, multiply2);
  graph.connect(multiply2, output);

  // Execute
  await graph.execute(10); // Result: 30
}

// ============================================================================
// 3. Conditional Routing Demo
// ============================================================================

async function conditionalRoutingDemo() {
  console.log("\n=== Conditional Routing Demo ===");
  const graph = new Graph();

  // Temperature monitoring system
  const tempSensor = createProcessor(() => Math.random() * 50, "temp-sensor");
  const threshold = createProcessor<number, boolean>(
    (temp) => temp > 30,
    "threshold"
  );
  const alert = createProcessor<number, void>(
    (temp) => console.log(`🔥 HIGH TEMP: ${temp}°C`),
    "alert"
  );
  const normal = createProcessor<number, void>(
    (temp) => console.log(`✓ Normal: ${temp}°C`),
    "normal"
  );

  // Create conditional routing
  const router = createProcessor<[boolean, number], void>(([isHigh, temp]) => {
    if (isHigh) {
      alert.process(temp);
    } else {
      normal.process(temp);
    }
  }, "router");

  graph
    .addNode(tempSensor)
    .addNode(threshold)
    .addNode(router)
    .addNode(alert)
    .addNode(normal);

  // Connect: sensor -> threshold check -> router
  const sensorOutput = await tempSensor.process(null);
  if (sensorOutput !== null) {
    const isHigh = await threshold.process(sensorOutput);
    if (isHigh !== null) {
      await router.process([isHigh, sensorOutput]);
    }
  }
}

// ============================================================================
// 4. Dynamic Graph Demo
// ============================================================================

function dynamicGraphDemo() {
  console.log("\n=== Dynamic Graph Demo ===");
  console.log("Dynamic graph demo is conceptual and commented out.");
  // Add nodes dynamically based on conditions
  // if (needsCaching) {
  //     const cache = createCacheNode(5000);
  //     graph.addNode(cache);
  //     graph.connect(dataSource, cache);
  //     graph.connect(cache, processor);
  //   } else {
  //     graph.connect(dataSource, processor);
  //   }
}

// ============================================================================
// 5. Error Handling Demo
// ============================================================================

async function errorHandlingDemo() {
  console.log("\n=== Error Handling Demo ===");
  // Define a recursive type for processable data
  type Processable =
    | string
    | number
    | Processable[]
    | { [key: string]: Processable };

  // Define the processData function with proper return type
  const processData = (data: Processable | null | undefined): Processable => {
    // Example processing logic
    if (typeof data === "number") {
      return data * 2;
    } else if (typeof data === "string") {
      return data.toUpperCase();
    } else if (Array.isArray(data)) {
      // Recursively process each item in the array
      return data.map(processData);
    } else if (data === null || data === undefined) {
      throw new Error("Cannot process null or undefined data");
    } else if (typeof data === "object") {
      // Recursively process each value in the object
      return Object.fromEntries(
        Object.entries(data).map(([key, value]) => [
          key,
          processData(value as Processable),
        ])
      );
    }
    // Handle unsupported data types
    throw new Error(`Unsupported data type: ${typeof data}`);
  };

  // Define the result type, which can be processed data or an error object
  type ProcessResult = Processable | { error: string; input: any };

  const safeProcessor = createProcessor<Processable | null, ProcessResult>(
    (input) => {
      try {
        // Processing logic
        return processData(input);
      } catch (error: unknown) {
        console.error("Processing failed:", error);
        // Handle the unknown error type
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        return { error: errorMessage, input };
      }
    },
    "safe-processor"
  );

  console.log('Processing { a: "hello", b: [1, 2] }');
  const result1 = await safeProcessor.process({ a: "hello", b: [1, 2] });
  console.log("Result:", result1);

  console.log("Processing null");
  const result2 = await safeProcessor.process(null);
  console.log("Result:", result2);
}

// ============================================================================
// 6. Machine Learning Demo
// ============================================================================

async function machineLearningDemo() {
  console.log("\n=== Machine Learning Demo ===");
  // ML Pipeline nodes
  class DataPreprocessor extends AdaptiveNode<any, Float32Array> {
    constructor() {
      super((data) => new Float32Array(data));

      this.register(Array, this.preprocessArray.bind(this));
      this.register(Float32Array, this.preprocessFloat32.bind(this));
    }

    private preprocessArray(data: number[]): Float32Array {
      // Normalize array data
      const arr = new Float32Array(data);
      const mean = arr.reduce((a, b) => a + b) / arr.length;
      const std = Math.sqrt(
        arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length
      );

      return arr.map((val) => (val - mean) / std);
    }

    private preprocessFloat32(data: Float32Array): Float32Array {
      return this.preprocessArray(Array.from(data));
    }
  }

  // Feature extractor
  const featureExtractor = createProcessor<Float32Array, Float32Array>(
    (data) => {
      // Extract statistical features
      const features = new Float32Array(5);

      features[0] = Math.min(...data);
      features[1] = Math.max(...data);
      features[2] = data.reduce((a, b) => a + b) / data.length;
      features[3] = Math.sqrt(
        data.reduce((sum, val) => sum + val * val, 0) / data.length
      );
      features[4] = data.reduce(
        (sum, val, i) => (i > 0 ? sum + Math.abs(val - data[i - 1]) : sum),
        0
      );

      return features;
    },
    "feature-extractor"
  );

  // Mock ML model
  const model = createProcessor<
    Float32Array,
    { class: string; confidence: number }
  >((features) => {
    // Simulate model inference
    const sum = features.reduce((a, b) => a + b);

    if (sum > 10) {
      return { class: "positive", confidence: 0.85 };
    } else if (sum < -10) {
      return { class: "negative", confidence: 0.9 };
    } else {
      return { class: "neutral", confidence: 0.75 };
    }
  }, "ml-model");

  // Post-processor
  const postProcessor = new AdaptiveNode<any, any>((prediction) => prediction)
    .register(Object, (pred: any) => {
      if (pred.confidence < 0.5) {
        return { ...pred, class: "uncertain" };
      }
      return pred;
    })
    .setLabel("post-processor");

  // Build ML pipeline
  const graph = new Graph();
  const preprocessor = new DataPreprocessor().setLabel("preprocessor");

  graph.addNode(preprocessor);
  graph.addNode(featureExtractor);
  graph.addNode(model);
  graph.addNode(postProcessor);

  graph.connect(preprocessor, featureExtractor);
  graph.connect(featureExtractor, model);
  graph.connect(model, postProcessor);

  // Test with different data types
  const testData = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    new Float32Array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]),
    Array.from({ length: 100 }, () => Math.random() * 10 - 5),
  ];

  for (const data of testData) {
    const result = await graph.execute(data, preprocessor.id);
    console.log("Prediction:", result);
  }
}

// ============================================================================
// 7. Multi-protocol Demo
// ============================================================================

async function multiProtocolDemo() {
  console.log("\n=== Multi-protocol Demo ===");
  // Request/Response types
  interface HTTPRequest {
    method: string;
    path: string;
    headers: Record<string, string>;
    body?: any;
  }

  interface WebSocketMessage {
    type: "ws";
    action: string;
    payload: any;
  }

  interface GRPCRequest {
    service: string;
    method: string;
    data: any;
  }

  // Multi-protocol gateway node
  class APIGateway extends AdaptiveNode<any, any> {
    private rateLimits = new Map<string, number[]>();
    private cache = new Map<string, { data: any; expires: number }>();

    constructor() {
      super((request) => ({ error: "Unknown protocol" }));

      this.register(Object, this.routeByShape.bind(this));
    }

    private routeByShape(request: any) {
      // HTTP Request
      if ("method" in request && "path" in request) {
        return this.handleHTTP(request as HTTPRequest);
      }

      // WebSocket Message
      if (request.type === "ws" && "action" in request) {
        return this.handleWebSocket(request as WebSocketMessage);
      }

      // gRPC Request
      if ("service" in request && "method" in request) {
        return this.handleGRPC(request as GRPCRequest);
      }

      return { error: "Unrecognized request format" };
    }

    private handleHTTP(request: HTTPRequest) {
      // Check rate limits
      const clientId = request.headers["x-client-id"] || "anonymous";
      if (!this.checkRateLimit(clientId)) {
        return { status: 429, error: "Rate limit exceeded" };
      }

      // Check cache
      const cacheKey = `${request.method}:${request.path}`;
      const cached = this.cache.get(cacheKey);
      if (cached && cached.expires > Date.now()) {
        return { status: 200, data: cached.data, cached: true };
      }

      // Route to appropriate service
      const response = this.routeHTTPRequest(request);

      // Cache successful GET requests
      if (request.method === "GET" && response.status === 200) {
        this.cache.set(cacheKey, {
          data: response.data,
          expires: Date.now() + 60000, // 1 minute
        });
      }

      return response;
    }

    private handleWebSocket(message: WebSocketMessage) {
      // Real-time message handling
      switch (message.action) {
        case "subscribe":
          return { type: "subscription", channel: message.payload.channel };
        case "publish":
          return { type: "broadcast", sent: true };
        default:
          return { type: "error", message: "Unknown action" };
      }
    }

    private handleGRPC(request: GRPCRequest) {
      // gRPC routing
      return {
        service: request.service,
        method: request.method,
        response: `Processed ${request.service}.${request.method}`,
      };
    }

    private checkRateLimit(clientId: string): boolean {
      const now = Date.now();
      const window = 60000; // 1 minute
      const limit = 100; // requests per minute

      if (!this.rateLimits.has(clientId)) {
        this.rateLimits.set(clientId, []);
      }

      const requests = this.rateLimits.get(clientId)!;
      const recentRequests = requests.filter((time) => now - time < window);

      if (recentRequests.length >= limit) {
        return false;
      }

      recentRequests.push(now);
      this.rateLimits.set(clientId, recentRequests);
      return true;
    }

    private routeHTTPRequest(request: HTTPRequest) {
      // Simplified routing logic
      if (request.path.startsWith("/api/users")) {
        return { status: 200, data: { users: [] } };
      }
      if (request.path.startsWith("/api/products")) {
        return { status: 200, data: { products: [] } };
      }
      return { status: 404, error: "Not found" };
    }
  }

  // Service-specific processors
  const authService = createProcessor<any, any>((request) => {
    // Authentication logic
    return { ...request, authenticated: true };
  }, "auth-service");

  const loggingService = createProcessor<any, any>((request) => {
    console.log(`[${new Date().toISOString()}] Request:`, request);
    return request;
  }, "logging-service");

  const analyticsService = new AdaptiveNode<any, any>((event) => {
    // Track metrics
    return event;
  })
    .register(Object, (event: any) => {
      if (event.status >= 400) {
        console.log("Error event:", event);
      }
      return event;
    })
    .setLabel("analytics");

  // Build the API gateway graph
  const graph = new Graph();
  const gateway = new APIGateway().setLabel("api-gateway");

  graph.addNode(loggingService);
  graph.addNode(authService);
  graph.addNode(gateway);
  graph.addNode(analyticsService);

  graph.connect(loggingService, authService);
  graph.connect(authService, gateway);
  graph.connect(gateway, analyticsService);

  // Test different protocols
  const requests = [
    // HTTP Request
    {
      method: "GET",
      path: "/api/users/123",
      headers: { "x-client-id": "client-1" },
    },
    // WebSocket Message
    {
      type: "ws",
      action: "subscribe",
      payload: { channel: "updates" },
    },
    // gRPC Request
    {
      service: "UserService",
      method: "GetUser",
      data: { userId: 123 },
    },
  ];

  for (const request of requests) {
    const result = await graph.execute(request, loggingService.id);
    console.log("Result:", result);
  }
}

// ============================================================================
// 8. Oscillator Demo
// ============================================================================

async function oscillatorDemo() {
  console.log("\n=== Oscillator Demo ===");
  const graph = new Graph();

  const osc = new OscillatorNode();
  const multiply = createFloat32MultiplyNode();

  graph.addNode(osc);
  graph.addNode(multiply);
  graph.connect(osc, multiply);

  const result = await graph.execute({
    frequency: 440,
    amplitude: 0.5,
    sampleRate: 44100,
  });
  console.log("Oscillator result:", result);
}

// ============================================================================
// 9. Performance Monitoring Demo
// ============================================================================

async function performanceMonitoringDemo() {
  console.log("\n=== Performance Monitoring Demo ===");
  // Define the missing function
  function expensiveOperation(input: any): any {
    // Simulate an expensive operation
    // This could be a complex calculation, data processing, etc.
    const iterations = input?.iterations || 1000000;
    let result = 0;

    for (let i = 0; i < iterations; i++) {
      result += Math.sin(i) * Math.cos(i);
    }

    return result;
  }

  const monitoredNode = new AdaptiveNode<any, any>((input) => {
    const start = performance.now();
    const result = expensiveOperation(input);
    const duration = performance.now() - start;

    if (duration > 100) {
      console.warn(`Slow operation: ${duration}ms`);
    }

    return result;
  }).setLabel("monitored");

  await monitoredNode.process({ iterations: 2000000 });

  // Check performance stats
  const stats = monitoredNode.getPerformanceStats();
  console.log("Performance stats:", stats);
}

// ============================================================================
// 10. Real-time Stream Demo
// ============================================================================

async function realtimeStreamDemo() {
  console.log("\n=== Real-time Stream Demo ===");
  // Stream processor that handles different event types
  class StreamProcessor extends AdaptiveNode<any, any> {
    private buffer: any[] = [];
    private windowSize = 100;

    constructor() {
      super((event) => event);

      // Register processors for different event types
      this.register(MouseEvent, this.processMouseEvent.bind(this));
      this.register(KeyboardEvent, this.processKeyboardEvent.bind(this));
      this.register(Object, this.processDataEvent.bind(this));
    }

    private processMouseEvent(event: MouseEvent) {
      return {
        type: "mouse",
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now(),
      };
    }

    private processKeyboardEvent(event: KeyboardEvent) {
      return {
        type: "keyboard",
        key: event.key,
        timestamp: Date.now(),
      };
    }

    private processDataEvent(event: any) {
      this.buffer.push(event);
      if (this.buffer.length > this.windowSize) {
        this.buffer.shift();
      }

      // Compute sliding window statistics
      return {
        type: "data",
        count: this.buffer.length,
        average: this.computeAverage(),
        timestamp: Date.now(),
      };
    }

    private computeAverage(): number {
      if (this.buffer.length === 0) return 0;
      const sum = this.buffer.reduce((acc, val) => acc + (val.value || 0), 0);
      return sum / this.buffer.length;
    }
  }

  // Create multiple workers for parallel processing
  const workers = Array.from({ length: 4 }, (_, i) =>
    new StreamProcessor().setLabel(`worker-${i}`)
  );

  // Load balancer distributes events across workers
  const loadBalancer = createLoadBalancerNode(workers);

  // Aggregator collects results from all workers
  const aggregator = createProcessor<any, void>((result) => {
    console.log(`Processed event:`, result);
  }, "aggregator");

  // Build the graph
  const graph = new Graph();
  graph.addNode(loadBalancer);
  workers.forEach((w) => graph.addNode(w));
  graph.addNode(aggregator);

  // Connect load balancer to all workers and workers to aggregator
  workers.forEach((worker) => {
    graph.connect(loadBalancer, worker);
    graph.connect(worker, aggregator);
  });

  // Simulate event stream
  const events = [
    new MouseEvent("click", { clientX: 100, clientY: 200 }),
    { value: 42 },
    new KeyboardEvent("keydown", { key: "Enter" }),
    { value: 38 },
    { value: 45 },
  ];

  for (const event of events) {
    await loadBalancer.process(event);
  }
}

// ============================================================================
// 11. Transformation Pipeline Demo
// ============================================================================

async function transformationPipelineDemo() {
  console.log("\n=== Transformation Pipeline Demo ===");
  interface User {
    id: number;
    name: string;
    email: string;
    age: number;
  }

  // Validation node
  const validator = createProcessor<User, User>((user) => {
    if (!user.email.includes("@")) {
      throw new Error("Invalid email");
    }
    if (user.age < 0 || user.age > 150) {
      throw new Error("Invalid age");
    }
    return user;
  }, "validator");

  // Enrichment node
  const enricher = createProcessor<User, User & { category: string }>(
    (user) => ({
      ...user,
      category: user.age < 18 ? "minor" : user.age < 65 ? "adult" : "senior",
    }),
    "enricher"
  );

  // Privacy filter
  const privacyFilter = new AdaptiveNode<any, any>((data) => data)
    .register(Object, (obj: any) => {
      const filtered = { ...obj };
      if ("email" in filtered) {
        filtered.email = filtered.email.replace(/(.{2}).*(@.*)/, "$1***$2");
      }
      return filtered;
    })
    .setLabel("privacy-filter");

  // Build pipeline
  const graph = new Graph();
  graph.addNode(validator);
  graph.addNode(enricher);
  graph.addNode(privacyFilter);

  graph.connect(validator, enricher);
  graph.connect(enricher, privacyFilter);

  // Process user data
  const userData: User = {
    id: 1,
    name: "John Doe",
    email: "john.doe@example.com",
    age: 25,
  };

  const result = await graph.execute(userData, validator.id);
  console.log(result);
}

// ============================================================================
// 12. Type Adaptive Demo
// ============================================================================

async function typeAdaptiveDemo() {
  console.log("\n=== Type Adaptive Demo ===");
  // Node that handles different data types
  const smartProcessor = new AdaptiveNode<any, string>(
    (input) => `Unknown type: ${typeof input}`
  )
    .register(Number, (num) => `Number: ${num.toFixed(2)}`)
    .register(String, (str) => `String: "${str.toUpperCase()}"`)
    .register(Array, (arr) => `Array[${arr.length}]: ${arr.join(", ")}`)
    .register(Date, (date) => `Date: ${date.toISOString()}`)
    .setLabel("smart-processor");

  const logger = createProcessor<string, void>(
    (msg) => console.log(msg),
    "logger"
  );

  const graph = new Graph();
  graph.addNode(smartProcessor);
  graph.addNode(logger);
  graph.connect(smartProcessor, logger);

  // Test with different types
  await smartProcessor.process(42);
  await smartProcessor.process("hello");
  await smartProcessor.process([1, 2, 3]);
  await smartProcessor.process(new Date());
}

// ============================================================================
// Run all demos
// ============================================================================

async function runAllDemos() {
  try {
    await audioProcessingDemo();
    await basicMathDemo();
    await conditionalRoutingDemo();
    dynamicGraphDemo();
    await errorHandlingDemo();
    await machineLearningDemo();
    await multiProtocolDemo();
    await oscillatorDemo();
    await performanceMonitoringDemo();
    await realtimeStreamDemo();
    await transformationPipelineDemo();
    await typeAdaptiveDemo();

    console.log("\n=== All demos completed successfully! ===");
  } catch (error) {
    console.error("Demo error:", error);
  }
}

// Export for use in other modules
export { runAllDemos };

// Run the demos when this file is executed directly
runAllDemos().catch(console.error);
