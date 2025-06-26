import {
  AdaptiveNode,
  Graph,
  OscillatorNode,
  createFloat32MultiplyNode,
  createLoadBalancerNode,
  createProcessor
} from "./chunks/chunk-PSE6PIOY.js";

// src/test-basics.ts
async function audioProcessingDemo() {
  console.log("=== Audio Processing Demo ===");
  const envelope = createProcessor((time) => {
    if (time < 0.1) return time / 0.1;
    if (time < 0.2) return 1;
    if (time < 0.8) return 0.7;
    return 0.7 * (1 - (time - 0.8) / 0.2);
  }, "envelope");
  const filter = new AdaptiveNode((samples2) => {
    const filtered2 = new Float32Array(samples2.length);
    filtered2[0] = samples2[0];
    for (let i = 1; i < samples2.length; i++) {
      filtered2[i] = filtered2[i - 1] * 0.9 + samples2[i] * 0.1;
    }
    return filtered2;
  }).setLabel("lowpass");
  const graph = new Graph();
  const osc = new OscillatorNode();
  const gain = createProcessor(
    ([samples2, gainValue]) => samples2.map((s) => s * gainValue),
    "gain"
  );
  graph.addNode(osc);
  graph.addNode(envelope);
  graph.addNode(filter);
  graph.addNode(gain);
  const params = {
    frequency: 440,
    amplitude: 0.5,
    sampleRate: 44100,
    length: 44100,
    waveform: "sawtooth"
  };
  const samples = await osc.process(params);
  const filtered = await filter.process(samples);
  const envelopeValues = await Promise.all(
    Array.from({ length: 44100 }, (_, i) => envelope.process(i / 44100))
  );
  const output = filtered?.map((sample, i) => sample * envelopeValues[i]);
  console.log("Audio processing demo complete.");
}
async function basicMathDemo() {
  console.log("\n=== Basic Math Demo ===");
  const graph = new Graph();
  const input = createProcessor((x) => x, "input");
  const add5 = createProcessor((x) => x + 5, "add5");
  const multiply2 = createProcessor((x) => x * 2, "multiply2");
  const output = createProcessor(
    (x) => console.log("Result:", x),
    "output"
  );
  graph.addNode(input).addNode(add5).addNode(multiply2).addNode(output);
  graph.connect(input, add5);
  graph.connect(add5, multiply2);
  graph.connect(multiply2, output);
  await graph.execute(10);
}
async function conditionalRoutingDemo() {
  console.log("\n=== Conditional Routing Demo ===");
  const graph = new Graph();
  const tempSensor = createProcessor(() => Math.random() * 50, "temp-sensor");
  const threshold = createProcessor(
    (temp) => temp > 30,
    "threshold"
  );
  const alert = createProcessor(
    (temp) => console.log(`\u{1F525} HIGH TEMP: ${temp}\xB0C`),
    "alert"
  );
  const normal = createProcessor(
    (temp) => console.log(`\u2713 Normal: ${temp}\xB0C`),
    "normal"
  );
  const router = createProcessor(([isHigh, temp]) => {
    if (isHigh) {
      alert.process(temp);
    } else {
      normal.process(temp);
    }
  }, "router");
  graph.addNode(tempSensor).addNode(threshold).addNode(router).addNode(alert).addNode(normal);
  const sensorOutput = await tempSensor.process(null);
  if (sensorOutput !== null) {
    const isHigh = await threshold.process(sensorOutput);
    if (isHigh !== null) {
      await router.process([isHigh, sensorOutput]);
    }
  }
}
function dynamicGraphDemo() {
  console.log("\n=== Dynamic Graph Demo ===");
  console.log("Dynamic graph demo is conceptual and commented out.");
}
async function errorHandlingDemo() {
  console.log("\n=== Error Handling Demo ===");
  const processData = (data) => {
    if (typeof data === "number") {
      return data * 2;
    } else if (typeof data === "string") {
      return data.toUpperCase();
    } else if (Array.isArray(data)) {
      return data.map(processData);
    } else if (data === null || data === void 0) {
      throw new Error("Cannot process null or undefined data");
    } else if (typeof data === "object") {
      return Object.fromEntries(
        Object.entries(data).map(([key, value]) => [
          key,
          processData(value)
        ])
      );
    }
    throw new Error(`Unsupported data type: ${typeof data}`);
  };
  const safeProcessor = createProcessor(
    (input) => {
      try {
        return processData(input);
      } catch (error) {
        console.error("Processing failed:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);
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
async function machineLearningDemo() {
  console.log("\n=== Machine Learning Demo ===");
  class DataPreprocessor extends AdaptiveNode {
    constructor() {
      super((data) => new Float32Array(data));
      this.register(Array, this.preprocessArray.bind(this));
      this.register(Float32Array, this.preprocessFloat32.bind(this));
    }
    preprocessArray(data) {
      const arr = new Float32Array(data);
      const mean = arr.reduce((a, b) => a + b) / arr.length;
      const std = Math.sqrt(
        arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length
      );
      return arr.map((val) => (val - mean) / std);
    }
    preprocessFloat32(data) {
      return this.preprocessArray(Array.from(data));
    }
  }
  const featureExtractor = createProcessor(
    (data) => {
      const features = new Float32Array(5);
      features[0] = Math.min(...data);
      features[1] = Math.max(...data);
      features[2] = data.reduce((a, b) => a + b) / data.length;
      features[3] = Math.sqrt(
        data.reduce((sum, val) => sum + val * val, 0) / data.length
      );
      features[4] = data.reduce(
        (sum, val, i) => i > 0 ? sum + Math.abs(val - data[i - 1]) : sum,
        0
      );
      return features;
    },
    "feature-extractor"
  );
  const model = createProcessor((features) => {
    const sum = features.reduce((a, b) => a + b);
    if (sum > 10) {
      return { class: "positive", confidence: 0.85 };
    } else if (sum < -10) {
      return { class: "negative", confidence: 0.9 };
    } else {
      return { class: "neutral", confidence: 0.75 };
    }
  }, "ml-model");
  const postProcessor = new AdaptiveNode((prediction) => prediction).register(Object, (pred) => {
    if (pred.confidence < 0.5) {
      return { ...pred, class: "uncertain" };
    }
    return pred;
  }).setLabel("post-processor");
  const graph = new Graph();
  const preprocessor = new DataPreprocessor().setLabel("preprocessor");
  graph.addNode(preprocessor);
  graph.addNode(featureExtractor);
  graph.addNode(model);
  graph.addNode(postProcessor);
  graph.connect(preprocessor, featureExtractor);
  graph.connect(featureExtractor, model);
  graph.connect(model, postProcessor);
  const testData = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    new Float32Array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]),
    Array.from({ length: 100 }, () => Math.random() * 10 - 5)
  ];
  for (const data of testData) {
    const result = await graph.execute(data, preprocessor.id);
    console.log("Prediction:", result);
  }
}
async function multiProtocolDemo() {
  console.log("\n=== Multi-protocol Demo ===");
  class APIGateway extends AdaptiveNode {
    rateLimits = /* @__PURE__ */ new Map();
    cache = /* @__PURE__ */ new Map();
    constructor() {
      super((request) => ({ error: "Unknown protocol" }));
      this.register(Object, this.routeByShape.bind(this));
    }
    routeByShape(request) {
      if ("method" in request && "path" in request) {
        return this.handleHTTP(request);
      }
      if (request.type === "ws" && "action" in request) {
        return this.handleWebSocket(request);
      }
      if ("service" in request && "method" in request) {
        return this.handleGRPC(request);
      }
      return { error: "Unrecognized request format" };
    }
    handleHTTP(request) {
      const clientId = request.headers["x-client-id"] || "anonymous";
      if (!this.checkRateLimit(clientId)) {
        return { status: 429, error: "Rate limit exceeded" };
      }
      const cacheKey = `${request.method}:${request.path}`;
      const cached = this.cache.get(cacheKey);
      if (cached && cached.expires > Date.now()) {
        return { status: 200, data: cached.data, cached: true };
      }
      const response = this.routeHTTPRequest(request);
      if (request.method === "GET" && response.status === 200) {
        this.cache.set(cacheKey, {
          data: response.data,
          expires: Date.now() + 6e4
          // 1 minute
        });
      }
      return response;
    }
    handleWebSocket(message) {
      switch (message.action) {
        case "subscribe":
          return { type: "subscription", channel: message.payload.channel };
        case "publish":
          return { type: "broadcast", sent: true };
        default:
          return { type: "error", message: "Unknown action" };
      }
    }
    handleGRPC(request) {
      return {
        service: request.service,
        method: request.method,
        response: `Processed ${request.service}.${request.method}`
      };
    }
    checkRateLimit(clientId) {
      const now = Date.now();
      const window = 6e4;
      const limit = 100;
      if (!this.rateLimits.has(clientId)) {
        this.rateLimits.set(clientId, []);
      }
      const requests2 = this.rateLimits.get(clientId);
      const recentRequests = requests2.filter((time) => now - time < window);
      if (recentRequests.length >= limit) {
        return false;
      }
      recentRequests.push(now);
      this.rateLimits.set(clientId, recentRequests);
      return true;
    }
    routeHTTPRequest(request) {
      if (request.path.startsWith("/api/users")) {
        return { status: 200, data: { users: [] } };
      }
      if (request.path.startsWith("/api/products")) {
        return { status: 200, data: { products: [] } };
      }
      return { status: 404, error: "Not found" };
    }
  }
  const authService = createProcessor((request) => {
    return { ...request, authenticated: true };
  }, "auth-service");
  const loggingService = createProcessor((request) => {
    console.log(`[${(/* @__PURE__ */ new Date()).toISOString()}] Request:`, request);
    return request;
  }, "logging-service");
  const analyticsService = new AdaptiveNode((event) => {
    return event;
  }).register(Object, (event) => {
    if (event.status >= 400) {
      console.log("Error event:", event);
    }
    return event;
  }).setLabel("analytics");
  const graph = new Graph();
  const gateway = new APIGateway().setLabel("api-gateway");
  graph.addNode(loggingService);
  graph.addNode(authService);
  graph.addNode(gateway);
  graph.addNode(analyticsService);
  graph.connect(loggingService, authService);
  graph.connect(authService, gateway);
  graph.connect(gateway, analyticsService);
  const requests = [
    // HTTP Request
    {
      method: "GET",
      path: "/api/users/123",
      headers: { "x-client-id": "client-1" }
    },
    // WebSocket Message
    {
      type: "ws",
      action: "subscribe",
      payload: { channel: "updates" }
    },
    // gRPC Request
    {
      service: "UserService",
      method: "GetUser",
      data: { userId: 123 }
    }
  ];
  for (const request of requests) {
    const result = await graph.execute(request, loggingService.id);
    console.log("Result:", result);
  }
}
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
    sampleRate: 44100
  });
  console.log("Oscillator result:", result);
}
async function performanceMonitoringDemo() {
  console.log("\n=== Performance Monitoring Demo ===");
  function expensiveOperation(input) {
    const iterations = input?.iterations || 1e6;
    let result = 0;
    for (let i = 0; i < iterations; i++) {
      result += Math.sin(i) * Math.cos(i);
    }
    return result;
  }
  const monitoredNode = new AdaptiveNode((input) => {
    const start = performance.now();
    const result = expensiveOperation(input);
    const duration = performance.now() - start;
    if (duration > 100) {
      console.warn(`Slow operation: ${duration}ms`);
    }
    return result;
  }).setLabel("monitored");
  await monitoredNode.process({ iterations: 2e6 });
  const stats = monitoredNode.getPerformanceStats();
  console.log("Performance stats:", stats);
}
async function realtimeStreamDemo() {
  console.log("\n=== Real-time Stream Demo ===");
  class StreamProcessor extends AdaptiveNode {
    buffer = [];
    windowSize = 100;
    constructor() {
      super((event) => event);
      if (typeof MouseEvent !== "undefined") {
        this.register(MouseEvent, this.processMouseEvent.bind(this));
      }
      if (typeof KeyboardEvent !== "undefined") {
        this.register(KeyboardEvent, this.processKeyboardEvent.bind(this));
      }
      this.register(Object, this.processDataEvent.bind(this));
    }
    processMouseEvent(event) {
      return {
        type: "mouse",
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
      };
    }
    processKeyboardEvent(event) {
      return {
        type: "keyboard",
        key: event.key,
        timestamp: Date.now()
      };
    }
    processDataEvent(event) {
      this.buffer.push(event);
      if (this.buffer.length > this.windowSize) {
        this.buffer.shift();
      }
      return {
        type: "data",
        count: this.buffer.length,
        average: this.computeAverage(),
        timestamp: Date.now()
      };
    }
    computeAverage() {
      if (this.buffer.length === 0) return 0;
      const sum = this.buffer.reduce((acc, val) => acc + (val.value || 0), 0);
      return sum / this.buffer.length;
    }
  }
  const workers = Array.from(
    { length: 4 },
    (_, i) => new StreamProcessor().setLabel(`worker-${i}`)
  );
  const loadBalancer = createLoadBalancerNode(workers);
  const aggregator = createProcessor((result) => {
    console.log(`Processed event:`, result);
  }, "aggregator");
  const graph = new Graph();
  graph.addNode(loadBalancer);
  workers.forEach((w) => graph.addNode(w));
  graph.addNode(aggregator);
  workers.forEach((worker) => {
    graph.connect(loadBalancer, worker);
    graph.connect(worker, aggregator);
  });
  const events = [];
  if (typeof MouseEvent !== "undefined") {
    events.push(new MouseEvent("click", { clientX: 100, clientY: 200 }));
  }
  events.push({ value: 42 });
  if (typeof KeyboardEvent !== "undefined") {
    events.push(new KeyboardEvent("keydown", { key: "Enter" }));
  }
  events.push({ value: 38 }, { value: 45 });
  for (const event of events) {
    await loadBalancer.process(event);
  }
}
async function transformationPipelineDemo() {
  console.log("\n=== Transformation Pipeline Demo ===");
  const validator = createProcessor((user) => {
    if (!user.email.includes("@")) {
      throw new Error("Invalid email");
    }
    if (user.age < 0 || user.age > 150) {
      throw new Error("Invalid age");
    }
    return user;
  }, "validator");
  const enricher = createProcessor(
    (user) => ({
      ...user,
      category: user.age < 18 ? "minor" : user.age < 65 ? "adult" : "senior"
    }),
    "enricher"
  );
  const privacyFilter = new AdaptiveNode((data) => data).register(Object, (obj) => {
    const filtered = { ...obj };
    if ("email" in filtered) {
      filtered.email = filtered.email.replace(/(.{2}).*(@.*)/, "$1***$2");
    }
    return filtered;
  }).setLabel("privacy-filter");
  const graph = new Graph();
  graph.addNode(validator);
  graph.addNode(enricher);
  graph.addNode(privacyFilter);
  graph.connect(validator, enricher);
  graph.connect(enricher, privacyFilter);
  const userData = {
    id: 1,
    name: "John Doe",
    email: "john.doe@example.com",
    age: 25
  };
  const result = await graph.execute(userData, validator.id);
  console.log(result);
}
async function typeAdaptiveDemo() {
  console.log("\n=== Type Adaptive Demo ===");
  const smartProcessor = new AdaptiveNode(
    (input) => `Unknown type: ${typeof input}`
  ).register(Number, (num) => `Number: ${num.toFixed(2)}`).register(String, (str) => `String: "${str.toUpperCase()}"`).register(Array, (arr) => `Array[${arr.length}]: ${arr.join(", ")}`).register(Date, (date) => `Date: ${date.toISOString()}`).setLabel("smart-processor");
  const logger = createProcessor(
    (msg) => console.log(msg),
    "logger"
  );
  const graph = new Graph();
  graph.addNode(smartProcessor);
  graph.addNode(logger);
  graph.connect(smartProcessor, logger);
  await smartProcessor.process(42);
  await smartProcessor.process("hello");
  await smartProcessor.process([1, 2, 3]);
  await smartProcessor.process(/* @__PURE__ */ new Date());
}
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
runAllDemos().catch(console.error);
export {
  runAllDemos
};
