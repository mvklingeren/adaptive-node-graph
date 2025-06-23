import {
  AdaptiveNode,
  Graph,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/multi-protocol.ts
var APIGateway = class extends AdaptiveNode {
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
};
var authService = createProcessor((request) => {
  return { ...request, authenticated: true };
}, "auth-service");
var loggingService = createProcessor((request) => {
  console.log(`[${(/* @__PURE__ */ new Date()).toISOString()}] Request:`, request);
  return request;
}, "logging-service");
var analyticsService = new AdaptiveNode((event) => {
  return event;
}).register(Object, (event) => {
  if (event.status >= 400) {
    console.log("Error event:", event);
  }
  return event;
}).setLabel("analytics");
var graph = new Graph();
var gateway = new APIGateway().setLabel("api-gateway");
graph.addNode(loggingService);
graph.addNode(authService);
graph.addNode(gateway);
graph.addNode(analyticsService);
graph.connect(loggingService, authService);
graph.connect(authService, gateway);
graph.connect(gateway, analyticsService);
var requests = [
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
