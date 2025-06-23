import { Graph, AdaptiveNode, createProcessor, createRouterNode } from './universal-adaptive-router';

// Request/Response types
interface HTTPRequest {
  method: string;
  path: string;
  headers: Record<string, string>;
  body?: any;
}

interface WebSocketMessage {
  type: 'ws';
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
    super((request) => ({ error: 'Unknown protocol' }));
    
    this.register(Object, this.routeByShape.bind(this));
  }
  
  private routeByShape(request: any) {
    // HTTP Request
    if ('method' in request && 'path' in request) {
      return this.handleHTTP(request as HTTPRequest);
    }
    
    // WebSocket Message
    if (request.type === 'ws' && 'action' in request) {
      return this.handleWebSocket(request as WebSocketMessage);
    }
    
    // gRPC Request
    if ('service' in request && 'method' in request) {
      return this.handleGRPC(request as GRPCRequest);
    }
    
    return { error: 'Unrecognized request format' };
  }
  
  private handleHTTP(request: HTTPRequest) {
    // Check rate limits
    const clientId = request.headers['x-client-id'] || 'anonymous';
    if (!this.checkRateLimit(clientId)) {
      return { status: 429, error: 'Rate limit exceeded' };
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
    if (request.method === 'GET' && response.status === 200) {
      this.cache.set(cacheKey, {
        data: response.data,
        expires: Date.now() + 60000 // 1 minute
      });
    }
    
    return response;
  }
  
  private handleWebSocket(message: WebSocketMessage) {
    // Real-time message handling
    switch (message.action) {
      case 'subscribe':
        return { type: 'subscription', channel: message.payload.channel };
      case 'publish':
        return { type: 'broadcast', sent: true };
      default:
        return { type: 'error', message: 'Unknown action' };
    }
  }
  
  private handleGRPC(request: GRPCRequest) {
    // gRPC routing
    return {
      service: request.service,
      method: request.method,
      response: `Processed ${request.service}.${request.method}`
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
    const recentRequests = requests.filter(time => now - time < window);
    
    if (recentRequests.length >= limit) {
      return false;
    }
    
    recentRequests.push(now);
    this.rateLimits.set(clientId, recentRequests);
    return true;
  }
  
  private routeHTTPRequest(request: HTTPRequest) {
    // Simplified routing logic
    if (request.path.startsWith('/api/users')) {
      return { status: 200, data: { users: [] } };
    }
    if (request.path.startsWith('/api/products')) {
      return { status: 200, data: { products: [] } };
    }
    return { status: 404, error: 'Not found' };
  }
}

// Service-specific processors
const authService = createProcessor<any, any>((request) => {
  // Authentication logic
  return { ...request, authenticated: true };
}, 'auth-service');

const loggingService = createProcessor<any, any>((request) => {
  console.log(`[${new Date().toISOString()}] Request:`, request);
  return request;
}, 'logging-service');

const analyticsService = new AdaptiveNode<any, any>((event) => {
  // Track metrics
  return event;
})
  .register(Object, (event: any) => {
    if (event.status >= 400) {
      console.log('Error event:', event);
    }
    return event;
  })
  .setLabel('analytics');

// Build the API gateway graph
const graph = new Graph();
const gateway = new APIGateway().setLabel('api-gateway');

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
    method: 'GET',
    path: '/api/users/123',
    headers: { 'x-client-id': 'client-1' }
  },
  // WebSocket Message
  {
    type: 'ws',
    action: 'subscribe',
    payload: { channel: 'updates' }
  },
  // gRPC Request
  {
    service: 'UserService',
    method: 'GetUser',
    data: { userId: 123 }
  }
];

for (const request of requests) {
  const result = await graph.execute(request, loggingService.id);
  console.log('Result:', result);
}