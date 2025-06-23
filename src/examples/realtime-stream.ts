import {
  Graph,
  AdaptiveNode,
  createProcessor,
  createLoadBalancerNode,
} from "../core";

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
