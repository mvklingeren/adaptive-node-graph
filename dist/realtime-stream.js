import {
  AdaptiveNode,
  Graph,
  createLoadBalancerNode,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/realtime-stream.ts
var StreamProcessor = class extends AdaptiveNode {
  buffer = [];
  windowSize = 100;
  constructor() {
    super((event) => event);
    this.register(MouseEvent, this.processMouseEvent.bind(this));
    this.register(KeyboardEvent, this.processKeyboardEvent.bind(this));
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
};
var workers = Array.from(
  { length: 4 },
  (_, i) => new StreamProcessor().setLabel(`worker-${i}`)
);
var loadBalancer = createLoadBalancerNode(workers);
var aggregator = createProcessor((result) => {
  console.log(`Processed event:`, result);
}, "aggregator");
var graph = new Graph();
graph.addNode(loadBalancer);
workers.forEach((w) => graph.addNode(w));
graph.addNode(aggregator);
workers.forEach((worker) => {
  graph.connect(loadBalancer, worker);
  graph.connect(worker, aggregator);
});
var events = [
  new MouseEvent("click", { clientX: 100, clientY: 200 }),
  { value: 42 },
  new KeyboardEvent("keydown", { key: "Enter" }),
  { value: 38 },
  { value: 45 }
];
for (const event of events) {
  await loadBalancer.process(event);
}
