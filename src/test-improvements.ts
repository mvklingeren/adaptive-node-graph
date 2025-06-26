// test-improvements.ts
// Demonstrates the new features in core.ts

import {
  AdaptiveNode,
  Graph,
  TestNode,
  createDelayNode,
  createThrottleNode,
  createErrorLoggerNode,
  createErrorRecoveryNode,
  createLoadBalancerNode,
  createCacheNode,
  SubGraphNode,
} from "./core";

// ============================================================================
// 1. Error Handling Demo
// ============================================================================

console.log('=== Error Handling Demo ===');

async function errorHandlingDemo() {
  const graph = new Graph();
  
  // Create a node that sometimes fails
  const unreliableNode = new AdaptiveNode<number, number>((input) => {
    if (input > 5) {
      throw new Error(`Value ${input} is too high!`);
    }
    return input * 2;
  }).setName('unreliable');
  
  // Error logger
  const errorLogger = createErrorLoggerNode();
  
  // Error recovery with default value
  const errorRecovery = createErrorRecoveryNode(0);
  
  // Add nodes to graph
  graph.addNode(unreliableNode);
  graph.addNode(errorLogger);
  graph.addNode(errorRecovery);
  
  // Connect error handling
  graph.connectError(unreliableNode, errorLogger);
  graph.connect(errorLogger, errorRecovery);
  
  // Test with valid input
  console.log('Processing 3:', await unreliableNode.process(3)); // Should work: 6
  
  // Test with invalid input
  console.log('Processing 10:', await unreliableNode.process(10)); // Should fail and recover
}

// ============================================================================
// 2. Type Safety Demo
// ============================================================================

console.log('\n=== Type Safety Demo ===');

function typeSafetyDemo() {
  // These connections are type-safe at compile time
  const numberNode = new AdaptiveNode<number, string>((n) => n.toString());
  const stringNode = new AdaptiveNode<string, boolean>((s) => s.length > 0);
  
  const graph = new Graph();
  graph.addNode(numberNode);
  graph.addNode(stringNode);
  
  // This connection is type-safe - string output connects to string input
  graph.connect(numberNode, stringNode);
  
  // This would cause a TypeScript error if uncommented:
  // graph.connect(stringNode, numberNode); // Error: boolean doesn't match number
  
  console.log('Type-safe connections established');
}

// ============================================================================
// 3. Async Flow Control Demo
// ============================================================================

console.log('\n=== Async Flow Control Demo ===');

async function flowControlDemo() {
  // Create a node with limited concurrency
  const slowNode = new AdaptiveNode<number, number>(
    async (input) => {
      console.log(`Processing ${input}...`);
      await new Promise(resolve => setTimeout(resolve, 100));
      return input * 2;
    },
    { maxConcurrent: 2 } // Only 2 concurrent operations
  ).setName('slowNode');
  
  // Fire multiple requests
  const promises = [];
  for (let i = 0; i < 5; i++) {
    promises.push(slowNode.process(i));
  }
  
  console.log('Started 5 operations with max concurrency of 2');
  const results = await Promise.all(promises);
  console.log('Results:', results);
}

// ============================================================================
// 4. Time-based Operators Demo
// ============================================================================

console.log('\n=== Time-based Operators Demo ===');

async function timeBasedDemo() {
  const graph = new Graph();
  
  // Delay node
  const delayNode = createDelayNode(500);
  
  // Throttle node (max 1 per second)
  const throttleNode = createThrottleNode(1000);
  
  // Test delay
  console.log('Testing delay node...');
  const start = Date.now();
  await delayNode.process('delayed');
  console.log(`Delayed by ${Date.now() - start}ms`);
  
  // Test throttle
  console.log('\nTesting throttle node...');
  for (let i = 0; i < 5; i++) {
    const result = await throttleNode.process(i);
    console.log(`Throttle result ${i}:`, result);
    await new Promise(resolve => setTimeout(resolve, 300));
  }
}

// ============================================================================
// 5. Testing Utilities Demo
// ============================================================================

console.log('\n=== Testing Utilities Demo ===');

async function testingDemo() {
  // Create test nodes
  const testNode1 = new TestNode<number>();
  const testNode2 = new TestNode<number>((n) => n * 2);
  
  const graph = new Graph();
  graph.addNode(testNode1);
  graph.addNode(testNode2);
  graph.connect(testNode1, testNode2);
  
  // Process some data
  await testNode1.process(5);
  await testNode1.process(10);
  
  // Assert results
  try {
    testNode1.assertReceived([5, 10]);
    testNode2.assertProcessed([10, 20]);
    testNode1.assertNoErrors();
    console.log('All tests passed!');
  } catch (error) {
    console.error('Test failed:', error);
  }
}

// ============================================================================
// 6. Circuit Breaker Demo
// ============================================================================

console.log('\n=== Circuit Breaker Demo ===');

async function circuitBreakerDemo() {
  // Create a node that always fails
  const failingNode = new AdaptiveNode<number, number>(
    () => {
      throw new Error('Always fails');
    },
    { 
      circuitBreakerThreshold: 3,
      circuitBreakerResetTime: 2000 
    }
  ).setName('failing');
  
  const errorLogger = createErrorLoggerNode();
  const graph = new Graph();
  graph.addNode(failingNode);
  graph.addNode(errorLogger);
  graph.connectError(failingNode, errorLogger);
  
  // Try to process multiple times
  console.log('Attempting to trigger circuit breaker...');
  for (let i = 0; i < 5; i++) {
    await failingNode.process(i);
    console.log(`Attempt ${i + 1} completed`);
  }
  
  console.log('Circuit breaker should be open now');
  
  // Wait for reset
  console.log('Waiting 2 seconds for circuit breaker reset...');
  await new Promise(resolve => setTimeout(resolve, 2100));
  
  // Try again
  console.log('Trying again after reset...');
  await failingNode.process(99);
}

// ============================================================================
// 7. Load Balancer Demo
// ============================================================================

console.log('\n=== Load Balancer Demo ===');

async function loadBalancerDemo() {
  // Create worker nodes
  const workers = [
    new AdaptiveNode<number, string>((n) => `Worker1: ${n}`).setName('worker1'),
    new AdaptiveNode<number, string>((n) => `Worker2: ${n}`).setName('worker2'),
    new AdaptiveNode<number, string>((n) => `Worker3: ${n}`).setName('worker3')
  ];
  
  // Create load balancer
  const loadBalancer = createLoadBalancerNode(workers, 'round-robin');
  
  // Process multiple requests
  console.log('Round-robin load balancing:');
  for (let i = 0; i < 6; i++) {
    const result = await loadBalancer.process(i);
    console.log(result);
  }
}

// ============================================================================
// 8. Cache Node Demo
// ============================================================================

console.log('\n=== Cache Node Demo ===');

async function cacheDemo() {
  let computeCount = 0;
  
  // Create expensive computation
  const expensiveCompute = async (n: number) => {
    computeCount++;
    console.log(`Computing ${n}... (computation #${computeCount})`);
    await new Promise(resolve => setTimeout(resolve, 100));
    return n * n;
  };
  
  // Create cached version
  const cachedNode = createCacheNode(expensiveCompute, 1000, 10);
  
  // First calls - will compute
  console.log('First calls:');
  console.log(await cachedNode.process(5));
  console.log(await cachedNode.process(10));
  
  // Repeated calls - will use cache
  console.log('\nRepeated calls (cached):');
  console.log(await cachedNode.process(5));
  console.log(await cachedNode.process(10));
  
  // Wait for cache expiry
  console.log('\nWaiting for cache expiry...');
  await new Promise(resolve => setTimeout(resolve, 1100));
  
  // Call again - will recompute
  console.log('\nAfter cache expiry:');
  console.log(await cachedNode.process(5));
}

// ============================================================================
// 9. Sub-graph Demo
// ============================================================================

console.log('\n=== Sub-graph Demo ===');

async function subGraphDemo() {
  // Create a sub-graph that adds numbers
  const subGraph = new Graph();
  const add1 = new AdaptiveNode<number, number>((n) => n + 1).setName('add1');
  const add2 = new AdaptiveNode<number, number>((n) => n + 2).setName('add2');
  
  subGraph.addNode(add1);
  subGraph.addNode(add2);
  subGraph.connect(add1, add2);
  
  // Create sub-graph node
  const subGraphNode = new SubGraphNode(subGraph, add1.id, add2.id);
  
  // Use it
  const result = await subGraphNode.process(10);
  console.log('Sub-graph result:', result); // Should be 13 (10 + 1 + 2)
}

// ============================================================================
// Run all demos
// ============================================================================

async function runAllDemos() {
  try {
    await errorHandlingDemo();
    typeSafetyDemo();
    await flowControlDemo();
    await timeBasedDemo();
    await testingDemo();
    await circuitBreakerDemo();
    await loadBalancerDemo();
    await cacheDemo();
    await subGraphDemo();
    
    console.log('\n=== All demos completed successfully! ===');
  } catch (error) {
    console.error('Demo error:', error);
  }
}

// Export for use in other modules
export { runAllDemos };

// Run the demos when this file is executed directly
runAllDemos().catch(console.error);
