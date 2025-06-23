import { AdaptiveNode } from "../core";

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
  }).setLabel('monitored');
  
  // Check performance stats
  const stats = monitoredNode.getPerformanceStats();
  console.log('Performance stats:', stats);
