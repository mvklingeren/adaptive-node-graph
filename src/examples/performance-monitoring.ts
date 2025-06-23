import { AdaptiveNode } from "../core";

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