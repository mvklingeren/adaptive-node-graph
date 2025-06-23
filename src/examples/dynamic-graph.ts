// Add nodes dynamically based on conditions
if (needsCaching) {
    const cache = createCacheNode(5000);
    graph.addNode(cache);
    graph.connect(dataSource, cache);
    graph.connect(cache, processor);
  } else {
    graph.connect(dataSource, processor);
  }