import {
  Graph,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/conditional-routing.ts
var graph = new Graph();
var tempSensor = createProcessor(() => Math.random() * 50, "temp-sensor");
var threshold = createProcessor((temp) => temp > 30, "threshold");
var alert = createProcessor((temp) => console.log(`\u{1F525} HIGH TEMP: ${temp}\xB0C`), "alert");
var normal = createProcessor((temp) => console.log(`\u2713 Normal: ${temp}\xB0C`), "normal");
var router = createProcessor(([isHigh, temp]) => {
  if (isHigh) {
    alert.process(temp);
  } else {
    normal.process(temp);
  }
}, "router");
graph.addNode(tempSensor).addNode(threshold).addNode(router).addNode(alert).addNode(normal);
var sensorOutput = await tempSensor.process(null);
if (sensorOutput !== null) {
  const isHigh = await threshold.process(sensorOutput);
  if (isHigh !== null) {
    await router.process([isHigh, sensorOutput]);
  }
}
