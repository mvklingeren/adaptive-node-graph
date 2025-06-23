import { Graph, createProcessor } from '../core';

const graph = new Graph();

// Temperature monitoring system
const tempSensor = createProcessor(() => Math.random() * 50, 'temp-sensor');
const threshold = createProcessor<number, boolean>(temp => temp > 30, 'threshold');
const alert = createProcessor<number, void>(temp => console.log(`🔥 HIGH TEMP: ${temp}°C`), 'alert');
const normal = createProcessor<number, void>(temp => console.log(`✓ Normal: ${temp}°C`), 'normal');

// Create conditional routing
const router = createProcessor<[boolean, number], void>(([isHigh, temp]) => {
  if (isHigh) {
    alert.process(temp);
  } else {
    normal.process(temp);
  }
}, 'router');

graph
  .addNode(tempSensor)
  .addNode(threshold)
  .addNode(router)
  .addNode(alert)
  .addNode(normal);

// Connect: sensor -> threshold check -> router
const sensorOutput = await tempSensor.process(null);
if (sensorOutput !== null) {
  const isHigh = await threshold.process(sensorOutput);
  if (isHigh !== null) {
    await router.process([isHigh, sensorOutput]);
  }
}
