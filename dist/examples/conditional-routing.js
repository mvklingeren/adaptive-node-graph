import { Graph, createProcessor } from '../core';
const graph = new Graph();
// Temperature monitoring system
const tempSensor = createProcessor(() => Math.random() * 50, 'temp-sensor');
const threshold = createProcessor(temp => temp > 30, 'threshold');
const alert = createProcessor(temp => console.log(`🔥 HIGH TEMP: ${temp}°C`), 'alert');
const normal = createProcessor(temp => console.log(`✓ Normal: ${temp}°C`), 'normal');
// Create conditional routing
const router = createProcessor(([isHigh, temp]) => {
    if (isHigh) {
        alert.process(temp);
    }
    else {
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
const isHigh = await threshold.process(sensorOutput);
await router.process([isHigh, sensorOutput]);
