import {
  AdaptiveNode,
  Graph,
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/transformation-pipeline.ts
var validator = createProcessor((user) => {
  if (!user.email.includes("@")) {
    throw new Error("Invalid email");
  }
  if (user.age < 0 || user.age > 150) {
    throw new Error("Invalid age");
  }
  return user;
}, "validator");
var enricher = createProcessor((user) => ({
  ...user,
  category: user.age < 18 ? "minor" : user.age < 65 ? "adult" : "senior"
}), "enricher");
var privacyFilter = new AdaptiveNode((data) => data).register(Object, (obj) => {
  const filtered = { ...obj };
  if ("email" in filtered) {
    filtered.email = filtered.email.replace(/(.{2}).*(@.*)/, "$1***$2");
  }
  return filtered;
}).setLabel("privacy-filter");
var graph = new Graph();
graph.addNode(validator);
graph.addNode(enricher);
graph.addNode(privacyFilter);
graph.connect(validator, enricher);
graph.connect(enricher, privacyFilter);
var userData = {
  id: 1,
  name: "John Doe",
  email: "john.doe@example.com",
  age: 25
};
var result = await graph.execute(userData, validator.id);
console.log(result);
