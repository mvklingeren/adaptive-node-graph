import {
  createProcessor
} from "./chunks/chunk-TRUTJY2J.js";

// src/examples/error-handling.ts
var processData = (data) => {
  if (typeof data === "number") {
    return data * 2;
  } else if (typeof data === "string") {
    return data.toUpperCase();
  } else if (Array.isArray(data)) {
    return data.map(processData);
  } else if (data === null || data === void 0) {
    throw new Error("Cannot process null or undefined data");
  } else if (typeof data === "object") {
    return Object.fromEntries(
      Object.entries(data).map(([key, value]) => [
        key,
        processData(value)
      ])
    );
  }
  throw new Error(`Unsupported data type: ${typeof data}`);
};
var safeProcessor = createProcessor((input) => {
  try {
    return processData(input);
  } catch (error) {
    console.error("Processing failed:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    return { error: errorMessage, input };
  }
}, "safe-processor");
