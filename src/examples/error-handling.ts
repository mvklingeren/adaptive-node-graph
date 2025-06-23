import { createProcessor } from "../core";

// todo: should processData be a class method?
// Define a recursive type for processable data
type Processable = string | number | Processable[] | { [key: string]: Processable };

// Define the processData function with proper return type
const processData = (data: Processable | null | undefined): Processable => {
  // Example processing logic
  if (typeof data === "number") {
    return data * 2;
  } else if (typeof data === "string") {
    return data.toUpperCase();
  } else if (Array.isArray(data)) {
    // Recursively process each item in the array
    return data.map(processData);
  } else if (data === null || data === undefined) {
    throw new Error("Cannot process null or undefined data");
  } else if (typeof data === "object") {
    // Recursively process each value in the object
    return Object.fromEntries(
      Object.entries(data).map(([key, value]) => [
        key,
        processData(value as Processable),
      ])
    );
  }
  // Handle unsupported data types
  throw new Error(`Unsupported data type: ${typeof data}`);
};

// Define the result type, which can be processed data or an error object
type ProcessResult = Processable | { error: string; input: any };

const safeProcessor = createProcessor<Processable, ProcessResult>((input) => {
    try {
      // Processing logic
      return processData(input);
    } catch (error: unknown) {
      console.error('Processing failed:', error);
      // Handle the unknown error type
      const errorMessage = error instanceof Error ? error.message : String(error);
      return { error: errorMessage, input };
    }
  }, 'safe-processor');
