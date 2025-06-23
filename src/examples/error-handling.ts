import { createProcessor } from "../core";

const safeProcessor = createProcessor<any, any>((input) => {
    try {
      // Processing logic
      return processData(input);
    } catch (error) {
      console.error('Processing failed:', error);
      return { error: error.message, input };
    }
  }, 'safe-processor');