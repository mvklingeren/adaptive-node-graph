import { createProcessor } from "../core";
// todo: should processData be a class method?
// Define the processData function with proper return type
const processData = (data) => {
    // Example processing logic
    if (typeof data === 'number') {
        return data * 2;
    }
    else if (typeof data === 'string') {
        return data.toUpperCase();
    }
    else if (Array.isArray(data)) {
        return data.map((item) => processData(item));
    }
    else if (data === null || data === undefined) {
        throw new Error('Cannot process null or undefined data');
    }
    else if (typeof data === 'object') {
        return Object.fromEntries(Object.entries(data).map(([key, value]) => [key, processData(value)]));
    }
    throw new Error(`Unsupported data type: ${typeof data}`);
};
const safeProcessor = createProcessor((input) => {
    try {
        // Processing logic
        return processData(input);
    }
    catch (error) {
        console.error('Processing failed:', error);
        // Handle the unknown error type
        const errorMessage = error instanceof Error ? error.message : String(error);
        return { error: errorMessage, input };
    }
}, 'safe-processor');
