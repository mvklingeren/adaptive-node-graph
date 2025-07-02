# Shakespeare LLM - Usage Guide

This guide explains how to pass text input to the compiled Shakespeare LLM binary.

## Compilation

First, compile the project:

```bash
make clean
make
```

This creates the `run_model` executable.

## Input Methods

### 1. Default Mode (Training Data)
Run without arguments to use the Shakespeare dataset:
```bash
./run_model
```

### 2. Custom Text Input
Pass any text as a command line argument:
```bash
./run_model "To be or not to be"
./run_model "Hello world"
./run_model "The quick brown fox"
```

### 3. Text Generation Mode
Add `--generate` flag to generate new text from your prompt:
```bash
./run_model "To be or not to be" --generate
./run_model "Once upon a time" --generate
./run_model "Romeo:" --generate
```

### 4. Longer Text from File
For longer input text, create a file and use shell redirection:
```bash
echo "Your very long text here..." > input.txt
./run_model "$(cat input.txt)"
```

## Output Explanation

### Without `--generate` flag:
- Shows raw logits (probability scores) for each position
- Shows predicted next characters for your input text
- Useful for debugging and understanding model predictions

### With `--generate` flag:
- Generates new text character by character
- Shows the generation process in real-time
- Produces creative continuations of your prompt

## Example Sessions

### Basic Prediction:
```bash
$ ./run_model "Romeo"
Using command line input: "Romeo"
Vocabulary size: 65 unique characters.
Input tokens: 48 42 40 28 42 
...
--- Predicted Next Characters ---
After 'R' -> predicted: 'o' (logit: 2.34)
After 'o' -> predicted: 'm' (logit: 1.89)
```

### Text Generation:
```bash
$ ./run_model "Romeo:" --generate
Using command line input: "Romeo:"
=== INTERACTIVE TEXT GENERATION ===
Starting with prompt: "Romeo:"
Generated text:  My dear Juliet, thy beauty doth shine brighter than the stars...
```

## Technical Notes

- **Character-level tokenization**: Each character becomes one token
- **Vocabulary**: Built from the Shakespeare dataset (65 unique characters)
- **Context window**: 128 characters maximum
- **Batch processing**: Input is replicated across 32 batch items
- **GPU memory**: Uses ~2GB workspace by default

## Troubleshooting

### "Could not open shakespeare.txt"
Make sure you run from the project root directory where `src/cuda-work/shakespeare.txt` exists.

### Out of memory errors
Reduce workspace size during compilation:
```bash
make clean
make WORKSPACE_SIZE_MB=512
```

### Invalid characters
The model only knows characters from the Shakespeare dataset. Unknown characters will be ignored or cause errors. 