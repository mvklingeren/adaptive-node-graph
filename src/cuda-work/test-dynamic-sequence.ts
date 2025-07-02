// ============================================================================
// Test Dynamic Sequence Length with Embedding Layer
// ============================================================================

import { MockCudaRuntime } from "./cuda-abstractions.js";
import { CudaGraphCompiler } from "./cuda-graph.js";
import { LanguageModel } from "./llm/language-model.js";

async function testDynamicSequence() {
    console.log("Testing Dynamic Sequence Length...");
    const runtime = new MockCudaRuntime();
    const compiler = new CudaGraphCompiler(runtime);

    // Model parameters
    const vocabSize = 65;
    const embedDim = 384;
    const numHeads = 6;
    const numLayers = 2; // Reduced for faster testing
    const ffnHiddenDim = 4 * embedDim;
    const maxLen = 512; // Maximum sequence length
    const batchSize = 32;

    // Test with different sequence lengths
    const testSequenceLengths = [64, 128, 256, 384];

    for (const seqLen of testSequenceLengths) {
        console.log(`\n--- Testing with sequence length: ${seqLen} ---`);
        
        // Build the Language Model
        const model = new LanguageModel(
            runtime,
            vocabSize,
            embedDim,
            numHeads,
            numLayers,
            ffnHiddenDim,
            maxLen
        );
        await model.initialize();
        model.build();

        // Compile with this specific sequence length
        const inputShapes = new Map([
            ["input", [batchSize, seqLen]],
        ]);

        try {
            const { kernel, parameters, kernelCode, workspaceSize } = await compiler.compile(model, inputShapes);
            console.log(`✅ Compilation successful for seq_len=${seqLen}`);
            console.log(`   - Kernel ID: ${kernel.id}`);
            console.log(`   - Workspace size: ${workspaceSize} bytes`);
            
            // Check if the generated kernel contains the correct sequence length
            const hasCorrectSeqLen = kernelCode.includes(`{${batchSize}, ${seqLen}, ${embedDim}}`);
            if (hasCorrectSeqLen) {
                console.log(`   - ✅ Kernel contains correct tensor shapes for seq_len=${seqLen}`);
            } else {
                console.log(`   - ❌ Kernel does not contain expected tensor shapes`);
            }
        } catch (error) {
            console.log(`❌ Compilation failed for seq_len=${seqLen}: ${(error as Error).message}`);
        }
    }

    // Test with sequence length exceeding maximum (should fail)
    console.log(`\n--- Testing with sequence length exceeding maximum (600 > 512) ---`);
    const model = new LanguageModel(
        runtime,
        vocabSize,
        embedDim,
        numHeads,
        numLayers,
        ffnHiddenDim,
        maxLen
    );
    await model.initialize();
    model.build();

    const invalidInputShapes = new Map([
        ["input", [batchSize, 600]], // Exceeds maxLen of 512
    ]);

    try {
        await compiler.compile(model, invalidInputShapes);
        console.log(`❌ Expected validation error but compilation succeeded`);
    } catch (error) {
        console.log(`✅ Validation working: ${(error as Error).message}`);
    }

    console.log("\nDynamic sequence length test completed!");
}

testDynamicSequence().catch(console.error);
