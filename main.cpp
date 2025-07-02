#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <cuda_runtime.h>

// Forward declaration of the main CUDA execution function from the .cu file
extern "C" void executeGraph(
    void* input_data,
    const int* input_shape,
    int input_dims,
    const char* input_dtype,
    
    void* output_data,
    const int* output_shape,
    int output_dims,
    const char* output_dtype,
    
    void** param_data,
    const int** param_shapes,
    const int* param_dims,
    const char** param_dtypes,
    int param_count,
    
    char* workspace,
    size_t workspace_size
);

// ============================================================================
// Data Preparation
// ============================================================================

class CharacterTokenizer {
public:
    std::map<char, int> charToIdx;
    std::map<int, char> idxToChar;
    int vocabSize = 0;

    CharacterTokenizer(const std::string& text) {
        std::set<char> charSet(text.begin(), text.end());
        vocabSize = charSet.size();
        int i = 0;
        for (char c : charSet) {
            charToIdx[c] = i;
            idxToChar[i] = c;
            i++;
        }
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> encoded;
        for (char c : text) {
            encoded.push_back(charToIdx[c]);
        }
        return encoded;
    }
};

void getBatch(const std::vector<int>& data, int blockSize, int batchSize, std::vector<int>& x, std::vector<int>& y) {
    x.resize(batchSize * blockSize);
    y.resize(batchSize * blockSize);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, data.size() - blockSize - 1);

    for (int i = 0; i < batchSize; i++) {
        int start = dist(rng);
        std::copy(data.begin() + start, data.begin() + start + blockSize, x.begin() + i * blockSize);
        std::copy(data.begin() + start + 1, data.begin() + start + 1 + blockSize, y.begin() + i * blockSize);
    }
}

// ============================================================================
// Helper for Tensor Allocation
// ============================================================================

struct DeviceTensor {
    void* data = nullptr;
    std::vector<int> shape;
    size_t bytes = 0;

    template<typename T>
    void allocate(const std::vector<int>& s) {
        shape = s;
        long long total_elements = 1;
        for (int dim : shape) {
            total_elements *= dim;
        }
        bytes = total_elements * sizeof(T);
        cudaMalloc(&data, bytes);
    }

    void free() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }
};

// ============================================================================
// Text Generation Functions
// ============================================================================

std::string generateText(
    const CharacterTokenizer &tokenizer,
    const std::string &prompt,
    int maxLength,
    void **param_data_ptrs,
    const int **param_shape_ptrs,
    const int *param_dims_array,
    const char **param_dtypes,
    char *workspace,
    size_t workspaceSize)
{
    std::vector<int> tokens = tokenizer.encode(prompt);
    std::string result = prompt;

    const int batchSize = 1; // Single sequence generation
    const int blockSize = 128;
    const int vocabSize = tokenizer.vocabSize;

    // Allocate memory for generation
    DeviceTensor d_input, d_output;
    d_input.allocate<int>({batchSize, blockSize});
    d_output.allocate<float>({batchSize, blockSize, vocabSize});
    std::vector<float> h_output_data(batchSize * blockSize * vocabSize);

    std::mt19937 rng(std::random_device{}());

    for (int i = 0; i < maxLength; i++)
    {
        // Prepare input (last blockSize tokens)
        std::vector<int> input_tokens(blockSize, 0);
        int start_idx = std::max(0, (int)tokens.size() - blockSize);
        for (int j = 0; j < blockSize && start_idx + j < tokens.size(); j++)
        {
            input_tokens[j] = tokens[start_idx + j];
        }

        // Copy to device
        cudaMemcpy(d_input.data, input_tokens.data(), d_input.bytes, cudaMemcpyHostToDevice);

        // Run model
        executeGraph(
            d_input.data, d_input.shape.data(), d_input.shape.size(), "int32",
            d_output.data, d_output.shape.data(), d_output.shape.size(), "float32",
            param_data_ptrs, param_shape_ptrs, param_dims_array, param_dtypes, 100,
            workspace, workspaceSize);

        // Get output
        cudaMemcpy(h_output_data.data(), d_output.data, d_output.bytes, cudaMemcpyDeviceToHost);

        // Sample next token from the last position
        int last_pos = std::min((int)tokens.size(), blockSize - 1);
        std::vector<float> logits(vocabSize);
        for (int v = 0; v < vocabSize; v++)
        {
            logits[v] = h_output_data[last_pos * vocabSize + v];
        }

        // Apply temperature and sample
        float temperature = 0.8f;
        for (float &logit : logits)
            logit /= temperature;

        // Softmax
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (float &logit : logits)
        {
            logit = std::exp(logit - max_logit);
            sum += logit;
        }
        for (float &logit : logits)
            logit /= sum;

        // Sample
        std::discrete_distribution<int> dist(logits.begin(), logits.end());
        int next_token = dist(rng);

        tokens.push_back(next_token);

        // Convert token to character and add to result
        if (tokenizer.idxToChar.find(next_token) != tokenizer.idxToChar.end())
        {
            result += tokenizer.idxToChar.at(next_token);
            std::cout << tokenizer.idxToChar.at(next_token) << std::flush;
        }
    }

    d_input.free();
    d_output.free();
    return result;
}

// ============================================================================
// Main Host Function
// ============================================================================

int main(int argc, char *argv[])
{
    // 1. Model Hyperparameters
    const int batchSize = 32;
    const int blockSize = 128;
    const int embedDim = 384;
    const int numHeads = 6;
    const int numLayers = 6;
    const int ffnHiddenDim = 4 * embedDim;
    
    // Workspace size configurable at compile time
    #ifndef WORKSPACE_SIZE_MB
    #define WORKSPACE_SIZE_MB 1024  // Default 1GB if not specified
    #endif
    const size_t workspaceSize = WORKSPACE_SIZE_MB * 1024ULL * 1024ULL; // Convert MB to bytes
    std::cout << "Using workspace size: " << WORKSPACE_SIZE_MB << "MB (" << workspaceSize << " bytes)" << std::endl;

    // 2. Handle input text from command line or default to file
    std::string inputText;
    bool useFile = true;

    if (argc > 1)
    {
        // Use command line argument as input text
        inputText = argv[1];
        useFile = false;
        std::cout << "Using command line input: \"" << inputText << "\"" << std::endl;
    }
    else
    {
        // Load Shakespeare dataset for tokenizer vocabulary
        std::cout << "Loading Shakespeare dataset for vocabulary..." << std::endl;
        std::ifstream file("src/cuda-work/shakespeare.txt");
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open shakespeare.txt" << std::endl;
            return 1;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        inputText = buffer.str();
        std::cout << "No command line input provided. Using Shakespeare dataset." << std::endl;
    }

    // Create tokenizer with full vocabulary
    CharacterTokenizer tokenizer(inputText);
    const int vocabSize = tokenizer.vocabSize;
    std::cout << "Vocabulary size: " << vocabSize << " unique characters." << std::endl;

    // 3. Prepare input data
    std::vector<int> h_input_data;
    std::vector<int> h_target_data;

    if (useFile)
    {
        // Use random batch from file (training mode)
        std::vector<int> data = tokenizer.encode(inputText);
        getBatch(data, blockSize, batchSize, h_input_data, h_target_data);
    }
    else
    {
        // Use provided text (inference mode)
        std::vector<int> encoded = tokenizer.encode(inputText);

        // Pad or truncate to blockSize
        h_input_data.resize(batchSize * blockSize, 0);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < blockSize && i < encoded.size(); i++)
            {
                h_input_data[b * blockSize + i] = encoded[i];
            }
        }

        std::cout << "Input tokens: ";
        for (int i = 0; i < std::min(10, (int)encoded.size()); i++)
        {
            std::cout << encoded[i] << " ";
        }
        if (encoded.size() > 10)
            std::cout << "...";
        std::cout << std::endl;
    }

    // 4. Allocate Host and Device Memory
    std::cout << "Allocating memory..." << std::endl;

    // Input/Output
    DeviceTensor d_input, d_output;
    d_input.allocate<int>({batchSize, blockSize});
    d_output.allocate<float>({batchSize, blockSize, vocabSize});
    std::vector<float> h_output_data(batchSize * blockSize * vocabSize);

    // Workspace
    DeviceTensor d_workspace;
    cudaMalloc(&d_workspace.data, workspaceSize);

    // Parameters - Load from file or initialize randomly
    std::vector<DeviceTensor> d_params(100);
    std::vector<std::vector<float>> h_params(100);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> weight_dist(-0.02f, 0.02f);

    auto init_param = [&](int i, const std::vector<int>& shape) {
        d_params[i].allocate<float>(shape);
        long long total_elements = 1;
        for(int dim : shape) total_elements *= dim;
        h_params[i].resize(total_elements);

        // Try to load from weights file first
        std::string weights_file = "weights.bin";
        std::ifstream wfile(weights_file, std::ios::binary);
        bool loaded = false;

        if (wfile.is_open())
        {
            // Skip to this parameter's position in the file
            size_t offset = 0;
            for (int j = 0; j < i; j++)
            {
                long long prev_elements = 1;
                for (int dim : param_shapes[j])
                    prev_elements *= dim;
                offset += prev_elements * sizeof(float);
            }
            wfile.seekg(offset);

            // Read this parameter's data
            wfile.read(reinterpret_cast<char *>(h_params[i].data()), total_elements * sizeof(float));
            loaded = wfile.good();
        }

        if (!loaded)
        {
            // Fallback to random initialization
            std::cout << "No trained weights found, using random initialization for param " << i << std::endl;
            for (long long j = 0; j < total_elements; ++j)
            {
                h_params[i][j] = weight_dist(rng);
            }
        }
        else
        {
            std::cout << "Loaded trained weights for param " << i << std::endl;
        }

        cudaMemcpy(d_params[i].data, h_params[i].data(), d_params[i].bytes, cudaMemcpyHostToDevice);
    };

    // Define all parameter shapes
    std::vector<std::vector<int>> param_shapes;
    param_shapes.push_back({vocabSize, embedDim}); // param 0
    param_shapes.push_back({embedDim / 2}); // param 1

    for (int i = 0; i < numLayers; ++i) {
        // MHA
        param_shapes.push_back({embedDim, embedDim}); // Wq_w
        param_shapes.push_back({embedDim});           // Wq_b
        param_shapes.push_back({embedDim, embedDim}); // Wk_w
        param_shapes.push_back({embedDim});           // Wk_b
        param_shapes.push_back({embedDim, embedDim}); // Wv_w
        param_shapes.push_back({embedDim});           // Wv_b
        param_shapes.push_back({embedDim, embedDim}); // Wo_w
        param_shapes.push_back({embedDim});           // Wo_b
        // Norm1
        param_shapes.push_back({embedDim}); // gamma
        param_shapes.push_back({embedDim}); // beta
        // FFN
        param_shapes.push_back({embedDim, ffnHiddenDim}); // ffn1_w
        param_shapes.push_back({ffnHiddenDim});          // ffn1_b
        param_shapes.push_back({ffnHiddenDim, embedDim}); // ffn2_w
        param_shapes.push_back({embedDim});              // ffn2_b
        // Norm2
        param_shapes.push_back({embedDim}); // gamma
        param_shapes.push_back({embedDim}); // beta
    }
    // Final LM Head
    param_shapes.push_back({embedDim, vocabSize}); // weights
    param_shapes.push_back({vocabSize});           // bias

    // Initialize all parameters
    for(int i=0; i < param_shapes.size(); ++i) {
        init_param(i, param_shapes[i]);
    }

    // 5. Copy Input Data to Device
    cudaMemcpy(d_input.data, h_input_data.data(), d_input.bytes, cudaMemcpyHostToDevice);

    // 6. Execute the Graph
    std::cout << "Executing CUDA graph..." << std::endl;
    
    // Prepare arrays for the new interface
    std::vector<void*> param_data_ptrs(100);
    std::vector<const int*> param_shape_ptrs(100);
    std::vector<int> param_dims_array(100);
    std::vector<const char*> param_dtypes(100, "float32");
    
    for (int i = 0; i < 100; i++) {
        param_data_ptrs[i] = d_params[i].data;
        param_shape_ptrs[i] = d_params[i].shape.data();
        param_dims_array[i] = d_params[i].shape.size();
    }
    
    executeGraph(
        d_input.data,                    // input_data
        d_input.shape.data(),            // input_shape
        d_input.shape.size(),            // input_dims
        "int32",                         // input_dtype
        
        d_output.data,                   // output_data
        d_output.shape.data(),           // output_shape
        d_output.shape.size(),           // output_dims
        "float32",                       // output_dtype
        
        param_data_ptrs.data(),          // param_data
        param_shape_ptrs.data(),         // param_shapes
        param_dims_array.data(),         // param_dims
        param_dtypes.data(),             // param_dtypes
        100,                             // param_count
        
        (char*)d_workspace.data,         // workspace
        workspaceSize                    // workspace_size
    );
    cudaDeviceSynchronize();
    std::cout << "Execution complete." << std::endl;

    // 7. Copy Output Data to Host and Print Sample
    cudaMemcpy(h_output_data.data(), d_output.data, d_output.bytes, cudaMemcpyDeviceToHost);

    // Check if we should run interactive generation
    if (argc > 2 && std::string(argv[2]) == "--generate")
    {
        std::cout << "\n=== INTERACTIVE TEXT GENERATION ===" << std::endl;
        std::cout << "Starting with prompt: \"" << inputText << "\"" << std::endl;
        std::cout << "Generated text: ";

        std::string generated = generateText(
            tokenizer, inputText, 100, // Generate 100 characters
            param_data_ptrs.data(),
            param_shape_ptrs.data(),
            param_dims_array.data(),
            param_dtypes.data(),
            (char *)d_workspace.data,
            workspaceSize);
        std::cout << "\n\nComplete generated text:\n"
                  << generated << std::endl;
    }
    else
    {
        // Original behavior: show raw logits
        std::cout << "\n--- Sample Output Logits (First 5 tokens of first batch item) ---" << std::endl;
        for (int i = 0; i < 5; ++i)
        {
            std::cout << "Token " << i << ": [";
            for (int j = 0; j < 5; ++j)
            {
                std::cout << h_output_data[i * vocabSize + j] << (j == 4 ? "" : ", ");
            }
            std::cout << "...]" << std::endl;
        }

        // Show predicted characters for the input
        if (!useFile)
        {
            std::cout << "\n--- Predicted Next Characters ---" << std::endl;
            std::vector<int> encoded = tokenizer.encode(inputText);
            int seq_len = std::min((int)encoded.size(), blockSize);

            for (int pos = 0; pos < std::min(5, seq_len); pos++)
            {
                // Find the token with highest probability
                float max_prob = -std::numeric_limits<float>::infinity();
                int best_token = 0;
                for (int v = 0; v < vocabSize; v++)
                {
                    float prob = h_output_data[pos * vocabSize + v];
                    if (prob > max_prob)
                    {
                        max_prob = prob;
                        best_token = v;
                    }
                }

                char input_char = (pos < encoded.size()) ? tokenizer.idxToChar[encoded[pos]] : '?';
                char predicted_char = tokenizer.idxToChar[best_token];
                std::cout << "After '" << input_char << "' -> predicted: '" << predicted_char
                          << "' (logit: " << max_prob << ")" << std::endl;
            }
        }
    }

    // 8. Free Memory
    std::cout << "\nFreeing memory..." << std::endl;
    d_input.free();
    d_output.free();
    d_workspace.free();
    for(auto& p : d_params) {
        p.free();
    }
    std::cout << "Done." << std::endl;

    return 0;
}
