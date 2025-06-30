#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// Forward declaration of the main CUDA execution function from the .cu file
extern "C" void executeGraph(
  int* input_data, const int* input_shape, int input_dims,
  float* output_data, const int* output_shape, int output_dims,
  float* param_0_embeddings_data, const int* param_0_embeddings_shape, int param_0_embeddings_dims,
  float* param_1_frequencies_data, const int* param_1_frequencies_shape, int param_1_frequencies_dims,
  float* param_2_weights_data, const int* param_2_weights_shape, int param_2_weights_dims,
  float* param_3_bias_data, const int* param_3_bias_shape, int param_3_bias_dims,
  float* param_4_weights_data, const int* param_4_weights_shape, int param_4_weights_dims,
  float* param_5_bias_data, const int* param_5_bias_shape, int param_5_bias_dims,
  float* param_6_weights_data, const int* param_6_weights_shape, int param_6_weights_dims,
  float* param_7_bias_data, const int* param_7_bias_shape, int param_7_bias_dims,
  float* param_8_weights_data, const int* param_8_weights_shape, int param_8_weights_dims,
  float* param_9_bias_data, const int* param_9_bias_shape, int param_9_bias_dims,
  float* param_10_gamma_data, const int* param_10_gamma_shape, int param_10_gamma_dims,
  float* param_11_beta_data, const int* param_11_beta_shape, int param_11_beta_dims,
  float* param_12_weights_data, const int* param_12_weights_shape, int param_12_weights_dims,
  float* param_13_bias_data, const int* param_13_bias_shape, int param_13_bias_dims,
  float* param_14_weights_data, const int* param_14_weights_shape, int param_14_weights_dims,
  float* param_15_bias_data, const int* param_15_bias_shape, int param_15_bias_dims,
  float* param_16_gamma_data, const int* param_16_gamma_shape, int param_16_gamma_dims,
  float* param_17_beta_data, const int* param_17_beta_shape, int param_17_beta_dims,
  float* param_18_weights_data, const int* param_18_weights_shape, int param_18_weights_dims,
  float* param_19_bias_data, const int* param_19_bias_shape, int param_19_bias_dims,
  float* param_20_weights_data, const int* param_20_weights_shape, int param_20_weights_dims,
  float* param_21_bias_data, const int* param_21_bias_shape, int param_21_bias_dims,
  float* param_22_weights_data, const int* param_22_weights_shape, int param_22_weights_dims,
  float* param_23_bias_data, const int* param_23_bias_shape, int param_23_bias_dims,
  float* param_24_weights_data, const int* param_24_weights_shape, int param_24_weights_dims,
  float* param_25_bias_data, const int* param_25_bias_shape, int param_25_bias_dims,
  float* param_26_gamma_data, const int* param_26_gamma_shape, int param_26_gamma_dims,
  float* param_27_beta_data, const int* param_27_beta_shape, int param_27_beta_dims,
  float* param_28_weights_data, const int* param_28_weights_shape, int param_28_weights_dims,
  float* param_29_bias_data, const int* param_29_bias_shape, int param_29_bias_dims,
  float* param_30_weights_data, const int* param_30_weights_shape, int param_30_weights_dims,
  float* param_31_bias_data, const int* param_31_bias_shape, int param_31_bias_dims,
  float* param_32_gamma_data, const int* param_32_gamma_shape, int param_32_gamma_dims,
  float* param_33_beta_data, const int* param_33_beta_shape, int param_33_beta_dims,
  float* param_34_weights_data, const int* param_34_weights_shape, int param_34_weights_dims,
  float* param_35_bias_data, const int* param_35_bias_shape, int param_35_bias_dims,
  float* param_36_weights_data, const int* param_36_weights_shape, int param_36_weights_dims,
  float* param_37_bias_data, const int* param_37_bias_shape, int param_37_bias_dims,
  float* param_38_weights_data, const int* param_38_weights_shape, int param_38_weights_dims,
  float* param_39_bias_data, const int* param_39_bias_shape, int param_39_bias_dims,
  float* param_40_weights_data, const int* param_40_weights_shape, int param_40_weights_dims,
  float* param_41_bias_data, const int* param_41_bias_shape, int param_41_bias_dims,
  float* param_42_gamma_data, const int* param_42_gamma_shape, int param_42_gamma_dims,
  float* param_43_beta_data, const int* param_43_beta_shape, int param_43_beta_dims,
  float* param_44_weights_data, const int* param_44_weights_shape, int param_44_weights_dims,
  float* param_45_bias_data, const int* param_45_bias_shape, int param_45_bias_dims,
  float* param_46_weights_data, const int* param_46_weights_shape, int param_46_weights_dims,
  float* param_47_bias_data, const int* param_47_bias_shape, int param_47_bias_dims,
  float* param_48_gamma_data, const int* param_48_gamma_shape, int param_48_gamma_dims,
  float* param_49_beta_data, const int* param_49_beta_shape, int param_49_beta_dims,
  float* param_50_weights_data, const int* param_50_weights_shape, int param_50_weights_dims,
  float* param_51_bias_data, const int* param_51_bias_shape, int param_51_bias_dims,
  float* param_52_weights_data, const int* param_52_weights_shape, int param_52_weights_dims,
  float* param_53_bias_data, const int* param_53_bias_shape, int param_53_bias_dims,
  float* param_54_weights_data, const int* param_54_weights_shape, int param_54_weights_dims,
  float* param_55_bias_data, const int* param_55_bias_shape, int param_55_bias_dims,
  float* param_56_weights_data, const int* param_56_weights_shape, int param_56_weights_dims,
  float* param_57_bias_data, const int* param_57_bias_shape, int param_57_bias_dims,
  float* param_58_gamma_data, const int* param_58_gamma_shape, int param_58_gamma_dims,
  float* param_59_beta_data, const int* param_59_beta_shape, int param_59_beta_dims,
  float* param_60_weights_data, const int* param_60_weights_shape, int param_60_weights_dims,
  float* param_61_bias_data, const int* param_61_bias_shape, int param_61_bias_dims,
  float* param_62_weights_data, const int* param_62_weights_shape, int param_62_weights_dims,
  float* param_63_bias_data, const int* param_63_bias_shape, int param_63_bias_dims,
  float* param_64_gamma_data, const int* param_64_gamma_shape, int param_64_gamma_dims,
  float* param_65_beta_data, const int* param_65_beta_shape, int param_65_beta_dims,
  float* param_66_weights_data, const int* param_66_weights_shape, int param_66_weights_dims,
  float* param_67_bias_data, const int* param_67_bias_shape, int param_67_bias_dims,
  float* param_68_weights_data, const int* param_68_weights_shape, int param_68_weights_dims,
  float* param_69_bias_data, const int* param_69_bias_shape, int param_69_bias_dims,
  float* param_70_weights_data, const int* param_70_weights_shape, int param_70_weights_dims,
  float* param_71_bias_data, const int* param_71_bias_shape, int param_71_bias_dims,
  float* param_72_weights_data, const int* param_72_weights_shape, int param_72_weights_dims,
  float* param_73_bias_data, const int* param_73_bias_shape, int param_73_bias_dims,
  float* param_74_gamma_data, const int* param_74_gamma_shape, int param_74_gamma_dims,
  float* param_75_beta_data, const int* param_75_beta_shape, int param_75_beta_dims,
  float* param_76_weights_data, const int* param_76_weights_shape, int param_76_weights_dims,
  float* param_77_bias_data, const int* param_77_bias_shape, int param_77_bias_dims,
  float* param_78_weights_data, const int* param_78_weights_shape, int param_78_weights_dims,
  float* param_79_bias_data, const int* param_79_bias_shape, int param_79_bias_dims,
  float* param_80_gamma_data, const int* param_80_gamma_shape, int param_80_gamma_dims,
  float* param_81_beta_data, const int* param_81_beta_shape, int param_81_beta_dims,
  float* param_82_weights_data, const int* param_82_weights_shape, int param_82_weights_dims,
  float* param_83_bias_data, const int* param_83_bias_shape, int param_83_bias_dims,
  float* param_84_weights_data, const int* param_84_weights_shape, int param_84_weights_dims,
  float* param_85_bias_data, const int* param_85_bias_shape, int param_85_bias_dims,
  float* param_86_weights_data, const int* param_86_weights_shape, int param_86_weights_dims,
  float* param_87_bias_data, const int* param_87_bias_shape, int param_87_bias_dims,
  float* param_88_weights_data, const int* param_88_weights_shape, int param_88_weights_dims,
  float* param_89_bias_data, const int* param_89_bias_shape, int param_89_bias_dims,
  float* param_90_gamma_data, const int* param_90_gamma_shape, int param_90_gamma_dims,
  float* param_91_beta_data, const int* param_91_beta_shape, int param_91_beta_dims,
  float* param_92_weights_data, const int* param_92_weights_shape, int param_92_weights_dims,
  float* param_93_bias_data, const int* param_93_bias_shape, int param_93_bias_dims,
  float* param_94_weights_data, const int* param_94_weights_shape, int param_94_weights_dims,
  float* param_95_bias_data, const int* param_95_bias_shape, int param_95_bias_dims,
  float* param_96_gamma_data, const int* param_96_gamma_shape, int param_96_gamma_dims,
  float* param_97_beta_data, const int* param_97_beta_shape, int param_97_beta_dims,
  float* param_98_weights_data, const int* param_98_weights_shape, int param_98_weights_dims,
  float* param_99_bias_data, const int* param_99_bias_shape, int param_99_bias_dims,
  char* workspace, size_t workspace_size
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
// Main Host Function
// ============================================================================

int main() {
    // 1. Model Hyperparameters
    const int batchSize = 32;
    const int blockSize = 128;
    const int embedDim = 384;
    const int numHeads = 6;
    const int numLayers = 6;
    const int ffnHiddenDim = 4 * embedDim;
    const size_t workspaceSize = 113246208 * 2; // A safe upper bound from the .cu file

    // 2. Load and Tokenize Data
    std::cout << "Loading Shakespeare dataset..." << std::endl;
    std::ifstream file("src/cuda-work/shakespeare.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open shakespeare.txt" << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    CharacterTokenizer tokenizer(text);
    const int vocabSize = tokenizer.vocabSize;
    std::cout << "Dataset loaded: " << text.length() << " characters, " << vocabSize << " unique." << std::endl;

    // 3. Prepare a batch of data
    std::vector<int> data = tokenizer.encode(text);
    std::vector<int> h_input_data;
    std::vector<int> h_target_data;
    getBatch(data, blockSize, batchSize, h_input_data, h_target_data);

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

    // Parameters
    std::vector<DeviceTensor> d_params(100);
    std::vector<std::vector<float>> h_params(100);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> weight_dist(-0.02f, 0.02f);

    auto init_param = [&](int i, const std::vector<int>& shape) {
        d_params[i].allocate<float>(shape);
        long long total_elements = 1;
        for(int dim : shape) total_elements *= dim;
        h_params[i].resize(total_elements);
        for(long long j = 0; j < total_elements; ++j) {
            h_params[i][j] = weight_dist(rng);
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
    executeGraph(
        (int*)d_input.data, d_input.shape.data(), d_input.shape.size(),
        (float*)d_output.data, d_output.shape.data(), d_output.shape.size(),
        (float*)d_params[0].data, d_params[0].shape.data(), d_params[0].shape.size(),
        (float*)d_params[1].data, d_params[1].shape.data(), d_params[1].shape.size(),
        (float*)d_params[2].data, d_params[2].shape.data(), d_params[2].shape.size(),
        (float*)d_params[3].data, d_params[3].shape.data(), d_params[3].shape.size(),
        (float*)d_params[4].data, d_params[4].shape.data(), d_params[4].shape.size(),
        (float*)d_params[5].data, d_params[5].shape.data(), d_params[5].shape.size(),
        (float*)d_params[6].data, d_params[6].shape.data(), d_params[6].shape.size(),
        (float*)d_params[7].data, d_params[7].shape.data(), d_params[7].shape.size(),
        (float*)d_params[8].data, d_params[8].shape.data(), d_params[8].shape.size(),
        (float*)d_params[9].data, d_params[9].shape.data(), d_params[9].shape.size(),
        (float*)d_params[10].data, d_params[10].shape.data(), d_params[10].shape.size(),
        (float*)d_params[11].data, d_params[11].shape.data(), d_params[11].shape.size(),
        (float*)d_params[12].data, d_params[12].shape.data(), d_params[12].shape.size(),
        (float*)d_params[13].data, d_params[13].shape.data(), d_params[13].shape.size(),
        (float*)d_params[14].data, d_params[14].shape.data(), d_params[14].shape.size(),
        (float*)d_params[15].data, d_params[15].shape.data(), d_params[15].shape.size(),
        (float*)d_params[16].data, d_params[16].shape.data(), d_params[16].shape.size(),
        (float*)d_params[17].data, d_params[17].shape.data(), d_params[17].shape.size(),
        (float*)d_params[18].data, d_params[18].shape.data(), d_params[18].shape.size(),
        (float*)d_params[19].data, d_params[19].shape.data(), d_params[19].shape.size(),
        (float*)d_params[20].data, d_params[20].shape.data(), d_params[20].shape.size(),
        (float*)d_params[21].data, d_params[21].shape.data(), d_params[21].shape.size(),
        (float*)d_params[22].data, d_params[22].shape.data(), d_params[22].shape.size(),
        (float*)d_params[23].data, d_params[23].shape.data(), d_params[23].shape.size(),
        (float*)d_params[24].data, d_params[24].shape.data(), d_params[24].shape.size(),
        (float*)d_params[25].data, d_params[25].shape.data(), d_params[25].shape.size(),
        (float*)d_params[26].data, d_params[26].shape.data(), d_params[26].shape.size(),
        (float*)d_params[27].data, d_params[27].shape.data(), d_params[27].shape.size(),
        (float*)d_params[28].data, d_params[28].shape.data(), d_params[28].shape.size(),
        (float*)d_params[29].data, d_params[29].shape.data(), d_params[29].shape.size(),
        (float*)d_params[30].data, d_params[30].shape.data(), d_params[30].shape.size(),
        (float*)d_params[31].data, d_params[31].shape.data(), d_params[31].shape.size(),
        (float*)d_params[32].data, d_params[32].shape.data(), d_params[32].shape.size(),
        (float*)d_params[33].data, d_params[33].shape.data(), d_params[33].shape.size(),
        (float*)d_params[34].data, d_params[34].shape.data(), d_params[34].shape.size(),
        (float*)d_params[35].data, d_params[35].shape.data(), d_params[35].shape.size(),
        (float*)d_params[36].data, d_params[36].shape.data(), d_params[36].shape.size(),
        (float*)d_params[37].data, d_params[37].shape.data(), d_params[37].shape.size(),
        (float*)d_params[38].data, d_params[38].shape.data(), d_params[38].shape.size(),
        (float*)d_params[39].data, d_params[39].shape.data(), d_params[39].shape.size(),
        (float*)d_params[40].data, d_params[40].shape.data(), d_params[40].shape.size(),
        (float*)d_params[41].data, d_params[41].shape.data(), d_params[41].shape.size(),
        (float*)d_params[42].data, d_params[42].shape.data(), d_params[42].shape.size(),
        (float*)d_params[43].data, d_params[43].shape.data(), d_params[43].shape.size(),
        (float*)d_params[44].data, d_params[44].shape.data(), d_params[44].shape.size(),
        (float*)d_params[45].data, d_params[45].shape.data(), d_params[45].shape.size(),
        (float*)d_params[46].data, d_params[46].shape.data(), d_params[46].shape.size(),
        (float*)d_params[47].data, d_params[47].shape.data(), d_params[47].shape.size(),
        (float*)d_params[48].data, d_params[48].shape.data(), d_params[48].shape.size(),
        (float*)d_params[49].data, d_params[49].shape.data(), d_params[49].shape.size(),
        (float*)d_params[50].data, d_params[50].shape.data(), d_params[50].shape.size(),
        (float*)d_params[51].data, d_params[51].shape.data(), d_params[51].shape.size(),
        (float*)d_params[52].data, d_params[52].shape.data(), d_params[52].shape.size(),
        (float*)d_params[53].data, d_params[53].shape.data(), d_params[53].shape.size(),
        (float*)d_params[54].data, d_params[54].shape.data(), d_params[54].shape.size(),
        (float*)d_params[55].data, d_params[55].shape.data(), d_params[55].shape.size(),
        (float*)d_params[56].data, d_params[56].shape.data(), d_params[56].shape.size(),
        (float*)d_params[57].data, d_params[57].shape.data(), d_params[57].shape.size(),
        (float*)d_params[58].data, d_params[58].shape.data(), d_params[58].shape.size(),
        (float*)d_params[59].data, d_params[59].shape.data(), d_params[59].shape.size(),
        (float*)d_params[60].data, d_params[60].shape.data(), d_params[60].shape.size(),
        (float*)d_params[61].data, d_params[61].shape.data(), d_params[61].shape.size(),
        (float*)d_params[62].data, d_params[62].shape.data(), d_params[62].shape.size(),
        (float*)d_params[63].data, d_params[63].shape.data(), d_params[63].shape.size(),
        (float*)d_params[64].data, d_params[64].shape.data(), d_params[64].shape.size(),
        (float*)d_params[65].data, d_params[65].shape.data(), d_params[65].shape.size(),
        (float*)d_params[66].data, d_params[66].shape.data(), d_params[66].shape.size(),
        (float*)d_params[67].data, d_params[67].shape.data(), d_params[67].shape.size(),
        (float*)d_params[68].data, d_params[68].shape.data(), d_params[68].shape.size(),
        (float*)d_params[69].data, d_params[69].shape.data(), d_params[69].shape.size(),
        (float*)d_params[70].data, d_params[70].shape.data(), d_params[70].shape.size(),
        (float*)d_params[71].data, d_params[71].shape.data(), d_params[71].shape.size(),
        (float*)d_params[72].data, d_params[72].shape.data(), d_params[72].shape.size(),
        (float*)d_params[73].data, d_params[73].shape.data(), d_params[73].shape.size(),
        (float*)d_params[74].data, d_params[74].shape.data(), d_params[74].shape.size(),
        (float*)d_params[75].data, d_params[75].shape.data(), d_params[75].shape.size(),
        (float*)d_params[76].data, d_params[76].shape.data(), d_params[76].shape.size(),
        (float*)d_params[77].data, d_params[77].shape.data(), d_params[77].shape.size(),
        (float*)d_params[78].data, d_params[78].shape.data(), d_params[78].shape.size(),
        (float*)d_params[79].data, d_params[79].shape.data(), d_params[79].shape.size(),
        (float*)d_params[80].data, d_params[80].shape.data(), d_params[80].shape.size(),
        (float*)d_params[81].data, d_params[81].shape.data(), d_params[81].shape.size(),
        (float*)d_params[82].data, d_params[82].shape.data(), d_params[82].shape.size(),
        (float*)d_params[83].data, d_params[83].shape.data(), d_params[83].shape.size(),
        (float*)d_params[84].data, d_params[84].shape.data(), d_params[84].shape.size(),
        (float*)d_params[85].data, d_params[85].shape.data(), d_params[85].shape.size(),
        (float*)d_params[86].data, d_params[86].shape.data(), d_params[86].shape.size(),
        (float*)d_params[87].data, d_params[87].shape.data(), d_params[87].shape.size(),
        (float*)d_params[88].data, d_params[88].shape.data(), d_params[88].shape.size(),
        (float*)d_params[89].data, d_params[89].shape.data(), d_params[89].shape.size(),
        (float*)d_params[90].data, d_params[90].shape.data(), d_params[90].shape.size(),
        (float*)d_params[91].data, d_params[91].shape.data(), d_params[91].shape.size(),
        (float*)d_params[92].data, d_params[92].shape.data(), d_params[92].shape.size(),
        (float*)d_params[93].data, d_params[93].shape.data(), d_params[93].shape.size(),
        (float*)d_params[94].data, d_params[94].shape.data(), d_params[94].shape.size(),
        (float*)d_params[95].data, d_params[95].shape.data(), d_params[95].shape.size(),
        (float*)d_params[96].data, d_params[96].shape.data(), d_params[96].shape.size(),
        (float*)d_params[97].data, d_params[97].shape.data(), d_params[97].shape.size(),
        (float*)d_params[98].data, d_params[98].shape.data(), d_params[98].shape.size(),
        (float*)d_params[99].data, d_params[99].shape.data(), d_params[99].shape.size(),
        (char*)d_workspace.data, workspaceSize
    );
    cudaDeviceSynchronize();
    std::cout << "Execution complete." << std::endl;

    // 7. Copy Output Data to Host and Print Sample
    cudaMemcpy(h_output_data.data(), d_output.data, d_output.bytes, cudaMemcpyDeviceToHost);

    std::cout << "\n--- Sample Output Logits (First 5 tokens of first batch item) ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Token " << i << ": [";
        for (int j = 0; j < 5; ++j) {
            std::cout << h_output_data[i * vocabSize + j] << (j == 4 ? "" : ", ");
        }
        std::cout << "...]" << std::endl;
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
