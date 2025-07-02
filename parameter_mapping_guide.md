# Shakespeare LLM: PyTorch → CUDA Parameter Mapping Guide

This guide shows exactly how trained PyTorch weights map to the 100 parameters expected by the CUDA kernel.

## Model Architecture Overview
- **Vocabulary Size**: 65 characters
- **Embedding Dimension**: 384
- **Number of Transformer Layers**: 6
- **Number of Attention Heads**: 6 (head_dim = 384/6 = 64)
- **FFN Hidden Dimension**: 1536 (4 × 384)

## Complete Parameter Mapping (100 Parameters)

### **Embedding & Positional Encoding (2 parameters)**
```python
# PyTorch → CUDA
model.embedding.weight           → param_0   [65, 384]      # Token embeddings
model.pos_encoding.frequencies   → param_1   [192]          # Positional frequencies (384/2)
```

### **Transformer Layer 0 (16 parameters)**
```python
# Multi-Head Attention
model.layers[0].attention.q_proj.weight → param_2   [384, 384]    # Query projection
model.layers[0].attention.q_proj.bias   → param_3   [384]         # Query bias
model.layers[0].attention.k_proj.weight → param_4   [384, 384]    # Key projection  
model.layers[0].attention.k_proj.bias   → param_5   [384]         # Key bias
model.layers[0].attention.v_proj.weight → param_6   [384, 384]    # Value projection
model.layers[0].attention.v_proj.bias   → param_7   [384]         # Value bias
model.layers[0].attention.out_proj.weight → param_8  [384, 384]    # Output projection
model.layers[0].attention.out_proj.bias   → param_9  [384]         # Output bias

# Layer Norm 1 (after attention)
model.layers[0].norm1.weight    → param_10  [384]         # Gamma
model.layers[0].norm1.bias      → param_11  [384]         # Beta

# Feed Forward Network
model.layers[0].ffn.linear1.weight → param_12 [384, 1536]   # First linear layer
model.layers[0].ffn.linear1.bias   → param_13 [1536]        # First linear bias
model.layers[0].ffn.linear2.weight → param_14 [1536, 384]   # Second linear layer  
model.layers[0].ffn.linear2.bias   → param_15 [384]         # Second linear bias

# Layer Norm 2 (after FFN)
model.layers[0].norm2.weight    → param_16  [384]         # Gamma
model.layers[0].norm2.bias      → param_17  [384]         # Beta
```

### **Transformer Layer 1 (16 parameters: param_18 - param_33)**
```python
# Multi-Head Attention  
model.layers[1].attention.q_proj.weight → param_18  [384, 384]
model.layers[1].attention.q_proj.bias   → param_19  [384]
model.layers[1].attention.k_proj.weight → param_20  [384, 384]
model.layers[1].attention.k_proj.bias   → param_21  [384]
model.layers[1].attention.v_proj.weight → param_22  [384, 384]
model.layers[1].attention.v_proj.bias   → param_23  [384]
model.layers[1].attention.out_proj.weight → param_24 [384, 384]
model.layers[1].attention.out_proj.bias   → param_25 [384]

# Layer Norms and FFN
model.layers[1].norm1.weight     → param_26  [384]
model.layers[1].norm1.bias       → param_27  [384]
model.layers[1].ffn.linear1.weight → param_28 [384, 1536]
model.layers[1].ffn.linear1.bias   → param_29 [1536]
model.layers[1].ffn.linear2.weight → param_30 [1536, 384]
model.layers[1].ffn.linear2.bias   → param_31 [384]
model.layers[1].norm2.weight     → param_32  [384]
model.layers[1].norm2.bias       → param_33  [384]
```

### **Transformer Layers 2-5 (64 parameters: param_34 - param_97)**
```python
# Each layer follows the same 16-parameter pattern:
# Layer 2: param_34 - param_49
# Layer 3: param_50 - param_65  
# Layer 4: param_66 - param_81
# Layer 5: param_82 - param_97

# Pattern for layer i (where i = 2,3,4,5):
base_idx = 18 + (i-1) * 16
model.layers[i].attention.q_proj.weight → param_{base_idx}     [384, 384]
model.layers[i].attention.q_proj.bias   → param_{base_idx+1}   [384]
model.layers[i].attention.k_proj.weight → param_{base_idx+2}   [384, 384]
model.layers[i].attention.k_proj.bias   → param_{base_idx+3}   [384]
model.layers[i].attention.v_proj.weight → param_{base_idx+4}   [384, 384]
model.layers[i].attention.v_proj.bias   → param_{base_idx+5}   [384]
model.layers[i].attention.out_proj.weight → param_{base_idx+6} [384, 384]
model.layers[i].attention.out_proj.bias   → param_{base_idx+7} [384]
model.layers[i].norm1.weight     → param_{base_idx+8}   [384]
model.layers[i].norm1.bias       → param_{base_idx+9}   [384]
model.layers[i].ffn.linear1.weight → param_{base_idx+10} [384, 1536]
model.layers[i].ffn.linear1.bias   → param_{base_idx+11} [1536]
model.layers[i].ffn.linear2.weight → param_{base_idx+12} [1536, 384]
model.layers[i].ffn.linear2.bias   → param_{base_idx+13} [384]
model.layers[i].norm2.weight     → param_{base_idx+14}  [384]
model.layers[i].norm2.bias       → param_{base_idx+15}  [384]
```

### **Output Layer (2 parameters)**
```python
# Final projection to vocabulary
model.output.weight              → param_98  [384, 65]     # Linear layer weight
model.output.bias                → param_99  [65]          # Linear layer bias
```

## PyTorch → Binary File Conversion

### 1. **Parameter Extraction Function**
```python
def extract_parameters_in_order(model):
    """Extract all 100 parameters in the exact order expected by CUDA kernel"""
    params = []
    
    # 0-1: Embedding & Positional
    params.append(model.embedding.weight.detach().cpu().numpy())           # [65, 384]
    params.append(model.pos_encoding.frequencies.detach().cpu().numpy())   # [192]
    
    # 2-97: Transformer layers (6 layers × 16 params each)
    for i in range(6):
        layer = model.layers[i]
        
        # Attention weights & biases (8 params)
        params.append(layer.attention.q_proj.weight.detach().cpu().numpy())    # [384, 384]
        params.append(layer.attention.q_proj.bias.detach().cpu().numpy())      # [384]
        params.append(layer.attention.k_proj.weight.detach().cpu().numpy())    # [384, 384]
        params.append(layer.attention.k_proj.bias.detach().cpu().numpy())      # [384]
        params.append(layer.attention.v_proj.weight.detach().cpu().numpy())    # [384, 384]
        params.append(layer.attention.v_proj.bias.detach().cpu().numpy())      # [384]
        params.append(layer.attention.out_proj.weight.detach().cpu().numpy())  # [384, 384]
        params.append(layer.attention.out_proj.bias.detach().cpu().numpy())    # [384]
        
        # Layer norm 1 (2 params)
        params.append(layer.norm1.weight.detach().cpu().numpy())               # [384]
        params.append(layer.norm1.bias.detach().cpu().numpy())                 # [384]
        
        # FFN weights & biases (4 params)
        params.append(layer.ffn.linear1.weight.detach().cpu().numpy())         # [384, 1536]
        params.append(layer.ffn.linear1.bias.detach().cpu().numpy())           # [1536]
        params.append(layer.ffn.linear2.weight.detach().cpu().numpy())         # [1536, 384]
        params.append(layer.ffn.linear2.bias.detach().cpu().numpy())           # [384]
        
        # Layer norm 2 (2 params)
        params.append(layer.norm2.weight.detach().cpu().numpy())               # [384]
        params.append(layer.norm2.bias.detach().cpu().numpy())                 # [384]
    
    # 98-99: Output layer
    params.append(model.output.weight.detach().cpu().numpy())                  # [384, 65]
    params.append(model.output.bias.detach().cpu().numpy())                    # [65]
    
    return params

def save_weights_binary(model, filename="shakespeare_weights.bin"):
    """Save weights in binary format for C++ loading"""
    params = extract_parameters_in_order(model)
    
    with open(filename, 'wb') as f:
        for param in params:
            # Convert to float32 and write
            param_float32 = param.astype(np.float32)
            f.write(param_float32.tobytes())
    
    print(f"Saved {len(params)} parameters to {filename}")
    return len(params)
```

### 2. **C++ Weight Loading**
```cpp
// In main.cpp, this loads the saved binary weights:
std::string weights_file = "shakespeare_weights.bin";
std::ifstream wfile(weights_file, std::ios::binary);
if (wfile.is_open()) {
    // Skip to this parameter's position  
    size_t offset = get_parameter_offset(i);
    wfile.seekg(offset);
    wfile.read(reinterpret_cast<char*>(h_params[i].data()), 
               h_params[i].size() * sizeof(float));
    loaded = true;
}
```

## Important Notes

1. **Parameter Order**: The exact order in `extract_parameters_in_order()` **must match** the CUDA kernel's `param_tensors[0]` through `param_tensors[99]`.

2. **Shape Consistency**: Each parameter shape must match exactly between PyTorch and CUDA (shown in brackets above).

3. **Data Type**: All parameters are `float32` in both PyTorch and CUDA.

4. **Transformer Layer Pattern**: Each of the 6 transformer layers has identical structure (16 parameters each).

5. **Binary File Format**: Parameters are stored sequentially as raw `float32` bytes, no headers or metadata.

This mapping ensures that when you train in PyTorch and save weights, the CUDA kernel can load them correctly and produce meaningful Shakespeare-style text generation. 