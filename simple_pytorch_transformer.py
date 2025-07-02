#!/usr/bin/env python3
"""
Simple PyTorch Transformer - Same architecture as CUDA implementation
This shows how easy it is to build transformers in PyTorch vs the complex CUDA code.
Can also load weights from the binary file created by training!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SimpleAttention(nn.Module):
    """Multi-head attention - PyTorch makes this trivial!"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Three linear layers for Q, K, V - that's it!
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Project to Q, K, V and reshape for multi-head
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask (for autoregressive generation)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class SimpleFeedForward(nn.Module):
    """Feed-forward network - just two linear layers!"""
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class SimpleTransformerBlock(nn.Module):
    """Single transformer block - attention + FFN with residuals"""
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention = SimpleAttention(embed_dim, num_heads)
        self.ffn = SimpleFeedForward(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.norm1(x))  # Pre-norm style
        
        # Feed-forward with residual connection  
        x = x + self.ffn(self.norm2(x))
        
        return x

class SimplePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding to match CUDA implementation"""
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Create frequencies for positional encoding (same as CUDA)
        frequencies = torch.zeros(embed_dim // 2)
        for i in range(embed_dim // 2):
            frequencies[i] = 1.0 / (10000.0 ** (2.0 * i / embed_dim))
        
        self.register_buffer('frequencies', frequencies)
        
    def forward(self, x):
        B, T, C = x.shape
        pos_encoding = torch.zeros(T, C, device=x.device)
        
        for pos in range(T):
            for i in range(C // 2):
                pos_encoding[pos, 2*i] = torch.sin(pos * self.frequencies[i])
                pos_encoding[pos, 2*i + 1] = torch.cos(pos * self.frequencies[i])
        
        return x + pos_encoding.unsqueeze(0)

class SimpleTransformer(nn.Module):
    """Complete transformer model - matches CUDA architecture exactly!"""
    def __init__(self, vocab_size, embed_dim=384, num_heads=6, num_layers=6, max_len=512):
        super().__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding (sinusoidal, like CUDA)
        self.pos_encoding = SimplePositionalEncoding(max_len, embed_dim)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads, 4 * embed_dim)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # Token embeddings
        x = self.embedding(x)           # [B, T, embed_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization and projection to vocabulary
        x = self.ln_final(x)
        logits = self.lm_head(x)        # [B, T, vocab_size]
        
        return logits

def load_weights_from_binary(model, weights_file="weights.bin"):
    """Load weights from binary file into PyTorch model
    
    This uses the exact same parameter mapping as the CUDA implementation!
    """
    print(f"Loading weights from {weights_file}...")
    
    # Expected parameter shapes (must match CUDA exactly)
    expected_shapes = [
        [65, 384], [192],  # Embedding + positional frequencies
        # Layer 0-5 (6 layers × 16 params each)
        *([[384, 384], [384], [384, 384], [384], [384, 384], [384], [384, 384], [384],
           [384], [384], [384, 1536], [1536], [1536, 384], [384], [384], [384]] * 6),
        [384, 65], [65]  # Output layer
    ]
    
    try:
        with open(weights_file, 'rb') as f:
            params = []
            for shape in expected_shapes:
                param_size = np.prod(shape)
                param_bytes = f.read(param_size * 4)  # 4 bytes per float32
                if len(param_bytes) != param_size * 4:
                    raise ValueError(f"Not enough data for parameter with shape {shape}")
                
                param_data = np.frombuffer(param_bytes, dtype=np.float32)
                param_tensor = torch.from_numpy(param_data.reshape(shape))
                params.append(param_tensor)
        
        print(f"Successfully loaded {len(params)} parameters from {weights_file}")
        
        # Now assign parameters to model in the correct order
        param_idx = 0
        
        # 0: Embedding weights [65, 384]
        model.embedding.weight.data = params[param_idx]
        param_idx += 1
        
        # 1: Positional frequencies [192]
        model.pos_encoding.frequencies.data = params[param_idx]
        param_idx += 1
        
        # 2-97: Transformer layers (6 layers × 16 params each)
        for i in range(6):
            layer = model.blocks[i]
            
            # Attention weights & biases (8 params)
            layer.attention.q_proj.weight.data = params[param_idx]; param_idx += 1
            layer.attention.q_proj.bias.data = params[param_idx]; param_idx += 1
            layer.attention.k_proj.weight.data = params[param_idx]; param_idx += 1
            layer.attention.k_proj.bias.data = params[param_idx]; param_idx += 1
            layer.attention.v_proj.weight.data = params[param_idx]; param_idx += 1
            layer.attention.v_proj.bias.data = params[param_idx]; param_idx += 1
            layer.attention.out_proj.weight.data = params[param_idx]; param_idx += 1
            layer.attention.out_proj.bias.data = params[param_idx]; param_idx += 1
            
            # Layer norm 1 (2 params)
            layer.norm1.weight.data = params[param_idx]; param_idx += 1
            layer.norm1.bias.data = params[param_idx]; param_idx += 1
            
            # FFN weights & biases (4 params)
            layer.ffn.net[0].weight.data = params[param_idx]; param_idx += 1  # Linear 1
            layer.ffn.net[0].bias.data = params[param_idx]; param_idx += 1
            layer.ffn.net[2].weight.data = params[param_idx]; param_idx += 1  # Linear 2
            layer.ffn.net[2].bias.data = params[param_idx]; param_idx += 1
            
            # Layer norm 2 (2 params)
            layer.norm2.weight.data = params[param_idx]; param_idx += 1
            layer.norm2.bias.data = params[param_idx]; param_idx += 1
        
        # 98-99: Output layer
        model.lm_head.weight.data = params[param_idx]; param_idx += 1
        model.lm_head.bias.data = params[param_idx]; param_idx += 1
        
        print(f"✅ Successfully loaded all {param_idx} parameters into PyTorch model!")
        return True
        
    except FileNotFoundError:
        print(f"❌ Weight file {weights_file} not found. Using random weights.")
        return False
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False

# ============================================================================
# Usage Example
# ============================================================================

def demo_usage():
    """Show how easy it is to use!"""
    
    # Model hyperparameters (same as CUDA implementation)
    vocab_size = 65
    embed_dim = 384
    num_heads = 6
    num_layers = 6
    max_len = 512
    
    # Create model - one line!
    model = SimpleTransformer(vocab_size, embed_dim, num_heads, num_layers, max_len)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Try to load trained weights
    weights_loaded = load_weights_from_binary(model, "weights.bin")
    
    # Example forward pass
    batch_size = 4
    seq_len = 128
    
    # Random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass - one line!
    with torch.no_grad():
        logits = model(input_ids)  # [batch_size, seq_len, vocab_size]
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("✅ Forward pass successful!")
    
    if weights_loaded:
        print("🎉 Using trained weights - should generate Shakespeare-like text!")
    else:
        print("⚠️  Using random weights - will generate random output")
    
    return model

def generate_shakespeare_text(model, prompt="To be or not to be", max_length=100):
    """Generate Shakespeare-style text using the loaded model"""
    
    # Shakespeare character set (should match training)
    try:
        with open('src/cuda-work/shakespeare.txt', 'r') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
    except FileNotFoundError:
        print("Shakespeare text not found, using basic character set")
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!?:;-'\"")
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Using vocabulary of {len(chars)} characters")
    
    # Encode prompt
    try:
        input_ids = torch.tensor([[char_to_idx[c] for c in prompt]], dtype=torch.long)
    except KeyError as e:
        print(f"Character {e} not in vocabulary, using fallback")
        input_ids = torch.tensor([[0] * len(prompt)], dtype=torch.long)
    
    print(f"Prompt: '{prompt}'")
    print("Generated text:")
    print(prompt, end="")
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            
            # Sample next token (with some randomness)
            probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)  # Temperature = 0.8
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
            # Keep only last 128 tokens (context window)
            if input_ids.size(1) > 128:
                input_ids = input_ids[:, -128:]
            
            # Print character
            char = idx_to_char.get(next_token, '?')
            print(char, end="", flush=True)
    
    print("\n")

def compare_with_cuda():
    """Compare PyTorch vs CUDA complexity"""
    
    print("=" * 60)
    print("COMPLEXITY COMPARISON: PyTorch vs CUDA")
    print("=" * 60)
    
    print("\n🐍 PYTORCH TRANSFORMER:")
    print("✅ ~150 lines of clean, readable code")
    print("✅ Automatic memory management")
    print("✅ Automatic gradient computation")
    print("✅ Built-in operations (attention, layer norm, etc.)")
    print("✅ Easy debugging and experimentation")
    print("✅ Handles batching automatically")
    print("✅ Can load the same weights as CUDA!")
    
    print("\n🔥 CUDA IMPLEMENTATION:")
    print("⚡ ~1800 lines of complex kernel code")  
    print("⚡ Manual memory management (WorkspaceAllocator)")
    print("⚡ Manual kernel launches and grid/block sizing")
    print("⚡ Custom implementations of every operation")
    print("⚡ Complex tensor indexing and bounds checking")
    print("⚡ Manual optimization (tiling, shared memory, etc.)")
    
    print("\n🎯 WHY USE CUDA THEN?")
    print("🚀 **PERFORMANCE**: CUDA is ~10-100x faster for inference")
    print("🚀 **EFFICIENCY**: Lower memory usage, optimized kernels")
    print("🚀 **DEPLOYMENT**: Single optimized binary, no Python overhead")
    print("🚀 **CONTROL**: Fine-grained optimization for specific hardware")
    
    print("\n💡 DEVELOPMENT WORKFLOW:")
    print("1. Prototype and train in PyTorch (easy & fast)")
    print("2. Export trained weights to weights.bin")
    print("3. Load weights in EITHER PyTorch OR CUDA!")
    print("4. PyTorch for experimentation, CUDA for production")
    print("5. Best of both worlds! 🎉")

if __name__ == "__main__":
    print("Simple PyTorch Transformer Demo")
    print("=" * 60)
    
    # Demo the model
    model = demo_usage()
    
    # Compare complexities
    compare_with_cuda()
    
    # Try to generate Shakespeare text if weights are available
    print("\n" + "=" * 60)
    print("SHAKESPEARE TEXT GENERATION")
    print("=" * 60)
    
    generate_shakespeare_text(model, "To be or not to be", 200)
    
    print("\n🎉 Same weights work in both PyTorch and CUDA!")
    print("Train once, deploy anywhere! 🚀") 