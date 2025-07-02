#!/usr/bin/env python3
"""
Simple Shakespeare training script that saves weights compatible with the C++ implementation.
This demonstrates how to train the model and export weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
from torch.utils.data import DataLoader, Dataset

class CharacterTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

class ShakespeareDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = tokenizer.encode(text)
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Causal mask for autoregressive generation
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Create frequencies for positional encoding
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

class ShakespeareTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=6, num_layers=6, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(max_len, embed_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4 * embed_dim)
            for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.output(x)

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
        params.append(layer.ffn[0].weight.detach().cpu().numpy())              # [384, 1536]
        params.append(layer.ffn[0].bias.detach().cpu().numpy())                # [1536]
        params.append(layer.ffn[2].weight.detach().cpu().numpy())              # [1536, 384]
        params.append(layer.ffn[2].bias.detach().cpu().numpy())                # [384]
        
        # Layer norm 2 (2 params)
        params.append(layer.norm2.weight.detach().cpu().numpy())               # [384]
        params.append(layer.norm2.bias.detach().cpu().numpy())                 # [384]
    
    # 98-99: Output layer
    params.append(model.output.weight.detach().cpu().numpy())                  # [384, 65]
    params.append(model.output.bias.detach().cpu().numpy())                    # [65]
    
    return params

def save_weights_binary(model, filename="weights.bin"):
    """Save weights in binary format for C++ loading"""
    params = extract_parameters_in_order(model)
    
    print(f"Saving {len(params)} parameters to {filename}")
    
    # Print parameter shapes for verification
    expected_shapes = [
        [65, 384], [192],  # Embedding + positional
        # Layer 0-5 (6 layers × 16 params each)
        *([[384, 384], [384], [384, 384], [384], [384, 384], [384], [384, 384], [384],
           [384], [384], [384, 1536], [1536], [1536, 384], [384], [384], [384]] * 6),
        [384, 65], [65]  # Output layer
    ]
    
    for i, (param, expected_shape) in enumerate(zip(params, expected_shapes)):
        if list(param.shape) != expected_shape:
            print(f"WARNING: param_{i} shape {list(param.shape)} != expected {expected_shape}")
    
    with open(filename, 'wb') as f:
        for param in params:
            # Convert to float32 and write
            param_float32 = param.astype(np.float32)
            f.write(param_float32.tobytes())
    
    print(f"Successfully saved {len(params)} parameters to {filename}")
    return len(params)

def train_model():
    # Load Shakespeare text
    try:
        with open('src/cuda-work/shakespeare.txt', 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print("Shakespeare text not found. Using dummy text for demonstration.")
        text = "To be or not to be that is the question. " * 1000
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Model hyperparameters (match CUDA kernel)
    vocab_size = tokenizer.vocab_size  # Should be 65
    embed_dim = 384
    num_heads = 6
    num_layers = 6
    block_size = 128
    batch_size = 4  # Smaller for demo
    learning_rate = 1e-4
    epochs = 3
    
    # Create model
    model = ShakespeareTransformer(vocab_size, embed_dim, num_heads, num_layers)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    dataset = ShakespeareDataset(text, tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            logits = model(x)  # [batch, seq, vocab]
            
            # Flatten for cross entropy
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Save the trained weights
    save_weights_binary(model, "shakespeare_weights.bin")
    
    # Test generation
    print("\n=== Testing Generation ===")
    model.eval()
    prompt = "To be or not to be"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(50):  # Generate 50 characters
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
            # Keep only the last 128 tokens (block_size)
            if input_ids.size(1) > 128:
                input_ids = input_ids[:, -128:]
    
    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    train_model() 