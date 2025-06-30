graph TD
    subgraph Transformer Block
        direction LR
        Input --> AddNorm1(Add & LayerNorm)
        subgraph Multi-Head Attention
            MHA_Internal(( ))
        end
        Input --> MHA(Multi-Head Attention) --> AddNorm1
        
        AddNorm1 --> AddNorm2(Add & LayerNorm)
        subgraph Feed Forward
            FFN_Internal(( ))
        end
        AddNorm1 --> FFN(Feed Forward) --> AddNorm2
        AddNorm2 --> Output
    end
