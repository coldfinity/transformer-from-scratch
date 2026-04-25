from typing import Any

import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_len = sequence_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (sequence length, d_model)
        pe = torch.zeros(sequence_len, d_model)

        # Create a tensor of shape (sequence length, 1)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)

        # Calculate in log spcae for numerical stability
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Sine for even positions
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)

        # Add batch of sentences
        pe = pe.unsqueeze(0)  # (1, sequence length, d_model)

        # Buffer of module
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1))  # Additive

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # Input: (Batch, sequence, d_model) -> (Batch, sequence, d_ff) -> (Batch, sequence, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
