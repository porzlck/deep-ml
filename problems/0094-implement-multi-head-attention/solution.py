import torch
import torch.nn.functional as F
from typing import Tuple


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    scores = torch.matmul(Q, K.T)
    d_k = Q.shape[-1]
    scores = scores / (d_k**0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    outputs = torch.matmul(attention_weights, V)
    return outputs


def multi_head_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int
) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    head_dim = d_model // n_heads

    Q = Q.reshape(seq_len, n_heads, head_dim).transpose(0, 1)
    K = K.reshape(seq_len, n_heads, head_dim).transpose(0, 1)
    V = V.reshape(seq_len, n_heads, head_dim).transpose(0, 1)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / (head_dim**0.5)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    output = output.transpose(0, 1)
    output = output.reshape(seq_len, d_model)
    return output
