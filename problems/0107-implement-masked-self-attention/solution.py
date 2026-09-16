import torch


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)


def masked_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    scores = torch.matmul(Q, K.T)
    d_k = Q.shape[-1]
    scores = scores / (d_k**0.5)
    scores = scores + mask
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output
