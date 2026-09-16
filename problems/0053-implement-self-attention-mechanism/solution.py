import torch


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
):
    """Compute Query, Key, Value matrices from input X and weight matrices."""

    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)

    return Q, K, V


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.
    """

    # 1. Q 和所有 K 计算匹配分数
    scores = torch.matmul(Q, K.T)

    # 2. 缩放
    d_k = Q.shape[-1]
    scores = scores / (d_k**0.5)

    # 3. 把分数变成注意力权重
    attention_weights = torch.softmax(scores, dim=-1)  # dim=-1,按行做softmax

    # 4. 根据权重获取 V 中的信息
    output = torch.matmul(attention_weights, V)

    return output
