import torch


def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model), dtype float16
        or -1 if position == 0 or d_model <= 0
    """
    if position == 0 or d_model <= 0:
        return -1
    pe = torch.zeros(position, d_model)
    pos = torch.arange(position).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term[: pe[:, 1::2].shape[1]])
    return pe.to(torch.float16)
