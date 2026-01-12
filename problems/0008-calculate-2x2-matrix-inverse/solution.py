import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    m = torch.as_tensor(matrix, dtype=torch.float)
    if torch.isclose(torch.det(m), torch.tensor(0.0)):
        return None
    return torch.linalg.inv(m)
