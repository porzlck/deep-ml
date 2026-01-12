import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2Ã2 matrix using PyTorch.
    Input: 2Ã2 tensor; Output: 1-D tensor with the two eigenvalues in ascending order.
    """
    # Your implementation here
    a = torch.as_tensor(matrix, dtype=torch.float)

    tr = a[0, 0] + a[1, 1]
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    disc = tr * tr - 4 * det

    r = torch.sqrt(disc)
    e1 = (tr + r) / 2
    e2 = (tr - r) / 2

    return torch.sort(torch.stack([e1, e2]), descending=False).values
	pass
