import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform Tâ»Â¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2Ã2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    if torch.det(S_t) == 0 or torch.det(T_t) == 0:
        return torch.tensor(-1.)
    out=torch.linalg.inv(T_t) @ A_t @ S_t
    return torch.round(out*1000)/1000
    pass

# A = [[1, 2], [3, 4]]
# T = [[2, 0], [0, 2]]
# S = [[1, 1], [0, 1]]
# print(transform_matrix(A,T,S))
