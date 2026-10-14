import torch


def ddpm_linear_schedule(beta_start: float, beta_end: float, T: int) -> dict:
    """
    Compute the DDPM linear noise schedule and derived quantities using PyTorch.

    Args:
        beta_start: Starting value of the beta (noise variance) schedule.
        beta_end: Ending value of the beta schedule.
        T: Number of diffusion timesteps.

    Returns:
        Dictionary with keys: 'betas', 'alphas', 'alpha_bars',
        'sqrt_alpha_bars', 'sqrt_one_minus_alpha_bars'.
        All values are torch.Tensor of length T.
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
    }
