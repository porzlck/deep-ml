import numpy as np


class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.

        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.

        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if not training:
            return x

        self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)

        return x * self.mask / (1 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.

        Use the stored self.mask from the most recent forward pass.
        """
        if self.mask is None:
            return grad
        return grad * self.mask / (1 - self.p)
