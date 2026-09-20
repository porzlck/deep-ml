import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    h = np.array(initial_hidden_state, dtype=float)
    Wx = np.array(Wx, dtype=float)
    Wh = np.array(Wh, dtype=float)
    b = np.array(b, dtype=float)
    for x in input_sequence:
        x = np.array(x, dtype=float)
        h = np.tanh(Wx @ x + Wh @ h + b)
    final_hidden_state = np.round(h, 4).tolist()
    return final_hidden_state
