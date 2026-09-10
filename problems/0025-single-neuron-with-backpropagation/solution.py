import numpy as np
import math


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> (np.ndarray, float, list[float]):
    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []
    while epochs:
        probabilities = []
        for feature in features:
            weighted_sum = 0
            for i in range(len(feature)):
                weighted_sum += feature[i] * weights[i]
            weighted_sum += bias
            weighted_sum = 1 / (1 + math.exp(-weighted_sum))
            probabilities.append(weighted_sum)
        mse = 0
        for i in range(len(labels)):
            mse += (probabilities[i] - labels[i]) ** 2
        mse = mse / len(labels)
        mse_values.append(round(mse, 4))

        weight_gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for i in range(len(labels)):
            y = labels[i]
            y_hat = probabilities[i]
            dz = 2 * (y_hat - y) * y_hat * (1 - y_hat)
            for j in range(len(weights)):
                weight_gradients[j] += dz * features[i][j]
            bias_gradient += dz
        n = len(labels)
        for j in range(len(weights)):
            weight_gradients[j] /= n
        bias_gradient /= n
        for j in range(len(weights)):
            weights[j] -= learning_rate * weight_gradients[j]
        bias -= learning_rate * bias_gradient
        epochs -= 1
    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)
    return updated_weights, updated_bias, mse_values
