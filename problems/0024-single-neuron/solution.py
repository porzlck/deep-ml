import math


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> (list[float], float):
    probabilities = []
    for feature in features:
        weighted_sum = 0
        for j in range(len(feature)):
            weighted_sum += feature[j] * weights[j]
        weighted_sum += bias
        probability = 1 / (1 + math.exp(-weighted_sum))
        probabilities.append(probability)
    mse = 0
    for i in range(len(labels)):
        mse += (probabilities[i] - labels[i]) ** 2
    mse = mse / len(labels)
    probabilities = [round(p, 4) for p in probabilities]
    mse = round(mse, 4)

    return probabilities, mse
