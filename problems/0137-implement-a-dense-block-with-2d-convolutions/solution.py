import numpy as np


def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    x = input_data
    padding_h = kernel_size[0] // 2
    padding_w = kernel_size[1] // 2

    N = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    for _ in range(num_layers):
        new_feature = np.zeros((N, H, W, growth_rate))
        k = kernels[_]
        if k.shape[2] != x.shape[-1]:
            raise ValueError
        activated = np.maximum(x, 0)
        padded = np.pad(
            activated,
            ((0, 0), (padding_h, padding_h), (padding_w, padding_w), (0, 0)),
            mode="constant",
        )
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    cur = padded[n, h : h + kernel_size[0], w : w + kernel_size[1], :]
                    for c in range(growth_rate):
                        new_feature[n, h, w, c] = np.sum(cur * k[:, :, :, c])
        x = np.concatenate((x, new_feature), axis=-1)
    return x
