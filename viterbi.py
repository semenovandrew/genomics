import numpy as np


def viterbii(seq, pi, e, t, m, l):
    v = np.zeros((l, m))
    psi = np.zeros((l, m))  # нужен далее для отыскания скрытой последовательности
    for j in range(m):
        v[0, j] = e[j, seq[0][1]] * pi[j]
    for i in range(1, l):
        for j in range(m):
            v[i, j] = max(v[i - 1] * t[:, j]) * e[j, seq[i][1]]  # перемножаются строка и стоблец
            psi[i, j] = np.argmax(v[i - 1] * t[:, j])  # перемножаются стока и стоблец

    return v, psi


