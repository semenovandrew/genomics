import numpy as np


def viterbii(seq, pi, e, t, m, l):
    v = np.zeros((l, m))
    psi = np.zeros((l, m))  # нужен далее для отыскания скрытой последовательности
    for j in range(m):
        v[0, j] = e[j][seq[0][1]] * pi[j]
    for i in range(1, l):
        for j in range(m):
            v[i, j] = max(v[i - 1][0] * t[0][j], v[i - 1][1] * t[1][j]) * e[j][seq[i][1]]  # перемножаются строка и стоблец

    return v, psi
