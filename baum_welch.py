import numpy as np
from viterbi import viterbii


#   forward algorithm
def forward(obs, pi, t, e, m, l):
    alpha = np.zeros((l, m))

    for k in range(m):  # задаем начальные параметры для дальнейшего их улучшения
        alpha[0, k] = pi[k] * e[k, obs[0][1]]

#   рекурсивно получаем следющиее элементы aplha
    for i in range(1, l):
        for l in range(m):
            for k in range(m):
                alpha[i, l] += alpha[i - 1, k] * t[k, l]
            alpha[i, l] *= e[l, obs[i][1]]

    return alpha


#   define the P(x)
def prob_x(obs, pi, t, e, m, l):
    # returns log P(Y  \mid  model)
    # using the forward part of the forward-backward algorithm
    return forward(obs, pi, t, e, m, l)[-1].sum()


#   backward algorithm
def backward(obs, t, e, m, l):
    beta = np.zeros((l, m))

    for k in range(m):
        beta[l - 1, k] = 1

    for i in range(l - 2, -1, -1):
        for k in range(m):
            for l in range(m):
                beta[i, k] += beta[i + 1, l] * t[k, l] * e[l, obs[i + 1][1]]

    return beta


#   update variables
def baum_post(obs, pi, t, e, m, l):
    alpha = forward(obs, pi, t, e, m, l)
    beta = backward(obs, t, e, m, l)
    prob = prob_x(obs, pi, t, e, m, l)
    posterior = np.zeros((l, m))

    for i in range(l):
        for k in range(m):
            posterior[i, k] = alpha[i, k] * beta[i, k] / prob
    return posterior


def gamma(obs, pi, t, e, m, l):
    alpha = forward(obs, pi, t, e, m, l)
    beta = backward(obs, t, e, m, l)
    g = np.zeros((l, m))
    summ = 0
    for t in range(l):
        for i in range(m):
            summ = 0
            for j in range(m):
                summ += alpha[t, j] * beta[t, j]
            g[t, i] = alpha[t, i] * beta[t, i] / summ
    return g


def epsillon(obs, pi, t, e, m, l):
    alpha = forward(obs, pi, t, e, m, l)
    beta = backward(obs, t, e, m, l)
    eps = np.zeros((l, m, m))
    summ = 0

    for k in range(l):
        for i in range(m):
            summ = 0
            for j in range(m):
                summ += alpha[k - 1, i] * t[i, j] * beta[k, j] * e[j, obs[k][1]]
            summ += alpha[k - 1, i] * t[i, j] * beta[k, j] * e[j, obs[k][1]]
            for j in range(m):
                eps[k, i, j] = alpha[k - 1, i] * t[i, j] * beta[k, j] * e[j, obs[k][1]] / summ

    return eps


def baum_welch(obs, pi, t, e, m, l, iters):

    for n in range(iters):

        gamm = gamma(obs, pi, t, e, m, l)
        epsil = epsillon(obs, pi, t, e, m, l)

        #   new parameters
        summgam = 0
        summeps = 0
        for i in range(l - 1):
            summgam += gamm[i]
            summeps += epsil[i]
        t = summeps / summgam

        summgam = 0
        summgamall = 0
        for k in range(l):
            summgamall += gamm[k]
        for i in range(m + 1):
            for j in range(m):
                summgam = 0
                if obs[j][1] == i:
                    for t in range(l):
                        summgam += gamm[t]
                    e[j] = summgam / summgamall

    return t, e
