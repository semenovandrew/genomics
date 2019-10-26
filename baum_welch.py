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
    for y in range(l):
        for i in range(m):
            summ = 0
            for j in range(m):
                summ += alpha[y, j] * beta[y, j]
            g[y, i] = alpha[y, i] * beta[y, i] / summ
    return g


def epsillon(obs, pi, t, e, m, l):

    alpha = forward(obs, pi, t, e, m, l)
    beta = backward(obs, t, e, m, l)
    eps = np.zeros((l, m, m))

    summ = 0
    for k in range(l):
        summ = 0
        for i in range(m):
            for j in range(m):
                summ += alpha[k - 1, i] * t[i, j] * beta[k, j] * e[j, obs[k][1]]
        for i in range(m):
            for j in range(m):
                eps[k - 1, i, j] = (alpha[k - 1, i] * t[i, j] * beta[k, j] * e[j, obs[k][1]]) / summ

    return eps


def baum_welch0(obs, pi, t, e, m, l):
    global transition_matrix, emission_matrix
    gamm = gamma(obs, pi, t, e, m, l)
    epsi = epsillon(obs, pi, t, e, m, l)

    for i in range(m):
        pi[i] = gamm[0, i]

    sumeps = 0
    sumgam = 0
    for i in range(m):
        for j in range(m):
            sumeps = 0
            sumgam = 0
            for k in range(l - 1):
                sumeps += epsi[k, i, j]
                sumgam += gamm[k, i]
            t[i, j] = sumeps / sumgam

    summg = 0
    summep = 0
    for i in range(m):
        for j in range(l):
            summgall = 0
            summg = 0
            for k in range(l):
                if obs[k][1] == obs[j][1]:
                    delta = 1
                else:
                    delta = 0
                summg += delta * gamm[k, i]
                summgall += gamm[k, i]
            e[i, obs[j][1]] = summg / summgall

    return t,  e

