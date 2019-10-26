import numpy as np
from viterbi import viterbii


#   forward algorithm
def baum_forward(obs, pi, t, e, m, l):
    alpha = np.zeros((l, m))

    for k in range(m):  # задаем начальные параметры для дальнейшего их улучшения
        alpha[0, k] = pi[k] * e[k, obs[0]]

#   рекурсивно получаем следющиее элементы aplha
    for i in range(1, l):
        for l in range(m):
            for k in range(m):
                alpha[i, l] += alpha[i - 1, k] * t[k, l]
            alpha[i, l] *= e[l, obs[i]]

    return alpha


#   define the P(x)
def prob_x(obs, pi, t, e, m, l):
    # returns log P(Y  \mid  model)
    # using the forward part of the forward-backward algorithm
    return baum_forward(obs, pi, t, e, m, l)[-1].sum()


#   backward algorithm
def baum_backward(obs, t, e, m, l):
    beta = np.zeros((l, m))

    for k in range(m):
        beta[l - 1, k] = 1

    for i in range(l - 2, -1, -1):
        for k in range(m):
            for l in range(m):
                beta[i, k] += beta[i + 1, l] * t[k, l] * e[l, obs[i + 1]]

    return beta


#   update variables
def baum_post(obs, pi, t, e, m, l):
    alpha = baum_forward(obs, pi, t, e, m, l)
    beta = baum_backward(obs, t, e, m, l)
    prob = prob_x(obs, pi, t, e, m, l)
    posterior = np.zeros((l, m))

    for i in range(l):
        for k in range(m):
            posterior[i, k] = alpha[i, k] * beta[i, k] / prob
    return posterior


def gamma(obs, pi, t, e, m, l):
    alpha = baum_forward(obs, pi, t, e, m, l)
    beta = baum_backward(obs, t, e, m, l)
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
    alpha = baum_forward(obs, pi, t, e, m, l)
    beta = baum_backward(obs, t, e, m, l)
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


def new_param(obs, pi, t, e, m, l):
    gam = gamma(obs, pi, t, e, m, l)
    epsi = epsillon(obs, pi, t, e, m, l)
    summgam = 0
    epsisumm = 0
    for i in range(l - 1):
        summgam += gam[i]
        epsisumm += epsi[i]
    a = epsisumm / summgam

    return a


def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    # forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # backward part of the algorithm
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    assert p_fwd == p_bkw
    return fwd, bkw, posterior


