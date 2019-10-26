import numpy as np


def forward(obs, pi, t, e, m, l):
    fwd = np.zeros((l, m))

    fwd[0] = pi * e[:, obs[0][1]]

    for i in range(1, l):
        for l in range(m):
            for k in range(m):
                fwd[i, l] += fwd[i - 1, k] * t[k, l]
            fwd[i, l] *= e[l, obs[i][1]]

    return fwd


def likelihood(obs_seq, pi, t, e, m, l):
    # returns log P(Y  \mid  model)
    # using the forward part of the forward-backward algorithm
    return forward(obs_seq, pi, t, e, m, l)[-1].sum()


def backward(obs, t, e, m, l):
    bwd = np.zeros((l, m))

    for j in range(m):
        bwd[l - 1, j] = 1

    for i in range(l - 2, -1, -1):
        for k in range(m):
            for l in range(m):
                bwd[i, k] += bwd[i + 1, l] * t[k, l] * e[l, obs[i + 1][1]]

    return bwd

def pbwd(obs, t, e, m, l):
    bwdp = 0
    bwd = backward(obs, t, e, m, l)
    for l in range(m):
        bwdp += bwd[1, l] * t[0, l] * e[l, obs[1][1]]
    return bwdp

def posterior_prob(obs, pi, t, e, m, l):
    ps = np.zeros((l, m))
    forw = forward(obs, pi, t, e, m, l)
    backw = backward(obs, t, e, m, l)
    like = likelihood(obs, pi, t, e, m, l)
    for i in range(l):
        for j in range(m):
            ps[i, j] = forw[i][j] * backw[i][j] / like
    return ps