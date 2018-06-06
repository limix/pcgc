from __future__ import division

from brent_search import brent
import numpy as np
import scipy.stats as st


def pcgc_cost(y, kinship, c, h2):
    cost = 0
    n = len(y)
    for i in range(0, len(y) - 1):
        d = y[i] * y[i + 1:] - c * h2 * kinship[i, i + 1:]
        cost += np.sum(d**2) / n
    return cost


def pcgc(y, kinship, K):
    y = (y - np.mean(y)) / np.std(y)
    P = np.sum(y > 0) / len(y)
    t = st.norm.isf(K)
    c = P * (1 - P) * st.norm.pdf(t)**2 / (K**2 * (1 - K)**2)
    return brent(lambda h2: pcgc_cost(y, kinship, c, h2), 1e-4, 1 - 1e-4)[0]
