import numpy as np
import matplotlib.pyplot as plt

S = np.arange(2)
card_s = len(S)

A = {
    0: np.arange(2),
    1: np.arange(2)
}
card_a = 2

P = np.empty((2, 2, 2))
P[0] = [[0.5, 0.5],
        [0.4, 0.6]]
P[1] = [[0.8, 0.2],
        [0.7, 0.3]]

R = np.empty((2, 2, 2))
R[0] = [[9, 3],
        [3, -7]]
R[1] = [[4, 4],
        [1, -19]]
r = np.empty((2, 2, 1))
r[0] = np.asarray([
    [6],
    [-3]
])
r[1] = np.asarray([
    [4],
    [-5]
])

v_np1 = np.zeros((card_s, 1))
d = np.zeros((card_s, 1))

epsilon = 0.01
lamb = 0.9

conv_test = epsilon * (1 - lamb) / (2 * lamb)
conv_val = np.inf

n = 0
max_iter = 500

inRunResultsTable = np.zeros((max_iter, 2))

while (conv_val >= conv_test) and (n < max_iter):
    v_n = v_np1.copy()
    for s in S:
        v_np1_sa_Best = -np.inf
        for a in A[s]:
            v_np1_sa = r[a][s] + lamb * np.matmul()