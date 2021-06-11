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
max_iter = 1500

inRunResultsTable = np.zeros((max_iter, 2))

while True:
    n += 1
    v_n = v_np1.copy()
    S = S[::-1]
    for s in S:
        Q = [float(r[a][s] + lamb * P[a][s, :].dot(v_n)) for a in A[s]]
        v_n[s] = max(Q)

    conv_val = np.linalg.norm(v_np1 - v_n, np.inf)
    if conv_val < conv_test:
        print("EPSILON OPTIMAL")
        break
    elif n == max_iter:
        print("Break due to max iterations reached.")
        break

policy = []
for s in S:
    Q = 0*A[s]
    for a in A[s]:
        Q[a] = (r[a][s] + lamb * P[a][s, :].dot(v_n))

    v_n[s] = Q.max()
    policy.append(int(Q.argmax()))