import numpy as np
import matplotlib.pyplot as plt

S = np.arange(2)
card_s = len(S)

A = {
    0: np.arange(2),
    1: np.arange(2)
}
card_a = 2
"""
P = {
    0: np.asarray([
        [0.5, 0.5],
        [0.4, 0.6]
    ]),
    1: np.asarray([
        [0.8, 0.2],
        [0.7, 0.3]
    ])
}"""
P = np.empty((2, 2, 2))
P[0] = [[0.5, 0.5],
        [0.4, 0.6]]
P[1] = [[0.8, 0.2],
        [0.7, 0.3]]
"""R = {
    0: np.asarray([
        [9, 3],
        [3, -7]
    ]),
    1: np.asarray([
        [4, 4],
        [1, -19]
    ])
}"""
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
    # if n != 0:
    #    del v_n
    v_n = v_np1.copy()
    for s in S:
        v_np1_sa_Best = -np.inf
        for a in A[s]:
            v_np1_sa = r[a][s] + lamb * np.matmul(P[a][s, :], v_n)
            if v_np1_sa > v_np1_sa_Best:
                v_np1_sa_Best = v_np1_sa

        v_np1[s] = v_np1_sa_Best

    conv_val = np.linalg.norm(v_np1 - v_n, np.inf)

    inRunResultsTable[n, :] = np.asarray([v_np1[0], conv_val])
    n += 1

Pd = np.zeros((card_s, card_s))
for s in S:
    v_np1_sa_Best = -np.inf
    for a in A[s]:
        v_np1_sa = r[a][s] + lamb * np.matmul(P[a][s, :], v_n)
        if v_np1_sa > v_np1_sa_Best:
            v_np1_sa_Best = v_np1_sa
            d[s] = a

    Pd[s, :] = P[int(d[s])][s, :]

# Plot the results...
plt.plot(np.arange(n), inRunResultsTable[inRunResultsTable[:, 0] != 0, 0], "bo", markersize=1)
plt.xlabel("Iteration (n)")
plt.ylabel("Present Equivalent ($)")
plt.title("Value of Successful Toy (State 1)")
plt.show()

plt.plot(np.arange(n), inRunResultsTable[inRunResultsTable[:, 1] > 0, 1], "ro", markersize=1)
plt.xlabel("Iteration (n)")
plt.ylabel("||v^n-v^{n+1}||")
plt.title("Convergence In Norm")
plt.show()
print(n)
print(v_n)
