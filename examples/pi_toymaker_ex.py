import numpy as np
from scipy import sparse

S = np.arange(2)
card_s = len(S)

A = np.arange(2)
card_a = len(A)

P = np.empty((2, 2, 2))
P[0] = [[0.5, 0.5], [0.4, 0.6]]
P[1] = [[0.8, 0.2], [0.7, 0.3]]

R = np.empty((2, 2, 2))
r = dict()
R[0] = [[9, 3], [3, -7]]
R[1] = [[4, 4], [1, -19]]

r[0] = np.sum(P[0]*R[0], axis=1)
r[1] = np.sum(P[1]*R[1], axis=1)

print(r)

lamb = 0.9

d = np.zeros((card_s, 1))
dtm1 = d.copy()

for s in S:
    QsaBest = -np.inf
    for a in A:
        print(a)
        Qsa = r[s][a]
        print("Qsa is.. ", Qsa)
        if Qsa > QsaBest:
            QsaBest = Qsa
            d[s] = a

n = 0
rd = 0*r[1]
Pd = np.zeros((card_s, card_s))

vT = np.zeros((10, 2))

is_value_changed = True

while is_value_changed:
    is_value_changed = False
    print("BEFORE LOOP WORK")
    print("DTM1 is...\n", dtm1)
    print("D is...\n", d)
    dtm1 = d.copy()
    for s in S:
        print(s)
        Pd[s, :] = P[int(d[s][0])][s, :]
        rd[s] = r[int(d[s][0])][s]

    # Policy evaluation...
    v = np.linalg.solve(sparse.eye(card_s).toarray()-lamb*Pd, rd)
    # Record value fx
    vT[n, :] = v.transpose()

    # Policy improvement...
    for s in S:
        QsaBest = -np.inf
        for a in A:
            Qsa = r[a][s] + lamb * np.matmul(P[a][s, :], v)
            if Qsa > QsaBest:
                print("WELCOME!")
                print("Action ", a)
                QsaBest = Qsa
                d[s] = a
                is_value_changed = True

    n += 1

print("The number of iterations: ", n)

