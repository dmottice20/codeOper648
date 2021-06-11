import numpy as np

# Build the formulation...
# S = {1,2,3,4}
S = np.arange(4)
# A_s = {1: 0, 2: (0,1), 3:(0,1), 4:1}
A = dict()
for s in S:
    if s == 0:
        A[s] = np.array([0])
    elif s == 3:
        A[s] = np.array([1]) 
    else:
        A[s] = np.array([0, 1])

card_s = len(S)

# Build P and R...
P = dict()
r = dict()
card_a = 2

for a in np.arange(card_a):
    P[a] = np.zeros((card_s, card_s))
    r[a] = np.zeros((card_s, 1))

for s in S:
    for a in A[s]:
        for j in S:
            if a == 1:
                if j == 0:
                    P[a][s, j] = 1
            else:
                if s == 0:
                    if j == 0:
                        P[a][s, j] = 0.7
                    elif j == 1:
                        P[a][s, j] = 0.3
                elif s == 1:
                    if j == 1:
                        P[a][s, j] = 0.8
                    elif j == 2:
                        P[a][s, j] = 0.2
                elif s == 2:
                    if j == 2:
                        P[a][s, j] = 0.9
                    elif j == 3:
                        P[a][s, j] = 0.1

        if a == 1:
            if s == 1:
                r[a][s] = 10
            elif s == 2:
                r[a][s] = 30
            elif s == 3:
                r[a][s] = 80

# Double check P's and r...
for a in P.keys():
    print('a={}'.format(a))
    print('P is...\n', P[a])
    print('r is...\,', r[a])

# Solve using policy iteration...
d = np.zeros((card_s, 1))
dtm1 = d.copy()

for s in S:
    QsaBest = -np.inf
    for a in A[s]:
        Qsa = r[a][s]
        if Qsa > QsaBest:
            QsaBest = Qsa
            d[s] = a

lamb = 0.92

n = 0
rd = 0*r[0]
v = 0
Pd = np.zeros((card_s, card_s))

while not np.array_equal(d, dtm1):
    dtm1 = d.copy()
    for s in S:
        Pd[s, :] = P[int(d[s])][s, :]
        rd[s, :] = r[int(d[s])][s]

    # Policy evaluation...
    v = np.linalg.solve(np.identity(card_s)-lamb*Pd,rd)

    # Policy improvement...
    for s in S:
        QsaBest = -np.inf
        for a in A[s]:
            Qsa = r[a][s] + lamb * P[a][s, :]@v
            if Qsa > QsaBest:
                QsaBest = Qsa.copy()
                d[s] = a
    
    n += 1

print(d)
print(v)