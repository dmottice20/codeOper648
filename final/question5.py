import numpy as np
import itertools

# T={1,2,...}
# S={1,2,3,4} X {1,2,3,4}
S = np.array(list(itertools.product(np.arange(4), np.arange(4))))
# S is now of the form...
# [[0 0],
#  [0, 1]...]
# where the first index represents the site
# of the worker while the second represents
# the site of the trailer.

A = np.arange(4)
print(A)
card_s = S.shape[0]
card_a = len(A)
P = np.zeros((card_a, card_s, card_s))

s_idx = 0
for s in S:
    # s --> [worker, trailer]
    # s[0] - worker site
    # s[1] - current trailer site
    for a in A:
        j_idx = 0
        for j in S:
            # j --> [worker, trailer]
            # j[0] - next worker site
            # j[1] - next trailer site
            # Action's do not affect probabilities
            if j[1] == a:
                if s[0] == 0:
                    if j[0] == 0:
                        P[a][s_idx, j_idx] = 0.1
                    elif j[0] == 1:
                        P[a][s_idx, j_idx] = 0.3
                    elif j[0] == 2:
                        P[a][s_idx, j_idx] = 0.3
                    elif j[0] == 3:
                        P[a][s_idx, j_idx] = 0.3
                elif s[0] == 1:
                    if j[0] == 1:
                        P[a][s_idx, j_idx] = 0.5
                    elif j[0] == 2:
                        P[a][s_idx, j_idx] = 0.5
                elif s[0] == 2:
                    if j[0] == 2:
                        P[a][s_idx, j_idx] = 0.8
                    elif j[0] == 3:
                        P[a][s_idx, j_idx] = 0.2
                elif s[0] == 3:
                    if j[0] == 0:
                        P[a][s_idx, j_idx] = 0.4
                    elif j[0] == 3:
                        P[a][s_idx, j_idx] = 0.6

            j_idx += 1

    s_idx += 1


r_move = np.zeros((card_a, card_s))
r_obtain = np.zeros((card_a, card_s))
s_idx = 0
for s in S:
    for a in A:
        if s[1] != a:
            r_move[a][s_idx] = -300
        
        if s[0] > 0:
            if s[1] == 0:
                r_obtain[a][s_idx] = -200
            elif s[1] != s[0]:
                r_obtain[a][s_idx] = -100
            elif s[1] == s[0]:
                r_obtain[a][s_idx] = -50            

    s_idx += 1

r = r_move + r_obtain

# Implement policy iteeration at a discount of 0.95
lamb = 0.95
d = np.zeros((card_s, 1))
dtm1 = d.copy()

s_idx = 0
for s in S:
    QsaBest = -np.inf
    for a in A:
        Qsa = r[a][s_idx]
        if Qsa > QsaBest:
            QsaBest = Qsa
            d[s] = a

    s_idx += 1

n = 0
rd = 0*r[0].reshape(16,1)
v = 0
Pd = np.zeros((card_s, card_s))

while not np.array_equal(d, dtm1):
    dtm1 = d.copy()
    s_idx = 0
    for s in S:
        Pd[s_idx, :] = P[int(d[s_idx])][s_idx, :]
        rd[s_idx, :] = r[int(d[s_idx])][s_idx]

        s_idx += 1

    v = np.linalg.solve(np.identity(card_s) - lamb * Pd, rd)

    s_idx = 0
    for s in S:
        QsaBest = -np.inf
        for a in A:
            Qsa = r[a][s_idx] + lamb * P[a][s_idx, :]@v
            if Qsa > QsaBest:
                QsaBest = Qsa.copy()
                d[s_idx] = a

        s_idx += 1

    n += 1

print("=======DECISION RULE=========")
print(d)

# For Part (C)

# Solve for the long run probability of being in each state...
def calculate_limiting_distribution(Pd):
    """
    :param Pd - induced DTMC from optimal policy
    :return pi_d - limiting distributino of Pd
    """
    # Right now -- this is for irreducible which is wrong.
    A_top_row = np.identity(Pd.shape[0])-Pd.transpose()
    A_bottom_row = np.repeat(1, Pd.shape[0])
    A = np.vstack((A_top_row, A_bottom_row))
    A = np.delete(A, 0, axis=0)
    b = np.vstack((np.asarray([np.repeat(0, Pd.shape[0]-1)]).transpose(), [1]))
    pi = np.matmul(np.linalg.inv(A), b)
    
    #pi = np.linalg.matrix_power(Pd, 1000)
    
    
    return pi

# THIS CODE IS OPEN SOURCE:
# https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

pi = calculate_limiting_distribution(Pd)

# If the optimal policy is followed, what state is the sytem
# most likely to be found upon inspection?

# This is just what state has the highest long run probability 
# and return its index --> i.e. the state.
state = np.argmax(pi)
print("The index of the state with the highest LR prob is...", state)
print("This corresponds to...", S[state])
print("The probability of being in that state is...", np.max(pi))
#print(bmatrix(pi))
print(S)
print(pi)