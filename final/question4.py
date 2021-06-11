import numpy as np
from numpy.testing._private.utils import clear_and_catch_warnings


# Build the model...
# T = {1,2,3,4}
T = np.arange(1,4+1)
# S = {0,1,2,3,4,5,6,7,8}
S = np.arange(8+1)
# A_s = 0,1,2,...,s
A = dict()
for t in T:
    A[t] = dict()
    for s in S:
        if t == 1:
            A[t][s] = np.arange(min(3, s)+1)
        elif t == 2:
            A[t][s] = np.arange(min(4, s)+1)
        else:
            A[t][s] = np.arange(min(3, s)+1)

"""for t in T:
    print('t={}'.format(t))
    print(A[t])"""

# Build p and r
card_a = 4
card_s = len(S)
P = dict()
r = dict()

for a in np.arange(card_a+1):
    P[a] = np.zeros((card_s, card_s))
    for s in S:
        for j in S:
            if j == s - a:
                P[a][s, j] = 1

print(P.keys())
for a in P.keys():
    r[a] = np.zeros((len(T), 1))
    for t in T:
        if t == 1:
            r[a][t] = a**3
        if t == 2:
            r[a][t] = 3*(a**2)
        if t== 3:
            r[a][t] = 9*a

print(r)

# Solve using backwards induction...
t = T[-1]

# There are no terminal rewards...
u_star = dict()
u_star[t] = np.zeros((card_s, 1))
d_star = dict()

while t > 1:
    t -= 1
    d_star[t] = np.zeros((card_s, 1))
    u_star[t] = np.zeros((card_s, 1))
    for s in S:
        utsaBest = -np.inf
        for a in A[t][s]:
            utsa = r[a][t] + P[a][s, :]@u_star[t+1]
            if utsa > utsaBest:
                utsaBest = utsa
                d_star[t][s] = a
            
        u_star[t][s] = utsaBest

print("\n========DECISION RULE========")
for t in d_star.keys():
    print('\nfor t = {}'.format(t))
    print('the decision rule is...\n',d_star[t])
print("\n========OPTIMAL VALUE FX========")
for t in u_star.keys():
    print('\nfor t = {}'.format(t))
    print('the optimal value fx is...\n',u_star[t])