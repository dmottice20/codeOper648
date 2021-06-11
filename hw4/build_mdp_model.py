import numpy as np
import os
import timeit

start = timeit.timeit()
# Read in data
#file_name = os.path.join("data", "ARP_data.txt")
data = np.loadtxt("ARP_data", delimiter=",")

# Split columns into representing vectors.
c, v, o, p = np.hsplit(data, 4)

# Construct the state space.
S = np.arange(1, 120+1)
card_s = len(S)

# Construct the action space.
A = np.arange(1, 121+1)
card_a = len(A)

# Interpretation:
# a = 1, a in A --> action of keeping present car for another month
# {a in A : a > 1} --> action of buying a car of age a - 2

# Construct transition probability matrices and reward vectors
P = np.zeros((card_a, card_s, card_s))
r = np.zeros((card_a, card_s, 1))

for s in S:
    for a in A:
        for j in S:
            # CONSTRUCT P Matrices.
            if a == 1:
                if s == 120:
                    if j == 120:
                        P[a-1][s-1, j-1] = 1
                elif s == 119:
                    if j == 120:
                        P[a-1][s-1, j-1] = 1
                else:
                    if j == s + 1:
                        P[a-1][s-1, j-1] = p[s]
                    elif j == 120:
                        P[a-1][s-1, j-1] = 1 - p[s]
            elif a == 121:
                if j == 120:
                    P[a-1][s-1, j-1] = 1
            else:
                if j == a - 2 + 1:
                    P[a-1][s-1, j-1] = 1 * p[a-2]
                elif j == 120:
                    P[a-1][s-1, j-1] = 1 * (1 - p[a-2])

        if a == 1:
            r[a-1, s-1] = -o[s]
        else:
            r[a-1, s-1] = v[s] - c[a-2] - o[a-2]

end = timeit.timeit()

print("Loading time for problem data is...", end-start)

# Save the transition probability matrix.
with open("data/transition_matrix.npy", "wb") as f:
    np.save(f, P)

# Save the reward vector.
with open("data/reward_vector.npy", "wb") as f:
    np.save(f, r)
