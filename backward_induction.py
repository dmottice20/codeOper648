import numpy as np

################################################
#  PART 0 - Initialization & Preprocessing     #
################################################
# Decision epochs...
N = 10
T = np.arange(N - 1)
# Set of nodes...
M = 8
S = np.arange(M)
cardS = len(S)
# Set of actions...
A = dict()
cardA = dict()

for s in S:
    if s == 1-1:
        A[s] = [1, 2, 3]
    elif s == 2-1:
        A[s] = [4, 5]
    elif s == 3-1:
        A[s] = [4, 5, 6]
    elif s == 4-1:
        A[s] = [6]
    else:
        A[s] = [7]

    cardA[s] = len(A[s])

# Instantiate some problem data...
C = np.zeros((M, M))
C[0, 1] = 3
C[0, 2] = 4
C[0, 3] = 2
C[1, 4] = 4
C[1, 5] = 5
C[2, 4] = 5
C[2, 5] = 6
C[2, 6] = 1
C[3, 6] = 2
C[4, 7] = 1
C[5, 7] = 2
C[6, 7] = 6
C = -C

# Probability of moving to selected node...
q = 0.8

# Initialize transition probability matrices and reward vectors
P = dict()
r = dict()

for a in range(0, M):
    P[a] = np.zeros((cardS, cardS))
    r[a] = np.zeros((cardS, 1))

# Loop through current states...
for s in S:
    # Loop through actions in A_s...
    for a in A[s]:
        # Loop through next states...
        for j in S:
            if len(A[s]) == 1:
                if j == a:
                    P[a][s, j] = 1
            else:
                if j == a:
                    P[a][s, j] = q
                elif j in A[s]:
                    P[a][s, j] = (1 - q) / (cardA[s] - 1)

        r[a][s] = np.matmul(P[a][s][:], C[s][:].transpose())

for p in P.keys():
    print(f"On index {p}, the matrix is\n\n{P[p]}")
    print(P[p].shape)

# Initialize decision rules
d_star = dict()
for t in T:
    d_star[t] = np.zeros((cardS, 1))

################################################
#  PART 1 - Running of Backward Induction      #
################################################

# Step 1
u_star = dict()
t = N - 1
u_star[t] = np.zeros((cardS, 1))

# Steps 2-4
while t > 0:
    t -= 1
    print(f"on t={t}...")
    u_star[t] = np.zeros((cardS, 1))
    for s in S:
        u_t_best = -1000000
        a_star = None
        for a in A[s]:
            u_t = r[a][s] + np.matmul(P[a][s][:], u_star[t + 1])
            if u_t > u_t_best:
                u_t_best = u_t
                a_star = a

        u_star[t][s] = r[a_star][s] + np.matmul(P[a_star][s][:], u_star[t + 1])
        d_star[t][s] = a_star

print("\n\n========POLICY VALUES========\n\n")
