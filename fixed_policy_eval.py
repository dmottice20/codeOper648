import numpy as np

###########################################
##   PART 0: Initialize & Preprocess     ##
###########################################

# Define decision epochs...
N = 4
T = np.arange(N-1)

# Define states...
M = 8
S = np.arange(M)
cardS = len(S)

# Define actions -- or arcs essentially
A = dict()
cardA = dict()
for s in S:
    if s == 1-1:
        A[s] = set([2-1, 3-1, 4-1])
    elif s == 2-1:
        A[s] = set([5-1, 6-1])
    elif s == 3-1:
        A[s] = set([5-1, 6-1, 7-1])
    elif s == 4-1:
        A[s] = set([7-1])
    else:
        A[s] = set([8-1])

    cardA[s] = len(A[s])    

# Define some problem data...
C = np.zeros((M,M))
C[0,1] = 3
C[0,2] = 4
C[0,3] = 2
C[1,4] = 4
C[1,5] = 5
C[2,4] = 5
C[2,5] = 6
C[2,6] = 1
C[3,6] = 2
C[4,7] = 1
C[5,7] = 2
C[6,7] = 6

# Probability of actually moving to that node if selected...
q = 0.8

# Create transition probability matrices and reward vectors...
# and decision rule
P = dict()
r = dict()
d = dict()

for a in range(0,M):
    P[a] = np.zeros((cardS, cardS))
    r[a] = np.zeros((cardS, 1))



for t in T:
    d[t] = np.zeros((cardS, 1))

for s in S:
    for a in A[s]:
        for j in S:
            if len(A[s]) == 1:
                if j == a:
                    P[a][s,j] = 1
            else:
                if j == a:
                    P[a][s,j] = q
                elif j in A[s]:
                    P[a][s,j] = (1-q) / (cardA[s]-1)
        
        # Compute expected immediate reward
        r[a][s] = np.matmul(P[a][s,:], C[s,:].transpose())


##############################################
##   PART 1: Implement FixedPolicy Eval     ##
##############################################

# Initialize a dictionary of expeted rewards
u = dict()

# Instantiate the time counter at N
t = N-1

# Intiailize the expeected rewards (terminal rewards) at N
# to be an arry of zeros
u[t] = np.zeros((cardS, 1))
# Steps 2 --> 4
while t > 0:
    t -=1
    u[t] = np.zeros((cardS, 1))
    for s in S:
        a = max(A[s])
        d[t][s] = a
        u[t][s] = r[a][s] + np.matmul(P[a][s][:], u[t+1])
