import numpy as np

################################################
##  PART 0 - Initialization & Preprocessing   ##
################################################

# Decision epochs...
N = 4
T = np.arange(N-1)

# State space...
M = 3
S = np.arange(M+1)
cardS = len(S)

# Action space (state-dependent)
A = dict()
for s in S:
    A[s] = np.arange(M-s+1)

print("The action space is...\n", A)

# Instantiate and fill the problem data
# Fixed ordering cost, unit ordering and holding cost, and revenue respectively
K = 4
c = 2
h = 1
f = 8
# Demand random variable
D = np.asarray([np.arange(M)]).transpose()
p = np.asarray([0.25, 0.5, 0.25]).transpose()
cardD = len(D)

# Compute reward vectors & transition probability matrices
P = dict()
r = dict()

for a in np.arange(M+1):
    P[a] = np.zeros((cardS, cardS))
    r[a] = np.zeros((cardS, 1))

for s in S:
    for a in A[s]:
        # Compute s_{t+1}
        s_tp1 = np.maximum(s*np.ones((3,1)) + a*np.ones((3,1)) - D, 0*np.ones((3,1)))

        # Compute reward given s, a, and D
        if a > 0:
            r_saD = f*(s+a-s_tp1) - (K+c*a) - h*(s+a)
        else:
            r_saD = f*(s+a-s_tp1) - h*(s+a)
        
        # Compute expected immediate reward using the pmf of D
        r[a+1][s+1] = np.matmul(p.transpose(), r_saD)

        # Compute the transition probability matrix
        for j in np.arange(s+a+1):
            if j > 0:
                if D == s + a - j:
                    P[a+1][s, j+1] =  p[j]
            else:
                
                P[a+1][s, j+1] = sum(p[D>=(s+a)])

u_star = dict()
