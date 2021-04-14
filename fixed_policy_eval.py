import numpy as np

## Step 0 -- Initialization & Preprocessing
# Define the decision epochs...
n = 4
T = np.arange(n-1)

# Define the states...
m = 8
S = np.arange(m)
cardS = len(S)

# Define the actions...
A = dict()
cardA = dict()

for s in S:
    # We have to subtract by 1's because python starts the index
    # on 0 (not 1 like Matlab)
    if s == 1-1:
        A[s] = np.asarray([2-1, 3-1, 4-1])
    elif s == 2-1:
        A[s] = np.asarray([5-1, 6-1])
    elif s == 3-1:
        A[s] = np.asarray([5-1, 6-1, 7-1])
    elif s == 4-1:
        A[s] = np.asarray([7-1])
    else:
        A[s] = np.asarray([8-1])
    
    cardA[s] = len(A[s])

print("Action sets...\n",A)
print("\nAction cardinalities...\n", cardA)

# Input some Problem data
C = np.zeros((m, m))
C[0][1] = 3
C[0][2] = 4
C[0][3] = 2
C[1][4] = 4
C[1][5] = 5
C[2][4] = 5
C[2][5] = 6
C[2][6] = 1
C[3][6] = 2
C[4][7] = 1
C[5][7] = 2
C[6][7] = 6
# Probability of actually moving to selected node...
q = 0.8

# Initialize transition probability matrices and reward vectors...
P = dict()
r = dict()

# Preallocate some memory by defining matrices and vectors of 0's
for a in np.arange(m):
    P[a] = np.zeros((cardS, cardS))
    r[a] = np.zeros((cardS, 1))

d = dict()
# Initialize a decision rule vector for each t...
for t in T:
    d[t] = np.zeros((cardS, 1))
print(T)
# NOW, compute reward vectors and transition probability matrices...
# Loop over current state
for s in S:
    # Loop over each action possible in state s
    for a in A[s]:
        # Loop over next state (S again)
        for j in S:
            if len(A[s]) == 1:
                P[a][s][j] = 1
            else:
                # |A_s| > 1
                if j == a:
                    P[a][s][j] = q
                elif j in A[s]:
                    P[a][s][j] = (1-q) / (cardA[s]-1)

        # Now I can compute expected immediate reward...
        r[a][s] = np.matmul(P[a][s,:], C[s,:].transpose())



# Step 1: RUN THE ALGORITHM
# Initialize a dictionary for ETR vectors
u = dict()
# Start at time = n ( -1 because of python)
t = n-1
# Initialize a value fx using terminal rewards fx.
# This ends being 0 for this problem instance.
u[n-1] = np.zeros((cardS, 1))

# Stepps 2-4:
while t > 0:
    print(t)
    # Step 3:
    # Decrement t and initialize value fx vector
    t -= 1
    u[t] = np.zeros((cardS, 1))
    # Loop over states
    for s in S:
        # Choose the action that maxes state number
        # 'Turn right' policy
        a = max(A[s])
        d[t][s] = a
        u[t][s] = r[a][s] + np.matmul(P[a][s,:], u[t+1])

# Display results...
print("\n\n========POLICY VALUES========\n\n")
print(u.keys())