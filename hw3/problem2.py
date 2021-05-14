import numpy as np
import logging
import pickle

logging.basicConfig(filename='problem2.log', filemode='w',
                    format='%(asctime)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

###############################################
#       Initialize & Preprocessing            #
###############################################
logging.info("=========INPUT=========")
# Decision epochs
N = 21
T = np.arange(1, N+1)
# Construct state space...
S = np.arange(4, 10 + 1)
# add a state = 11 that represents the delta 
# we are keeping as numbers so that we don't have to switch to
# a range() (more memory and slower) as well as use a character.
S = np.append(S, 11)
card_s = len(S)

# Construct action sets...
A = dict()
card_a = dict()

for s in S:
    if s < 11:
        A[s] = np.asarray([1, 2, 3])
    else:
        A[s] = np.asarray([0])

    card_a[s] = len(A[s])

# Construct transition matrices and reward vectors...
P = dict()
r = dict()

for a in np.arange(0, 3 + 1):
    P[a] = np.zeros((card_s, card_s))
    r[a] = np.zeros((card_s, 1))

# Problem parameters...
# Prob of predation.
q = {
    1: 0,
    2: 0.004,
    3: 0.020
}
# Prob of finding food.
f = {
    1: 0,
    2: 0.4,
    3: 0.6
}
# Energy gain
g = {
    1: 0,
    2: 3,
    3: 5
}
# Energy req. to forage
h = 1

# Create a var (int) to track the offset from
# state value and index.
indx_offset = 4

# Construct P and r.
# The use of continue is extremely bad practice -- I need to come back
# and rework on properly defining this P matrix without the need for that
# command.
for s in S:
    for a in A[s]:
        for j in S:
            if s == 11:
                if j == 11:
                    P[a][s - indx_offset, j - indx_offset] = 1
                    continue
            if s in np.arange(4, 4 + h):
            # Check if j = delta
                if j == 11:
                    P[a][s - indx_offset, j - indx_offset] = q[a] + (1 - q[a]) * (1 - f[a])
                    continue
            if s in np.arange(4 + h, 10 + 1):
                # Check if j = s-1
                if j == s - 1:
                    if a == 1:
                        print(f"s={s} and j={j}")
                    P[a][s - indx_offset, j - indx_offset] = (1 - q[a]) * (1 - f[a])
                    continue
                if j == 11:
                    P[a][s - indx_offset, j - indx_offset] = q[a]
                    continue
            if s in np.arange(4, 10+1):
                # Check if j = min(s+g-h,10)
                if j == min(s + g[a] - h, 10):
                    P[a][s - indx_offset, j - indx_offset] = (1 - q[a]) * f[a]
                    continue

        r[a][s - indx_offset] = 0

logging.info("The transition matrices are built.")
for a in P.keys():
    logging.info(f"For a={a}, P is...\n{P[a]}")

###############################################
#       Backward Induction Solving            #
###############################################
u_star = dict()
t = N

# Initliaze a decsion rule struct
d_star = dict()
for t in T:
    d_star[t] = np.zeros((card_s, 1))

# Initialize U_star[n] = r_N(X_n)
u_star[N] = np.vstack((np.ones((card_s - 1, 1)), np.asarray([0])))

while t > 1:
    t -= 1
    u_star[t] = np.zeros((card_s, 1))
    # For s in S, compute u_t(s)
    for s in S:
        u_t_best = -10000
        a_star = None
        for a in A[s]:
            # Compute u*t_(s) when applying action a
            # 1) calculate expected immediate reward
            # 2) expected total reward when executing
            #    policy pi* during subsequent periods
            u_t = r[a][s - indx_offset] + np.matmul(P[a][s - indx_offset, :], u_star[t + 1])
            if u_t > u_t_best:
                u_t_best = u_t
                a_star = a

        u_star[t][s - indx_offset] = r[a_star][s - indx_offset] + np.matmul(P[a_star][s - indx_offset, :],
                                                                            u_star[t + 1])
        d_star[t][s - indx_offset] = a_star

logging.info("========OUTPUT========")
logging.info("---POLICY VALUES---")
logging.info(f"The u_star is ...\n{u_star}")
logging.info("---Policy Decision---")
logging.info(f"The d_star is ...\n{d_star}")

# Save the hash of matrices...
f1 = open("data/problem2_u_star.pkl", "wb")
pickle.dump(u_star, f1)
f1.close()
# Save the hash of reward vectors...
f2 = open("data/problem2_d_star.pkl", "wb")
pickle.dump(d_star, f2)
f2.close()
