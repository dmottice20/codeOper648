import numpy as np
import logging
import itertools

logging.basicConfig(filename='problem1.log', filemode='w',
                    format='%(asctime)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

###############################################
#       Initialize & Preprocessing            #
###############################################
logging.info("=========INPUT=========")
# Decision epochs...
N = 3
T = np.arange(N)

# State space...
# List of possible states
s = [
    ['s'],
    ['t', 'u']
]
state_space = dict()
S = itertools.product(*s)
card_s = len(list(S))
i = 0
for element in itertools.product(*s):
    state_space[i] = element
    i += 1

# Construct action sets...
A = dict()
card_a = dict()
for s in state_space.keys():
    # Can choose project 1 or 2 no matter the state
    A[s] = np.asarray([1, 2])
    card_a[s] = len(A[s])

# Construct transition probability matrices and reward
# vectors.
P = dict()
r = dict()
for a in np.arange(1, 2+1):
    P[a] = np.zeros((card_s, card_s))
    r[a] = np.zeros((card_s, 1))

for i, curr in state_space.items():
    for a in A[i]:
        for j, next in state_space.items():
            if a == 1:
                if i == j:
                    P[a][i, j] = 1
            elif a == 2:
                if i == 0:
                    P[a][i, j] = 0.5
                elif i == 1 and j == 0:
                    P[a][i, j] = 1

        # Construct rewards...
        if a == 1:
            r[a][i] = 1
        elif i == 0:
            r[a][i] = 2
        elif i == 1:
            r[a][i] = 0

for a in P.keys():
    logging.info(f"for a={a}, the probabilities are...\n{P[a]}")

for a in r.keys():
    logging.info(f"for a={a}, the rewards are...\n{r[a]}")

###############################################
#       Backward Induction Solving            #
###############################################
u_star = dict()

d_star = dict()
for t in T:
    d_star[t] = np.zeros((card_s, 1))

t = T[-1]
u_star[t] = np.zeros((card_s, 1))

while t > 0:
    t -= 1
    u_star[t] = np.zeros((card_s, 1))
    # For s in S, compute u_t(s)
    for s, curr in state_space.items():
        u_t_best = -10000
        a_star = None
        for a in A[s]:
            u_t = r[a][s] + np.matmul(P[a][s, :], u_star[t + 1])
            if u_t > u_t_best:
                u_t_best = u_t
                a_star = a

        u_star[t][s] = r[a_star][s] + np.matmul(P[a_star][s, :],
                                                u_star[t + 1])
        d_star[t][s] = a_star

logging.info("=========OUTPUT==========")
logging.info("---Policy Values---")
logging.info(f"The u_star is ...\n{u_star}")
logging.info("---Policy Decisions---")
logging.info(f"The d_star is ...\n{d_star}")

