import numpy as np
import logging
import time
import pickle

logging.basicConfig(filename='problem4.log', filemode='w',
                    format='%(asctime)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

###############################################
#       Initialize & Preprocessing            #
###############################################
logging.info("=========INPUT=========")
# Decision epochs...
N = 31
#N = 62
T = np.arange(1, N+1)
M = 5

# State space...
S = np.arange(10+1)
card_s = len(S)

# Create my action space...
A = dict()
card_a = dict()
for s in S:
    if s >= 4:
        print(s)
        A[s] = np.asarray([1])
    else:
        A[s] = np.asarray([0, 1])

    card_a[s] = len(A[s])

# Create transition probabilities and reward vectors
c = 10
h = 1
P = dict()
r = dict()
for a in np.asarray([0, 1]):
    P[a] = np.zeros((card_s, card_s))
    r[a] = np.zeros((card_s, 1))

for s in S:
    for a in A[s]:
        for j in S:
            if s in S[:-1]:
                if a == 0:
                    if j == s+1:
                        P[a][s, j] = 0.6
                    elif j == s:
                        P[a][s, j] = 0.4
                elif a == 1:
                    if j == s+1-min(s, 5):
                        P[a][s, j] = 0.6
                    elif j == s-min(s, 5):
                        P[a][s, j] = 0.4
            else:
                if a == 1:
                    if j == 6:
                        P[a][s, j] = 0.6
                    elif j == 5:
                        P[a][s, j] = 0.4

        if a == 1:
            r[a][s] = -c-h*max(0, s-M)
        else:
            r[a][s] = -h*s


for a in P.keys():
    logging.info(f"for a={a}, the probabilities are...\n{P[a]}")
    logging.info(f"for a={a}, the rewards are...\n{r[a]}")

###############################################
#       Backward Induction Solving            #
###############################################
start = time.time()
u_star = dict()
d_star = dict()

for t in T:
    d_star[t] = np.zeros((card_s, 1))

t = T[-1]
# Terminal rewards are all zero for all system states
u_star[t] = np.zeros((card_s, 1))
while t > 1:
    t -= 1
    u_star[t] = np.zeros((card_s, 1))

    for s in S:
        u_t_best = -100000000
        a_star = None
        for a in A[s]:
            u_t = r[a][s] + np.matmul(P[a][s, :], u_star[t+1])
            if u_t > u_t_best:
                u_t_best = u_t
                a_star = a

        u_star[t][s] = r[a_star][s] + np.matmul(P[a_star][s, :], u_star[t+1])
        d_star[t][s] = a_star

end = time.time()
logging.info("=========BI OUTPUT==========")
logging.info(f"Traditional BI Alg ran in {end - start}")
logging.info("---Policy Values---")
logging.info(f"The u_star is ...\n{u_star}")
logging.info("---Policy Decisions---")
logging.info(f"The d_star is ...\n{d_star}")

"""# Save the hash of matrices...
f1 = open("data/problem4_partd_u_star.pkl", "wb")
pickle.dump(u_star, f1)
f1.close()
# Save the hash of reward vectors...
f2 = open("data/problem4_partd_d_star.pkl", "wb")
pickle.dump(d_star, f2)
f2.close()"""

P_pi_star = np.zeros((card_s, card_s))
for s in S:
