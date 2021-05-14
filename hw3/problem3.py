import numpy as np
import logging
import pickle

logging.basicConfig(filename='problem3.log', filemode='w',
                    format='%(asctime)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

###############################################
#       Initialize & Preprocessing            #
###############################################
logging.info("=========INPUT=========")

# Decision epochs...
N = 84
# Go from 0 to 84 and skip every 4 --> 0,4,8,12,...,84
T = np.arange(0, 84 + 1, 4)

# Capacity is M
M = 20

# State space...(include 20 therefore +1)
S = np.arange(0, 20 + 1)
card_s = len(S)

# Action space
A = dict()
card_a = dict()
for s in S:
    A[s] = np.arange(0, M - s + 1)
    card_a[s] = len(A[s])

# Problem parameters
q = {
    0: 0.7,
    1: 0.15,
    2: 0.1,
    3: 0.04,
    4: 0.01
}
ordering_cost = 0.20
daily_per_unit_cost = 0.01
penalty_cost = 0.50
# ARE WE REBUILDING THE MATRIX OR JUST RUNNING THE BI ALGORITHM>?
building_matrix = False
if building_matrix:
    # Construct transition probabilities and reward vectors
    P = dict()
    R = dict()

    for a in np.arange(0, M + 1):
        P[a] = np.zeros((card_s, card_s))
        R[a] = np.zeros((card_s, 1))

    for s in S:
        for a in A[s]:
            for j in S:
                # Now, loop over and calculate all possible demand combinations in a 4-day intraperiod
                # P(W_t,o = k)
                for k, prob_k in q.items():
                    # P(W_t,1 = l)
                    for l, prob_l in q.items():
                        # P(W_t,2 = m)
                        for m, prob_m in q.items():
                            # P(W_t,3 = r)
                            for r, prob_r in q.items():
                                # See formulation - don't want to go above capacity (20=M)
                                next_state = min(s - k - l - m - r + a, M)
                                P[a][s][max(next_state, 0)] = prob_k * prob_l * prob_m * prob_r
                                R[a][s] = max(-ordering_cost * a, -ordering_cost) - \
                                          daily_per_unit_cost * (
                                                  max(s - k, 0) + max(s - k - l, 0) + max(s - k - l - m, 0) + max(
                                              s - k - l - m - r, 0)) - \
                                          penalty_cost * -(
                                            min(s - k, 0) + min(s - k - l, 0) + min(s - k - l - m, 0) + min(
                                        s - k - l - m - r, 0))

    for a in P.keys():
        logging.info(f"for a={a}, the probabilities are...\n{P[a]}")
        logging.info(f"for a={a}, the rewards are...\n{R[a]}")

    # Save the hash of matrices...
    f1 = open("p_matrix_problem3.pkl", "wb")
    pickle.dump(P, f1)
    f1.close()
    # Save the hash of reward vectors...
    f2 = open("reward_vec_problem3.pkl", "wb")
    pickle.dump(R, f2)
    f2.close()

if not building_matrix:
    # Because we are not building a matrix anymore...
    # Let's load in the needed data.
    with open("p_matrix_problem3.pkl", "rb") as f1:
        P = pickle.load(f1)
    with open("reward_vec_problem3.pkl", "rb") as f2:
        R = pickle.load(f2)

###############################################
#       Backward Induction Solving            #
###############################################
u_star = dict()
d_star = dict()

for t in T:
    d_star[t] = np.zeros((card_s, 1))

t = T[-1]
# Terminal rewards are all zero for all system states
u_star[t] = np.zeros((card_s, 1))
for s in S:
    u_star[t][s] = s * 0.7
while t > 0:
    t -= 4
    u_star[t] = np.zeros((card_s, 1))

    for s in S:
        u_t_best = -100000000
        a_star = None
        for a in A[s]:
            u_t = R[a][s] + np.matmul(P[a][s, :], u_star[t + 4])
            if u_t > u_t_best:
                u_t_best = u_t
                a_star = a

        u_star[t][s] = R[a_star][s] + np.matmul(P[a_star][s, :], u_star[t + 4])
        d_star[t][s] = a_star

logging.info("=========BI OUTPUT==========")
logging.info("---Policy Values---")
logging.info(f"The u_star is ...\n{u_star}")
logging.info("---Policy Decisions---")
logging.info(f"The d_star is ...\n{d_star}")
# Save the hash of matrices...
f1 = open("data/problem3_u_star.pkl", "wb")
pickle.dump(u_star, f1)
f1.close()
# Save the hash of reward vectors...
f2 = open("data/problem3_d_star.pkl", "wb")
pickle.dump(d_star, f2)
f2.close()