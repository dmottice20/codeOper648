import numpy as np
import logging
import pickle

logging.basicConfig(filename='problem3analysis.log', filemode='w',
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

# ARE WE REBUILDING THE MATRIX OR JUST RUNNING THE BI ALGORITHM>?
building_matrix = True

# Create the test:
test_u_stars = dict()
test_d_stars = dict()
possible_penalty_cost = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for penalty_cost in possible_penalty_cost:
    print(f"currently on {penalty_cost}")
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

    test_u_stars[penalty_cost] = u_star
    test_d_stars[penalty_cost] = d_star


# Save the hash of matrices...
f1 = open("data/test_results_u_star.pkl", "wb")
pickle.dump(test_u_stars, f1)
f1.close()
# Save the hash of reward vectors...
f2 = open("data/test_results_d_star.pkl", "wb")
pickle.dump(test_d_stars, f2)
f2.close()
