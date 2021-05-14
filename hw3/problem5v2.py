import numpy as np
import logging
import pickle

logging.basicConfig(filename='problem5.log', filemode='w',
                    format='%(asctime)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


# Borrowed this memory-saver cartesian product fx.
# Source: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-
# points-into-single-array-of-2d-points
# ONLY GOOD FOR 1-DIMENSIONAL ARRAYS
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


###############################################
#       Initialize & Preprocessing            #
###############################################
logging.info("=========INPUT=========")
# Decision epochs...
N = 26
T = np.arange(1, N + 1)

# State space...
S_c = np.asarray([
    1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25
])
# Append an index 26 for Delta_i and 27 Delta_cheese,i
S_ca = np.append(S_c, 26)
S_ca = np.append(S_ca, 27)

S_m = np.asarray([
    1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25
])
S_ma = np.append(S_m, 26)
S_ma = np.append(S_ma, 27)

# Take the cartesian product and store it as S - our state space.
S_norm = cartesian_product(S_c, S_m)
S = cartesian_product(S_ca, S_ma)
card_s = len(S)

# Create a hash to store indices and corresponding state values for our state space.
i = 0
state_space = dict()
for s in S:
    state_space[i] = s
    i += 1

# Create the action space...
A_cat = {
    1: np.asarray([2, 6]),
    2: np.asarray([1, 3, 7]),
    3: np.asarray([2, 4, 8]),
    4: np.asarray([3, 5]),
    5: np.asarray([4]),
    6: np.asarray([1, 7, 11]),
    7: np.asarray([2, 6, 8, 12]),
    8: np.asarray([3, 7, 13]),
    11: np.asarray([6, 12, 16]),
    12: np.asarray([7, 11, 13]),
    13: np.asarray([8, 12, 14, 18]),
    14: np.asarray([13, 15, 19]),
    15: np.asarray([14, 20]),
    16: np.asarray([11, 21]),
    18: np.asarray([13, 19, 23]),
    19: np.asarray([14, 18, 20, 24]),
    20: np.asarray([15, 19, 25]),
    21: np.asarray([16]),
    23: np.asarray([18, 24]),
    24: np.asarray([23, 25]),
    25: np.asarray([20, 24]),
    26: np.asarray([26]),
    27: np.asarray([27])
}
A_mouse = {
    1: np.asarray([1, 2, 6]),
    2: np.asarray([1, 2, 3, 7]),
    3: np.asarray([2, 3, 4, 8]),
    4: np.asarray([3, 4, 5]),
    5: np.asarray([4, 5]),
    6: np.asarray([1, 6, 7, 11]),
    7: np.asarray([2, 6, 7, 8, 12]),
    8: np.asarray([3, 7, 8, 13]),
    11: np.asarray([6, 11, 12, 16]),
    12: np.asarray([7, 11, 12, 13]),
    13: np.asarray([8, 12, 13, 14, 18]),
    14: np.asarray([13, 14, 15, 19]),
    15: np.asarray([14, 15, 20]),
    16: np.asarray([11, 16, 21]),
    18: np.asarray([13, 18, 19, 23]),
    19: np.asarray([14, 18, 19, 20, 24]),
    20: np.asarray([15, 19, 20, 25]),
    21: np.asarray([16, 21]),
    23: np.asarray([18, 23, 24]),
    24: np.asarray([23, 24, 25]),
    25: np.asarray([27]),
    26: np.asarray([26]),
    27: np.asarray([27])
}

card_a = dict()
for i in S_m:
    if i <= 25:
        card_a[i] = len(A_mouse[i])

# Create transition probability matrices and reward vectors
P = dict()
r = dict()

q = 0

for a in S_ma:
    P[a] = np.zeros((card_s, card_s))
    r[a] = np.zeros((card_s, 1))

for s_index, s in state_space.items():
    for a in A_mouse[s[0]]:
        for j_index, j in state_space.items():
            if s[0] in A_cat[s[1]]:
                if (j == np.asarray([26, 26])).all():
                    P[a][s_index][j_index] = 1
            elif (s == np.asarray([26, 26])).all():
                if (j == np.asarray([26, 26])).all():
                    P[a][s_index][j_index] = 1
            elif s[0] == 25:
                if (j == np.asarray([27, 27])).all():
                    P[a][s_index][j_index] = 1
            elif (s == np.asarray([27, 27])).all():
                if (j == np.asarray([27, 27])).all():
                    P[a][s_index][j_index] = 1
            elif s in S_norm:
                if (j == np.asarray([a, s[1]])).all():
                    P[a][s_index][j_index] = q
                elif s[1] in A_cat[s[1]]:
                    P[a][s_index][j_index] = (1 - q) / len(A_cat[j[1]])

        r[a][s_index] = 0

###############################################
#       Backward Induction Solving            #
###############################################
u_star = dict()
d_star = dict()

epsilon = 2.2E-6
lamb = 1 - epsilon

desired_end_state = np.asarray([27, 27])
desired_dict = dict((k, v) for k, v in state_space.items() if (v == desired_end_state).all())
desired_index = list(desired_dict.keys())[0]

for t in T:
    d_star[t] = np.zeros((card_s, 1))

t = N
u_star[t] = np.zeros((card_s, 1))
# Terminal rewards are all zero for all system states
for s_index, s in state_space.items():
    if s_index == desired_index:
        print(desired_index)
        u_star[t][s_index] = 1

while t > 1:
    t -= 1
    u_star[t] = np.zeros((card_s, 1))

    for s_index, s in state_space.items():
        u_t_best = -100000000
        a_star = None
        for a in A_mouse[s[0]]:
            u_t = r[a][s_index] + np.matmul(P[a][s_index, :], u_star[t+1])
            if u_t > u_t_best:
                u_t_best = u_t
                a_star = a

        u_star[t][s_index] = r[a_star][s_index] + lamb * np.matmul(P[a_star][s_index, :], u_star[t+1])
        d_star[t][s_index] = a_star

logging.info("=========BI OUTPUT==========")
logging.info("---Policy Values---")
logging.info(f"The u_star is ...\n{u_star}")
logging.info("---Policy Decisions---")
logging.info(f"The d_star is ...\n{d_star}")

start = np.asarray([1, 15])
index_of_interest = dict((k, v) for k, v in state_space.items() if (v == start).all())
start_index = list(index_of_interest.keys())[0]

"""# Save the hash of matrices...
f1 = open("data/problem5_u_star.pkl", "wb")
pickle.dump(u_star, f1)
f1.close()
# Save the hash of reward vectors...
f2 = open("data/problem5_d_star.pkl", "wb")
pickle.dump(d_star, f2)
f2.close()"""

print(u_star[1][start_index])
