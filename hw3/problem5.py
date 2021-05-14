import itertools
import numpy as np
import logging

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

# Create the state space...
s = np.asarray([1, 2, 3, 4, 5])
# Create one state space for cat player...
S_part = np.transpose([np.tile(s, len(s)), np.repeat(s, len(s))])
# S = cartesian_product(s, s)
S_list = S_part.tolist()
logging.info(S_list)
S_list.remove([2, 1])
S_list.remove([2, 2])
S_list.remove([4, 4])
S_list.remove([5, 4])
S_part2 = np.array(S_list, dtype=object)
# logging.info(S)
# Calculate the cartesian product for mouse player...
# S = cartesian_product(S, S)
# logging.info(f"The state space is... \n{S}")
S = np.dstack(np.meshgrid(S_part2, S_part2)).reshape(-1, 4)
# Append a delta state for mouse getting to the goal cell...
S = np.vstack([S, np.asarray([0, 0, 0, 0])])
# Append a delta state for cat eating the mouse...
S = np.vstack([S, np.asarray([-1, -1, -1, -1])])
card_s = len(S)
# Create an index scheme using a hash table...
state_space = dict()
i = 0
for state in S:
    state_space[i] = state
    i += 1

# Create the actions sets...
A = dict()
card_a = dict()
A_possible = np.asarray([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0]
])

for i, s in state_space.items():
    # Check for the delta states first (indices 882 and 883)
    if i >= 882:
        A[i] = np.asarray([0, 0, 0, 0])
    else:
        #
        A[i] = []
        for a in A_possible:
            # See if that action is even possible.
            if (a + s).all() in S_part2:
                A[i].append(a)

        # Convert the list to an array
        A[i] = np.array(A[i])

    card_a[i] = A[i].shape[0]

# Create the transition probabilities...
P = dict()
r = dict()

for t in T[:-1]:
    P[t] = np.zeros((card_s, card_s))
    r[t] = np.zeros((card_s, 1))

for s_index, s in state_space.items():
    for a in A[s_index]:
        for j_index, j in state_space.items():
            if