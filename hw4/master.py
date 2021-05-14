import numpy as np
# import build_mdp_model
from vi import run_value_iteration

#####################################################
##          Controls & Params                      ##
#####################################################
build_mdp = False
A = np.arange(1, 121+1)
S = np.arange(1, 120+1)

# Read in the data.
with open("data/transition_matrix.npy", "rb") as f:
    P = np.load(f)

# Save the reward vector.
with open("data/reward_vector.npy", "rb") as f:
    r = np.load(f)

v_star, d_star, P_dstar, results_table = run_value_iteration(P, r, A, S, epsilon=0.01, lamb=0.99)
print("v_star is...\n", v_star)
print("D_star is...\n", d_star)
