import numpy as np
from algorithms.vi import run_value_iteration
from algorithms.pi import run_policy_iteration
from algorithms.mpi import run_modified_policy_iteration
from algorithms.gsmpi import run_gauss_seidel_modified_policy_iteration
from algorithms.isc_vi import run_isc_value_iteration
from algorithms.isc_mpi import run_isc_modified_policy_iteration
from algorithms.isc_gsmpi import run_isc_gauss_seidel_modified_policy_iteration
from algorithms.variable_order_isc_mpi import run_variable_order_isc_gauss_seidel_modified_policy_iteration
from pi_old import evaluate_policy_old
from limiting_dist import calculate_limiting_distribution
from algorithms.lp import solve_mdp_with_lp

#####################################################
##          Problem 1                              ##
#####################################################
build_mdp = True
A = np.arange(1, 121 + 1)
S = np.arange(1, 120 + 1)

# Read in the data.
with open("data/transition_matrix.npy", "rb") as f:
    P = np.load(f)

# Save the reward vector.
with open("data/reward_vector.npy", "rb") as f:
    r = np.load(f)

# Store the times for each:
timers = []

v_star_pi, d_star_pi, Pd_pi, time = run_policy_iteration(P, r, A, S, lamb=0.99)
with open("outputs/solnQuality/policy_iteration.npy", "wb") as f:
    np.save(f, v_star_pi)

timers.append(time)

v_star_vi, d_star_vi, Pd_vi, results_table, time = run_value_iteration(P, r, A, S, epsilon=0.1, lamb=0.99)
with open("outputs/solnQuality/value_iteration.npy", "wb") as f:
    np.save(f, v_star_vi)

timers.append(time)

v_star_gs_vi, d_star_gs_vi, Pd_gs_vi, results_gs_table, time = run_value_iteration(P, r, A, S, epsilon=0.1, lamb=0.99)
with open("outputs/solnQuality/gs_value_iteration.npy", "wb") as f:
    np.save(f, v_star_vi)

timers.append(time)

v_star_mpi, time = run_modified_policy_iteration(P, r, A, S, epsilon=0.1, lamb=0.99, m=90, max_iter=1500)
with open("outputs/solnQuality/modified_policy_iteration.npy", "wb") as f:
    np.save(f, v_star_mpi)

timers.append(time)

v_star_gsmpi, time = run_gauss_seidel_modified_policy_iteration(P, r, A, S, epsilon=0.1, lamb=0.99, m=90, max_iter=1500)
with open("outputs/solnQuality/gs_modified_policy_iteration.npy", "wb") as f:
    np.save(f, v_star_gsmpi)

timers.append(time)

v_star_isc_vi, d_star_isc_vi, Pd_isc_vi, v_approx_vi, time = run_isc_value_iteration(P, r, A, S, epsilon=0.1, lamb=0.99,
                                                                                     max_iter=5000)
with open("outputs/solnQuality/isc_value_iteration.npy", "wb") as f:
    np.save(f, v_approx_vi)

timers.append(time)

v_star_isc_gs_vi, d_star_isc_gs_vi, Pd_isc_gs_vi, v_approx_gs_vi, time = run_isc_value_iteration(P, r, A, S, epsilon=0.1, lamb=0.99,
                                                                                     max_iter=5000)
with open("outputs/solnQuality/isc_gs_value_iteration.npy", "wb") as f:
    np.save(f, v_approx_vi)

timers.append(time)

v_star_isc_mpi, d_star_isc_mpi, v_approx_mpi, time = run_isc_modified_policy_iteration(P, r, A, S, epsilon=0.1,
                                                                                       lamb=0.99,
                                                                                       m=90, max_iter=5000)
with open("outputs/solnQuality/isc_modified_policy_iteration.npy", "wb") as f:
    np.save(f, v_approx_mpi)

timers.append(time)

v_star_isc_gsmpi, d_star_isc_gsmpi, v_approx_gsmpi, time = run_isc_gauss_seidel_modified_policy_iteration(P, r, A, S,
                                                                                                          epsilon=0.1,
                                                                                                          lamb=0.99,
                                                                                                          m=90,
                                                                                                          max_iter=5000)

with open("outputs/solnQuality/isc_gs_modified_policy_iteration.npy", "wb") as f:
    np.save(f, v_approx_gsmpi)

timers.append(time)

v_star_lp, time, d_star_lp = solve_mdp_with_lp(P, r, S, A, lamb=0.99)

with open("outputs/solnQuality/lp.npy", "wb") as f:
    np.save(f, v_star_lp)

timers.append(time)

timer_array = np.asarray(timers)
with open("outputs/compQuality/times.npy", "wb") as f:
    np.save(f, timer_array)

################################################
##              Problem 2                     ##
################################################
pi = calculate_limiting_distribution(Pd_pi)

################################################
##              Problem 4                     ##
################################################
# Evaluate d_old
v_old, d_old, Pd_old = evaluate_policy_old(S, P, r, age=62, lamb=0.99)
pi_old = calculate_limiting_distribution(Pd_old)

################################################
##              Problem 6                     ##
################################################
v_star_vo_isc_gsmpi, d_star_vo_isc_gsmpi, v_approx_vo_gsmpi, time = \
    run_variable_order_isc_gauss_seidel_modified_policy_iteration(
    P, r, A, S, epsilon=0.1, lamb=0.99, m=90, max_iter=5000)

with open('outputs/solnQuality/vo_isc_gs_mpi.npy', 'wb') as f:
    np.save(f, v_star_vo_isc_gsmpi)