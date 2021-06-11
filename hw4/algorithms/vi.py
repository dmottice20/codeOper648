import numpy as np
import time


def run_value_iteration(P, r, A, S, epsilon, lamb):
    """

    :param P: transition probability matrix of shape (card_a, card_s, card_s)
    :param r: reward vectors of shape (card_a, card_s, 1)
    :param A: action space (in this ex. not state dependent)
    :param S: state space
    :param epsilon:  error term (used in convergence / stopping criterion)
    :param lamb: discount factor for IH MDP
    :return: NOT SURE YET
    """
    print("Running value iteration...")
    start = time.time()
    v_np1 = np.zeros((P.shape[1], 1))
    d = np.zeros((P.shape[1], 1))
    conv_test = epsilon * (1 - lamb) / (2 * lamb)
    conv_val = np.inf
    n = 0
    max_iter = 1750
    in_run_results_table = np.zeros((max_iter, 2))

    while (conv_val >= conv_test) and (n < max_iter):
        v_n = v_np1.copy()
        for s in S:
            v_np1_sa_best = -np.inf
            for a in A:
                v_np1_sa = r[a-1][s-1] + lamb * np.matmul(P[a-1][s-1, :], v_n)
                if v_np1_sa > v_np1_sa_best:
                    v_np1_sa_best = v_np1_sa

            v_np1[s-1] = v_np1_sa_best

        conv_val = np.linalg.norm(v_np1 - v_n, np.inf)
        in_run_results_table[n, :] = np.asarray([v_np1[0], conv_val])
        n += 1

    Pd = np.zeros((len(S), len(S)))
    for s in S:
        v_np1_sa_best = -np.inf
        for a in A:
            v_np1_sa = r[a-1][s-1] + lamb * np.matmul(P[a-1][s-1, :], v_n)
            if v_np1_sa > v_np1_sa_best:
                v_np1_sa_best = v_np1_sa
                d[s-1] = a

        Pd[s-1, :] = P[int(d[s-1][0])-1][s-1, :]

    end = time.time()

    return v_np1, d, Pd,  in_run_results_table, end - start
