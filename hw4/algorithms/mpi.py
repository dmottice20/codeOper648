import numpy as np
import time


def run_modified_policy_iteration(P, r, A, S, epsilon, lamb, m, max_iter):
    """

    :param P: transition probability matrix of shape (card_a, card_s, card_s)
    :param r: reward vectors of shape (card_a, card_s, 1)
    :param A: action space (in this ex. not state dependent)
    :param S: state space
    :param epsilon:  error term (used in convergence / stopping criterion)
    :param lamb: discount factor for IH MDP
    :param m: fixed order for inner loop.
    :param max_iter: max # of iterations to run MPI on.
    :return: NOT SURE YET
    """
    print("Running modified policy iteration...")
    start = time.time()
    # Initialize optimal decision rule vectors.
    card_s = len(S)
    card_a = len(A)
    dnp1 = np.zeros((card_s, 1))

    conv_test = epsilon * (1 - lamb) / (2 * lamb)

    # Reshape rewards into |S| x |A| matrix
    R = np.reshape(r, (card_s, card_a))

    upsilon_n = np.min(R, axis=1) * np.ones((card_s, 1)) / (1 - lamb)
    upsilon_n = np.asarray([upsilon_n[:, 1]]).transpose()

    n = 0
    rd = 0 * r[0]
    Pd = 0 * P[0]
    un0 = np.zeros((card_s, 1))

    while n < max_iter:
        # Policy Improvement
        for s in S:
            un0Star = -np.inf
            for a in A:
                un0Sa = r[a - 1][s - 1] + lamb * P[a - 1][s - 1, :] @ upsilon_n
                if un0Sa > un0Star:
                    un0Star = un0Sa
                    dnp1[s - 1] = a

            un0[s - 1] = un0Star

        # Partial Policy Evaluation
        conv_val = np.linalg.norm(un0 - upsilon_n, np.inf)
        k = 0
        unk = un0.copy()

        if conv_val >= conv_test:
            for s in S:
                Pd[s - 1, :] = P[int(dnp1[s - 1]) - 1][s - 1, :]
                rd[s - 1] = r[int(dnp1[s - 1]) - 1][s - 1]

            while k < m:
                unkp1 = rd + lamb * Pd @ unk
                k += 1
                unk = unkp1.copy()

            n += 1
            upsilon_n = unk.copy()
        else:
            d_eps = dnp1.copy()
            break

    end = time.time()

    return upsilon_n, end - start
