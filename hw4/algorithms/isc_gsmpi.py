import numpy as np
import time


def run_isc_gauss_seidel_modified_policy_iteration(P, r, A, S, epsilon, lamb, m, max_iter):
    """

    :param P: transition probability matrix of shape (card_a, card_s, card_s)
    :param r: reward vectors of shape (card_a, card_s, 1)
    :param A: action space (in this ex. not state dependent)
    :param S: state space
    :param epsilon:  error term (used in convergence / stopping criterion)
    :param lamb: discount factor for IH MDP
    :param m: fixed order for inner loop.
    :param max_iter: max # of iterations to run GSMPI on.
    :return: NOT SURE YET
    """
    print("Running Gauss Seidel modified policy iteration w/ improved stopping criteria...")
    start = time.time()
    card_s = len(S)
    card_a = len(A)
    # Step 1:
    dnp1 = np.zeros((card_s, 1))
    conv_test = epsilon * (1 - lamb) / lamb

    def seminorm_span(v):
        return np.max(v) - np.min(v)

    R = np.reshape(r, (card_s, card_a))
    upsilon_n = np.min(R, axis=1) * np.ones((card_s, 1)) / (1 - lamb)
    upsilon_n = np.asarray([upsilon_n[:, 1]]).transpose()

    n = 0
    rd = 0 * r[0]
    Pd = 0 * P[0]

    un0 = np.zeros((card_s, 1))

    while n < max_iter:
        S = S[::-1]
        for s_i in S:
            un0Star = -np.inf
            for a in A:
                un0sa = r[a - 1][s_i - 1] + lamb * P[a - 1][s_i - 1, :] @ upsilon_n
                if un0sa > un0Star:
                    un0Star = un0sa
                    dnp1[s_i - 1] = a

            un0[s_i - 1] = un0Star

        k = 0
        unk = un0.copy()
        conv_val = seminorm_span(un0 - upsilon_n)

        if conv_val >= conv_test:
            while k < m:
                unkp1 = unk.copy()
                S = S[::-1]
                for s_i in S:
                    unkp1[s_i - 1] = r[int(dnp1[s_i - 1]) - 1][s_i - 1] + lamb * P[int(dnp1[s_i - 1]) - 1][s_i - 1,
                                                                                 :] @ unkp1

                k += 1
                unk = unkp1.copy()

            n += 1
            upsilon_n = unk.copy()
        else:
            d_eps = dnp1.copy()
            break

    end = time.time()
    # Approximate v_approx
    v_approx = unkp1 + lamb * (1 - lamb) ** (-1) * np.min(unkp1 - unk) * np.ones((unkp1.shape[1], 1))

    return upsilon_n, dnp1, v_approx, end - start
