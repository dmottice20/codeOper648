import numpy as np
import time


def run_isc_value_iteration(P, r, A, S, epsilon, lamb, max_iter):
    """
    :param P: transition probability matrix of shape (card_a, card_s, card_s)
    :param r: reward vectors of shape (card_a, card_s, 1)
    :param A: action space (in this ex. not state dependent)
    :param S: state space
    :param epsilon:  error term (used in convergence / stopping criterion)
    :param lamb: discount factor for IH MDP
    :param max_iter: # of maximum iterations to run.
    :return: NOT SURE YET
    """
    print("Running value iteration w/ improve stopping criteria...")
    start = time.time()
    card_s = len(S)
    card_a = len(A)
    
    v_np1 = np.zeros((card_s, 1))
    d = np.zeros((card_s, 1))
    
    def seminorm_span(v):
        return np.max(v) - np.min(v)
    
    conv_test = epsilon * (1 - lamb) / lamb
    conv_val = np.inf
    
    n = 0
    
    while (conv_val >= conv_test) and (n < max_iter):
        v_n = v_np1.copy()
        for s in S:
            v_np1_sa_best = -np.inf
            for a in A:
                v_np1_sa = r[a-1][s-1] + lamb * P[a-1][s-1, :] @ v_n
                if v_np1_sa > v_np1_sa_best:
                    v_np1_sa_best = v_np1_sa
                    
            v_np1[s-1] = v_np1_sa_best
        
        conv_val = seminorm_span(v_np1-v_n)
        
        n += 1
        
    Pd = np.zeros((card_s, card_s))
    for s in S:
        v_np1_sa_best = -np.inf
        for a in A:
            v_np1_sa = r[a-1][s-1] + lamb * np.matmul(P[a-1][s-1, :], v_n)
            if v_np1_sa > v_np1_sa_best:
                v_np1_sa_best = v_np1_sa
                d[s-1] = a

        Pd[s-1, :] = P[int(d[s-1][0])-1][s-1, :]

    end = time.time()
    # Project out / approximate v*
    # Lv + lamb(1-lamb)^-1*min(Bv)*e
    v_approx = v_np1 + lamb*(1-lamb)**(-1)*np.min(v_np1-v_n)*np.ones((v_np1.shape[1], 1))
        
    return v_np1, d, Pd, v_approx, end - start
                    