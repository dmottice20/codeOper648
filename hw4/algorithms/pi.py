import numpy as np
import time


def run_policy_iteration(P, r, A, S, lamb):
    """
    
    :param P: transition probability matrix of shape (card_a, card_s, card_s)
    :param r: reward vectors of shape (card_a, card_s, 1)
    :param A: action space (in this ex. not state dependent)
    :param S: state space
    :param lamb: discount factor for IH MDP
    :return: NOT SURE YET 
    """
    print("Running policy iteration...")
    start = time.time()
    card_s = len(S)
    d = np.zeros((card_s, 1))
    dtm1 = d.copy()
    # Execute policy iteration
    for s in S:
        QsaBest = -np.inf
        for a in A:
            #print(a)
            Qsa  = r[a-1][s-1]
            # Update best alternative
            if Qsa > QsaBest:
                QsaBest = Qsa
                d[s-1] = a
    
    n = 0
    rd = 0 * r[0]
    v = 0
    Pd = np.zeros((card_s, card_s))
    
    
    while not np.array_equal(d, dtm1):
        dtm1 = d.copy()
        for s in S:
            Pd[s-1, :] = P[int(d[s-1])-1][s-1, :]
            rd[s-1] = r[int(d[s-1])-1][s-1]
            
        # Policy Evaluation...
        v = np.linalg.solve(np.identity(card_s) - lamb * Pd, rd)
        
        # Policy Improvement
        for s in S:
            QsaBest = -np.inf
            for a in A:
                Qsa = r[a-1][s-1] + lamb * P[a-1][s-1, :] @ v
                if Qsa > QsaBest:
                    QsaBest = Qsa.copy()
                    d[s-1] = a
        
        n += 1

    end = time.time()

    return v, d, Pd, end - start


    
    