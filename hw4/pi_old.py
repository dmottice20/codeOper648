import numpy as np


def evaluate_policy_old(S, P, r, age, lamb):
    """
    :param month - age of car (in mths) to buy once s= 120
    :return value fx, decision rule, induced P matrix
    """
    # Policy Creation
    card_s = len(S)
    d = np.zeros((card_s, 1))
    for s in S:
        if s == S[-1]:
            d[s-1] = age
        else:
            d[s-1] = 1
             
    # Create Induced P matrix
    Pd = np.zeros((card_s, card_s))
    rd = np.zeros((card_s, 1))
    for s in S:
        Pd[s-1, :] = P[int(d[s-1])-1][s-1, :]
        rd[s-1] = r[int(d[s-1])-1][s-1]
    
    v = np.linalg.solve(np.identity(card_s) - lamb * Pd, rd)
             
    return v, d, Pd