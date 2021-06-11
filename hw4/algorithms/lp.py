import numpy as np
import time
from pulp import *


def solve_mdp_with_lp(P, r, S, A, lamb):
    """

    :param P: probability matrix (transition)
    :param r: reward vector
    :param lamb: discount factor
    :return: upsilon (v-stars)
    """
    print("Running LP approach...")
    start = time.time()
    alpha = np.ones((P.shape[1], 1)) / P.shape[1]

    M = lamb * P.reshape((14520, 120)) - np.tile(np.identity(P.shape[1]), (P.shape[0], 1))
    r_f = -r.reshape((14520, 1))

    model = LpProblem("Hw4-Lp", LpMinimize)
    variable_names = [str(i) for i in np.arange(1, 120+1)]
    DV_variables = LpVariable.matrix("V", variable_names)
    upsilon = np.array(DV_variables).reshape((1, 120))
    objFunc = lpSum(upsilon*np.transpose(alpha))
    model += objFunc

    print(model)

    for i in np.arange(M.shape[0]):
        model += lpSum(upsilon*M[i]) <= r_f[i], "Constraint" + str(i)


    model.solve(COIN_CMD())

    status = LpStatus[model.status]
    end = time.time()
    print(status)

    print("Total Cost:", model.objective.value())

    # Decision Variables
    var = []
    for v in model.variables():
        try:
            var.append(v.value())
            print(v.name, "=", v.value())
        except:
            print("error: could not find value")

    upsilon_star = np.array(var).transpose()
    d_star = np.zeros((120, 1))
    Pd = np.zeros((120, 120))
    for s in S:
        QsaBest = -np.inf
        for a in A:
            upsilon_np1sa = r[a-1][s-1] + lamb * (np.matmul(P[a-1][s-1, :], upsilon_star))
            if upsilon_np1sa > QsaBest:
                QsaBest = upsilon_np1sa
                d_star[s-1] = a

        Pd[s-1, :] = P[int(d_star[s-1])-1][s-1, :]

    return var, end - start, d_star