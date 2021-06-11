import numpy as np
import time


def run_gs_value_iteration(P, r, A, S, epsilon, lamb, m, max_iter):
    # Run the value iteration Gauss-Seidel algorithm.
    print("Running Gauss-seidel value iteration...")
    start = time.time()
    v_np1 = np.zeros((P.shape[1], 1))
    d = np.zeros((P.shape[1], 1))
    conv_test = epsilon * (1 - lamb) / (2 * lamb)
    conv_val = np.inf
    n = 0
    max_iter = 1750

    while True:
        n += 1

        v_n = v_np1.copy()

        for s in range(len(S)):
            Q = [float(r[a][s] +
                       lamb * P[a][s, :].dot(v_np1))
                 for a in range(len(A))]

            v_n[s] = max(Q)

        conv_val = np.linalg.norm(v_n - v_np1)

        if conv_val < conv_test:
            print("MSG_STOP_EPSILON_OPTIMAL_POLICY")
            break
        elif n == max_iter:
            print("_MSG_STOP_MAX_ITER")
            break

    d = []
    for s in range(len(S)):
        Q = np.zeros(len(A))
        for a in range(len(A)):
            Q[a] = (r[a][s] +
                    lamb * P[a][s, :].dot(v_np1))

        v_n[s] = Q.max()
        d.append(int(Q.argmax()))

    end = time.time()

    return v_n, d, end - start