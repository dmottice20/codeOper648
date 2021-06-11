import numpy as np
import matplotlib.pyplot as plt


def plot_optimal_policy(S, d, v, pi):
    """
    :param S - 1D array representation of the state space
    :param d - optimal policy of same shape as S
    :return - matplotlib pyplot bar graph of the data
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col")
    ax1.bar(S, d+1, align="center", width=0.5)
    ax1.set_xticks(np.arange(0, 120+1, 10))
    ax1.tick_params(axis="both", which="minor", direction="out", length=5)
    ax1.set_xlabel("s")
    ax1.set_ylabel("d*(s)")
    ax1.set_title("Optimal Policy (pi*)")
    
    ax2.bar(S, -v, align="center", width=0.5)
    ax2.set_xlabel("s")
    ax2.set_ylabel("v*(s)")
    ax2.set_title("Value Fx of Optimal Policy")
    
    ax3.bar(S, pi, align="center", width=0.5)
    ax3.set_xlabel("s")
    ax3.set_ylabel("pi*(s)")
    ax3.set_title("Limiting Distribution of Optimal Policy")
    
    plt.tight_layout()
    
    return plt.show()
    
    