import numpy as np
from pulp import *


S = np.arange(2)
card_s = len(S)

A = {
    0: np.arange(2),
    1: np.arange(2)
}
card_a = 2

P = np.empty((2, 2, 2))
P[0] = [[0.5, 0.5],
        [0.4, 0.6]]
P[1] = [[0.8, 0.2],
        [0.7, 0.3]]

R = np.empty((2, 2, 2))
R[0] = [[9, 3],
        [3, -7]]
R[1] = [[4, 4],
        [1, -19]]
r = np.empty((2, 2, 1))
r[0] = np.asarray([
    [6],
    [-3]
])
r[1] = np.asarray([
    [4],
    [-5]
])

lamb = 0.9

alpha = np.ones((card_s, 1)) / card_s

# cell2mat(P) -->
#print(P.reshape((4, 2)))
#print(np.tile(np.identity(card_s), (card_a, 1)))
M = lamb * P.reshape((4, 2)) - np.tile(np.identity(card_s), (card_a, 1))
model = LpProblem("Toymaker-Ex-Lp-Approach", LpMinimize)
variable_names = [str(i) for i in range(2)]
variable_names.sort()
print("Variable indices: ", variable_names)
DV_variables = LpVariable.matrix("V", variable_names)

upsilon = np.array(DV_variables).reshape((1, 2))

objFunc = lpSum(upsilon*np.transpose(alpha))
model += objFunc

r = r.reshape((4, 1))
for i in np.arange(M.shape[0]):
    model += lpSum(upsilon*M[i]) <= -r[i], "Constraint " + str(i)

#model.solve()
model.solve(COIN_CMD())

status = LpStatus[model.status]

print(status)

print("Total Cost:", model.objective.value())

# Decision Variables

for v in model.variables():
    try:
        print(v.name, "=", v.value())
    except:
        print("error: could not find value")

