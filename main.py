import numpy as np

from simplex_duas_fases import MetodoSimplexDuasFases

A = np.array([
    [-3, 4, 1, 0],
    [1, -1, 0, 1],
    [ 1, 1, 0, 0]
])

C = np.array([ [-1], [-3], [0], [0] ])

b = np.array([ [12], [4], [6] ])


df_sol = MetodoSimplexDuasFases(A, b, C).solve()

df_sol.to_csv('solution.csv')
