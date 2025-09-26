import numpy as np
import pandas as pd
import scipy
import math
import copy

from PIL.ImageChops import offset
from networkx.classes import non_edges


class Finite_Difference:
    def __init__(self, delta_t, delta_S: float, N: int, M: int):
        self.delta_t = delta_t
        self.delta_S = delta_S
        # time steps
        self.N = N
        # spatial steps
        self.M = M

    # def create_grid(self, S_max) -> float | np.ndarray:
    #     grid = np.array([np.array([np.array([S_max, i*self.delta_t]) for i in range(int(self.N / self.delta_t)+1)], dtype=float)] * int(self.M / self.delta_S))
    #
    #     return grid

    def explicit_method(self, rate, sigma, S_max, K, T):

        i = np.arange(1, self.M)
        dt = T / self.N
        dS = S_max / self.M

        a = i * dt * (i * sigma**2 - rate) / 2
        b = 1 -dt * (i**2 * sigma**2 + rate)
        c = i * dt * (i * sigma**2 + rate) / 2

        # call payoffs
        V = np.maximum(np.linspace(0, S_max, self.M+1) - K, 0)

        V_new = V.copy()

        for _ in range(self.N):
            V_new[1:-1] = a * V[:-2] + b * V[1:-1] + c * V[2:]

        # tridiagonal matrix
        # diagonals = [alpha[1:], beta, gamma[:-1]]
        # offsets = [-1, 0, 1]
        # A = scipy.sparse.diags(diagonals, offsets, shape=(self.M-1, self.M-1))
        # A = A.tocsr()

        # call payoffs
        # V = np.maximum(np.linspace(0, S_max, self.M+1) - K, 0)

        updated_grid = []

        # backward evaluation
        # for _ in range(self.N):
        #     V[1:self.M] = A @ V[1:self.M]



# TESTS ################################################################################################################

FD_test = Finite_Difference(0.1, 1, 1000, 100)
print(FD_test.explicit_method(rate=0.05, sigma=0.02, S_max=200, K=100, T=1))


