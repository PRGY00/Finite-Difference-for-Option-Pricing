import numpy as np
import pandas as pd
import scipy
import math
import copy
import matplotlib.pyplot as plt

from PIL.ImageChops import offset
from networkx.classes import non_edges


class Finite_Difference:
    def __init__(self, N: int, M: int) -> None:
        # time steps
        self.N = N
        # spatial steps
        self.M = M

    def explicit_method(self, rate: float, sigma: float, S_max: float, K: float, T: float, is_call: bool=True) -> list:

        i = np.arange(1, self.M)
        dt = T / self.N
        dS = S_max / self.M

        a = i * dt * (i * sigma**2 - rate) / 2
        b = 1 - dt * (i**2 * sigma**2 + rate)
        c = i * dt * (i * sigma**2 + rate) / 2

        # call payoffs (last column = present time)
        V = np.maximum((-1)**(1-is_call) * np.linspace(0, S_max, self.M+1) - K, 0)

        V_new = copy.deepcopy(V)

        # list for payoff values
        payoffs_grid = []

        for _ in range(self.N):
            V_new[1:-1] = a * V[:-2] + b * V[1:-1] + c * V[2:]
            payoffs_grid.append(V_new)

        return payoffs_grid

    def implicit_method(self, rate: float, sigma: float, S_max: float, K: float, T: float, is_call: bool=True) -> list:

        i = np.arange(1, self.M)
        dt = T / self.N
        dS = S_max / self.M

        a = i * dt * (rate - i * sigma**2) / 2
        b = 1 + dt * (i**2 * sigma**2 + rate)
        c = -i * dt * (i * sigma**2 + rate) / 2

        # call payoffs (last column = present time)
        V = np.maximum((-1)**(1-is_call) * np.linspace(0, S_max, self.M+1) - K, 0)

        V_new = copy.deepcopy(V)

        # list for payoff values
        payoffs_grid = []

        for _ in range(self.N):
            V_new[1:-1] = a * V[:-2] + b * V[1:-1] + c * V[2:]
            payoffs_grid.append(V_new)

        return payoffs_grid

    def crank_nicolson_method(self, rate: float, sigma: float, S_max: float, K: float, T: float, is_call: bool=True) -> list:

        i = np.arange(1, self.M)
        dt = T / self.N
        dS = S_max / self.M

        a = -i * dt * (i * sigma**2 - rate) / 4
        b = 1 - dt * (i**2 * sigma**2 + rate) / 2
        c = -i * dt * (i * sigma**2 + rate) / 4

        # call payoffs (last column = present time)
        V = np.maximum((-1)**(1-is_call) * np.linspace(0, S_max, self.M+1) - K, 0)

        V_new = copy.deepcopy(V)

        # list for payoff values
        payoffs_grid = []

        for _ in range(self.N):
            V_new[1:-1] = a * V[:-2] + b * V[1:-1] + c * V[2:]
            payoffs_grid.append(V_new)

        return payoffs_grid

    def payoff_grid_plot(self, payoff_grid, S_max, T):

        payoff_grid = np.array(payoff_grid)

        # create gird for plot
        S = np.linspace(0, S_max, self.M+1)
        t = np.linspace(0, T, self.N)
        T_grid, S_grid = np.meshgrid(t, S, indexing="ij")

        # plot the surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(S_grid, T_grid, payoff_grid, cmap="viridis")

        ax.set_xlabel("Stock Price (S)")
        ax.set_ylabel("Time to Maturity (t)")
        ax.set_zlabel("Option Value")
        ax.set_title("Option Payoff Grid")
        plt.show()

# TESTS ################################################################################################################

FD_test = Finite_Difference(1000, 100)
Explicit_Method_run = FD_test.explicit_method(rate=0.05, sigma=0.02, S_max=200, K=100, T=1)
FD_test.payoff_grid_plot(Explicit_Method_run, 200, 1)

Implicit_Method_run = FD_test.implicit_method(rate=0.05, sigma=0.02, S_max=200, K=100, T=1)
FD_test.payoff_grid_plot(Implicit_Method_run, 200, 1)

CN_Method_run = FD_test.crank_nicolson_method(rate=0.05, sigma=0.02, S_max=200, K=100, T=1)
FD_test.payoff_grid_plot(CN_Method_run, 200, 1)






