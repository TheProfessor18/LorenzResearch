import numpy as np
from scipy.integrate import solve_ivp

class System:

    def __init__(self, sigma, rho, beta, precision = 1, initialState = [1,1,1]):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initialState = initialState
        self.precision = precision

    def values(self, t):
        timePoints = np.linspace(0, t, t*self.precision)
        def lorenz(time, initialState):
            x, y, z = initialState
            dxdt = self.sigma * (y-x)
            dydt = (x * (self.rho - z)) - y
            dzdt = (x*y) - (self.beta*z)
            return [dxdt, dydt, dzdt]
        return solve_ivp(lorenz, [0,t], self.initialState, t_eval=timePoints).y, timePoints