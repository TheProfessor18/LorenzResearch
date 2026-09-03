import numpy as np

class System:

    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def coordinates(self, timeRange):

        #timeRange coefficent is points per unit
        t = np.linspace(0, timeRange, timeRange * 10)
        x = 0
        y = 0
        z = 0
        return x, y, z