import numpy as np
import scipy.integrate as scint

class Lorenz63model:

    def f(self, t, state):
        theta = self.theta
        x, y, z = state  # unpack the state vector
        dX = theta[0] * (y-x) 
        dY = x * (theta[1]-z) - y 
        dZ = x*y - theta[2]*z 
        return np.array([dX, dY, dZ])

    def solve(self, params, state0, T, t_range, dt):
        self.theta = params 
        sol = scint.solve_ivp(
                self.f,
                [0,T],
                state0,
                method = 'LSODA',
                t_eval = t_range,
                max_step = dt)
        states = sol.y.transpose()
        return states
