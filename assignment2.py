import math
import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 12
        self.dt = 0.2
        self.L = 2.5 # Length between the front wheels and the rear wheels

        # Reference or set point the controller will achieve.
        self.reference1 = [8, 6.6, 3.14/2]
        self.reference2 = [0, 2, 3.14/8]
        self.reference3 = [12, 9, 0]

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        x_t_1 = x_t + v_t * self.dt * np.cos(psi_t)
        y_t_1 = y_t + v_t * self.dt * np.sin(psi_t)
        v_t_1 = v_t + (pedal * self.dt) - (v_t/25.0)
        psi_t_1 = psi_t + ((v_t * self.dt * np.tan(steering)) / self.L)

        return [x_t_1, y_t_1, psi_t_1, v_t_1]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0
        
        for i in range(self.horizon):
            state = self.plant_model(state, self.dt, u[i*2], u[i*2+1])

            x = state[0]
            y = state[1]
            psi = state[2]
            v = state[3]

            # distance cost
            distance_cost =(abs(x - ref[0]) + abs(y - ref[1]))**2
            # steering cost
            steering_cost = abs(psi - ref[2])**2

            distance_weight = 0.45
            steering_weight = 0.55

            cost = distance_cost * distance_weight + steering_cost * steering_weight


        return cost

sim_run(options, ModelPredictiveControl)
