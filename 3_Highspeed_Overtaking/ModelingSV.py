#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import scipy.stats as stats

class ModelingSV( ):
    def __init__(self, Params):
        self.w_lane = Params['w_lane']
        self.N = Params['N']
        self.DSV = Params['DSV']
        self.T = Params['T']
        self.max_speed = Params['max_speed']
        self.cdf_x = Params['cdf_x']
        self.cdf_f = Params['cdf_f']
        self.con_level = Params['con_level']

    def Return(self, k, current_ax_SV, current_x_SV):
        w_lane = self.w_lane
        N      = self.N
        DSV    = self.DSV
        T      = self.T
        max_speed = self.max_speed
        cdf_x = self.cdf_x
        cdf_f = self.cdf_f
        con_level = self.con_level
        
        A_SV = np.array([[1, T], [0, 1]])
        B_SV = np.array([0, T])

        random_pro = np.random.uniform(0 + con_level, 1 - con_level, (1, N))
        pro_error  = np.abs(cdf_f - random_pro)
        index      = np.argmin(pro_error, axis = 0)
        control_SV_horizon = cdf_x[index]
        
        if current_x_SV[1] >=  max_speed:
            control_SV_horizon = -np.abs(control_SV_horizon)
        x_SV_planning = np.zeros((DSV, N + 1))
        y_SV_planning = np.array([w_lane/2]*(N + 1))
        x_SV_planning[:, 0] = current_x_SV
        
        for t in range(1, N + 1):
            x_SV_planning[:, t] = A_SV@x_SV_planning[:, t-1] + B_SV*control_SV_horizon[t - 1]
                
        return control_SV_horizon, x_SV_planning, y_SV_planning
    
