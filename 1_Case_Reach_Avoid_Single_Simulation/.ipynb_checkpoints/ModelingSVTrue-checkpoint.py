#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import casadi
import scipy.stats as stats

class ModelingSVTrue( ):
    def __init__(self, Params):
        self.N = Params['N']
        self.DSV = Params['DSV']
        self.l_veh = Params['l_veh']
        self.w_veh = Params['w_veh']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.Q4 = Params['Q4']
        self.Q5 = Params['Q5']
        self.Q6 = Params['Q6']
        self.T = Params['T']
        self.A_road = Params['A_road']
        self.b_road = Params['b_road']
        self.v_low = Params['v_low']
        self.v_up = Params['v_up']
        self.acc_low = Params['acc_low']
        self.acc_up = Params['acc_up']
        self.delta_low = Params['delta_low']
        self.delta_up = Params['delta_up']
        self.MPCFormulation = self.MPCFormulation( )
    
    def Return(self, current_x_SV, RefPos):
            
        # MPC-based planning for SV
        RefXPos = RefPos[0]
        RefYPos = RefPos[1]
        RefPhi  = RefPos[2]
        Trajectory_k, Control_k = self.MPCFormulation(current_x_SV, RefXPos, RefYPos, RefPhi)
        Trajectory_k = Trajectory_k.full( )
        Control_k = Control_k.full( )

        return Control_k[:, 0], Trajectory_k
    
    def MPCFormulation(self):
        N = self.N
        DSV = self.DSV
        T = self.T
        Q1 = self.Q1
        Q2 = self.Q2
        Q3 = self.Q3
        Q4 = self.Q4
        Q5 = self.Q5
        Q6 = self.Q6
        
        A_road = self.A_road
        b_road = self.b_road
        v_low = self.v_low 
        v_up = self.v_up 
        acc_low = self.acc_low 
        acc_up = self.acc_up 
        delta_low = self.delta_low 
        delta_up = self.delta_up

        opti = casadi.Opti( )
        X = opti.variable(DSV, N + 1)
        U = opti.variable(2, N)
        delta   = U[0, :]
        a       = U[1, :]
        
        Initial = opti.parameter(DSV, 1)
        x_ref = opti.parameter( )
        y_ref = opti.parameter( )
        phi_ref = opti.parameter( )
        
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], delta[k], a[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, delta[k], a[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, delta[k], a[k])
            k4 = self.vehicle_model(X[:, k] + T*k3, delta[k], a[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 
        x = X[0, 1::]
        y = X[1, 1::]
        phi = X[2, 1::]
        v   = X[3, 1::]
        x_error     = x[-1] - x_ref # longitudinal position error
        y_error     = y[-1] - y_ref # lateral position error
        phi_error   = phi[-1] - phi_ref  # longitudinal terminal speed
        v_error     = v[-1]
        # collision-avoidance w.r.t the region bound
        for k in range(N):
            p_point = X[0:2, k + 1]
            opti.subject_to(A_road@p_point <= b_road)

        opti.subject_to(opti.bounded(0, v, v_up))
        opti.subject_to(opti.bounded(acc_low, a, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + a@Q2@a.T + Q3*phi_error@phi_error.T + Q4*x_error@x_error.T + Q5*y_error@y_error.T + Q6*v_error@v_error.T
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0,
                "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [Initial, x_ref, y_ref, phi_ref], [X, U])


    def  vehicle_model(self, w, delta, a):

        l_f = self.l_f
        l_r = self.l_r
        
        beta = np.arctan(l_r/(l_f + l_r)*np.tan(delta))
        x_dot   = w[3]*np.cos(w[2] + beta) 
        y_dot   = w[3]*np.sin(w[2] + beta)
        phi_dot = w[3]/(l_r)*np.sin(beta)
        v_dot = a
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot)
    
