#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import math
import casadi
from pytope import Polytope
import scipy.stats as stats
from scipy import interpolate
from numpy.linalg import matrix_power
import time

class Planner_P( ): # Proposed method
    def __init__(self, Params):
        
        self.d_min = Params['d_min']
        self.T = Params['T']
        self.N = Params['N']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.DEV = Params['DEV']
        self.A_SV = Params['A_SV']
        self.B_SV = Params['B_SV']
        self.H  = Params['H']
        self.h  = Params['h']
        self.nv = Params['nv']
        self.nu = Params['nu']
        self.U_SV_Poly = Params['U_SV_Poly']
        self.d_min = Params['d_min']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.Q4 = Params['Q4']
        self.Q5 = Params['Q5']
        self.v_low = Params['v_low']
        self.v_up = Params['v_up']
        self.acc_low = Params['acc_low']
        self.acc_up = Params['acc_up']
        self.delta_low = Params['delta_low']
        self.delta_up = Params['delta_up']
        self.x_track = Params['x_track']
        self.y_track = Params['y_track']
        self.v_target = Params['v_target']
        self.num_points = Params['num_points']
        self.LinearProgramming = self.LinearProgramming( )
        self.MPCFormulation = self.MPCFormulation( )
    
    def ReachableSet(self, current_x_SV, U_hat_k, phi):
        # Predict the reachable set of SV -- online
        N = self.N
        A_SV  = self.A_SV
        B_SV  = self.B_SV

        BU = B_SV*U_hat_k
        
        G = np.zeros((4, 2*N)) 
        g = np.zeros((4, N))
        Reachable_Set = list( ) # exact SV reachable set
        Occupancy_SV_aug = list( ) # augmented approximation of SV occupancy
        Reachable_Set.append(current_x_SV)
        for t in range(1, N + 1):
            if t == 1:
                reachable_set_t         = (A_SV@Reachable_Set[t - 1] + BU) 
            else:
                reachable_set_t         = (A_SV*Reachable_Set[t - 1] + BU)
            vertex = reachable_set_t.V
            vertex_xy = np.delete(vertex, [1, 3], axis = 1)
            occupancy_SV_t = Polytope(vertex_xy) # project the occupancy from the reach. set
            occupancy_SV_t.minimize_V_rep( )
            temp_poly_aug = occupancy_SV_t
            G[:, 2*t-2:2*t] = temp_poly_aug.A
            g[:, t-1]       = temp_poly_aug.b.reshape(4, )
                
            Reachable_Set.append(reachable_set_t)
            Occupancy_SV_aug.append(temp_poly_aug)

        return G, g, Occupancy_SV_aug
    
    def Return(self, current_x_SV, current_x_EV, theta_before, y_before, u_before):
        # Return planned trajectory of EV
        H = self.H
        phi = current_x_SV[2]
        current_x_SV = np.array([current_x_SV[0], current_x_SV[3]*np.cos(current_x_SV[2]), current_x_SV[1], current_x_SV[3]*np.sin(current_x_SV[2])])

        if (H @ u_before <= theta_before + H @ y_before).all():
            theta_k = theta_before
            y_k     = y_before
        else:
            theta_k, y_k = self.LinearProgramming(theta_before, y_before, u_before)
            theta_k = theta_k.full( )
            y_k     = y_k.full( )
        
        U_hat_k = Polytope(H, theta_k) + y_k
            
        G, g, Occupancy_SV_aug = self.ReachableSet(current_x_SV, U_hat_k, phi)

        # The following is to extract the reference sequence for the EV from the track
        distances = np.sqrt((self.x_track - current_x_EV[0])**2 + (self.y_track - current_x_EV[1])**2)
        min_distance_index = np.argmin(distances)
        index_ref = list(range(min_distance_index + 1, min_distance_index + self.N + 1))
        index_ref = [num if num <= (self.num_points - 1) else num - self.num_points for num in index_ref]
        x_ref_k =  self.x_track[index_ref]
        y_ref_k = self.y_track[index_ref]

        Trajectory_k, Control_k, J_k = self.MPCFormulation(G, g, current_x_EV, x_ref_k[-1], y_ref_k[-1])
        Trajectory_k = Trajectory_k.full( )
        Control_k = Control_k.full( )
        
        return Control_k[:, 0], Trajectory_k, J_k.full( ), Occupancy_SV_aug, theta_k, y_k, U_hat_k
    
    def LinearProgramming(self): 
        # Learn the control set of EV by solving LP problem
        H = self.H
        h = self.h
        nv = self.nv
        nu = self.nu
        
        opti = casadi.Opti( )
        
        rho = opti.variable( )
        theta = opti.variable(nv, 1)
        y = opti.variable(nu, 1)
        
        theta_before = opti.parameter(nv, 1)
        y_before = opti.parameter(nu, 1)
        u_before = opti.parameter(nu, 1)
        
        opti.minimize(rho + np.ones((1, nv))@theta)
        opti.subject_to(-H @ y <= theta - H @ u_before)
        opti.subject_to(theta_before + H @ y_before <= theta + H @ y)
        opti.subject_to(H @ y <= 1 - rho)
        opti.subject_to(opti.bounded(0, theta, 1))
        opti.subject_to(opti.bounded(0, rho, 1))
        opti.subject_to(theta <= rho)

        opts = {"ipopt.print_level": 0,
                "ipopt.linear_solver": "ma57",
                "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('f', [theta_before, y_before, u_before], [theta, y])
    
    def MPCFormulation(self):
        # MPC problem for motion planning of EV
        d_min = self.d_min
        N = self.N
        DEV = self.DEV
        T = self.T
        Q1 = self.Q1
        Q2 = self.Q2
        Q3 = self.Q3
        Q4 = self.Q4
        Q5 = self.Q5
 
        v_low = self.v_low 
        v_up = self.v_up 
        acc_low = self.acc_low 
        acc_up = self.acc_up 
        delta_low = self.delta_low 
        delta_up = self.delta_up

        opti = casadi.Opti( )
        X = opti.variable(DEV, N + 1)
        U = opti.variable(2, N)
        delta = U[0, :]
        acc   = U[1, :]
        lam = opti.variable(4, N)
        
        v_target = opti.parameter( )
        G = opti.parameter(4, 2*N)
        g = opti.parameter(4, N)
        Initial = opti.parameter(DEV, 1)
        
        x_ref = opti.parameter( )
        y_ref = opti.parameter( )

        
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], delta[k], acc[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, delta[k], acc[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, delta[k], acc[k])
            k4 = self.vehicle_model(X[:, k] + T*k3, delta[k], acc[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 

        x   = X[0, 1::]
        y   = X[1, 1::]
        v   = X[3, 1::]

        v_error = v - self.v_target # longitudinal speed error
        x_error = x - x_ref # long. position error
        y_error = y - y_ref # lateral position error
 
        # collision-avoidance
        for k in range(N):
            p_point = X[0:2, k + 1]
            
            G_point = G[:, 2*k:2*k + 2]
            g_point = g[:, k]
            temp = G_point.T@lam[:, k]
            opti.subject_to((G_point@p_point - g_point).T@lam[:, k] >= d_min)
            opti.subject_to(temp[0]**2 + temp[1]**2 <= 1)
            opti.subject_to(0 <= lam[:, k])

        opti.subject_to(opti.bounded(v_low, v, v_up))
        opti.subject_to(opti.bounded(acc_low, acc, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + acc@Q2@acc.T + Q3*v_error@v_error.T + Q4*x_error@x_error.T + Q5*y_error@y_error.T
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [G, g, Initial, x_ref, y_ref], [X, U, J])

    def  vehicle_model(self, w, delta, acc):
        # EV model
        l_f = self.l_f
        l_r = self.l_r
        
        beta = np.arctan(l_r/(l_f + l_r)*np.tan(delta))
        x_dot   = w[3]*np.cos(w[2] + beta) 
        y_dot   = w[3]*np.sin(w[2] + beta)
        phi_dot = w[3]/(l_r)*np.sin(beta)
        v_dot = acc
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot)
