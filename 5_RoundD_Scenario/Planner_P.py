import numpy as np
from pytope import Polytope
import casadi
from scipy.linalg import sqrtm, svd
import time
import math

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
        self.nv = Params['nv']
        self.nx = Params['nx']
        self.nu = Params['nu']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.v_low = Params['v_low']
        self.v_up  = Params['v_up']
        self.acc_low    = Params['acc_low']
        self.acc_up     = Params['acc_up']
        self.delta_low  = Params['delta_low']
        self.delta_up   = Params['delta_up']
        self.x_track = Params['x_track']
        self.y_track = Params['y_track']
        self.num_points = Params['num_points']
        self.Veh_SV_Shape = Params['Veh_SV_Shape']
        self.H       = Params['H']
        self.h       = Params['h']
        self.LinearProgramming = self.LinearProgramming( )
        self.MPCFormulation = self.MPCFormulation( )
    
    def ReachableSet(self, current_x_SV, U_hat_k):
        # Predict the reachable set of SV -- online
        N = self.N
        A_SV  = self.A_SV
        B_SV  = self.B_SV
        Veh_SV_Shape = self.Veh_SV_Shape
        
        BU = B_SV*U_hat_k
        
        G = np.zeros((4, 2*N)) 
        g = np.zeros((4, N))
        Reachable_Set = list( )    # exact SV reachable set
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
            Occupancy_SV_aug.append(temp_poly_aug + Veh_SV_Shape)

        return G, g, Occupancy_SV_aug
    
    def Return(self, current_x_SV0, current_x_SV1, current_x_EV, theta0_before, y0_before, u0_before, theta1_before, y1_before, u1_before):
        # Return planned trajectory of EV
        H = self.H
        current_x_SV0 = np.array([current_x_SV0[0], current_x_SV0[3]*np.cos(current_x_SV0[2]), current_x_SV0[1], current_x_SV0[3]*np.sin(current_x_SV0[2])])
        current_x_SV1 = np.array([current_x_SV1[0], current_x_SV1[3]*np.cos(current_x_SV1[2]), current_x_SV1[1], current_x_SV1[3]*np.sin(current_x_SV1[2])])

        if (H @ u0_before <= theta0_before + H @ y0_before).all():
            theta0_k = theta0_before
            y0_k     = y0_before
        else:
            theta0_k, y0_k = self.LinearProgramming(theta0_before, y0_before, u0_before)
            theta0_k = theta0_k.full( )
            y0_k     = y0_k.full( )

        if (H @ u1_before <= theta1_before + H @ y1_before).all():
            theta1_k = theta1_before
            y1_k     = y1_before
        else:
            theta1_k, y1_k = self.LinearProgramming(theta1_before, y1_before, u1_before)
            theta1_k = theta1_k.full( )
            y1_k     = y1_k.full( )


        U_hat0_k = Polytope(H, theta0_k) + y0_k
        U_hat1_k = Polytope(H, theta1_k) + y1_k

        G0, g0, Occupancy_SV0_aug = self.ReachableSet(current_x_SV0, U_hat0_k)
        G1, g1, Occupancy_SV1_aug = self.ReachableSet(current_x_SV1, U_hat1_k)

        distances = np.sqrt((self.x_track - current_x_EV[0])**2 + (self.y_track - current_x_EV[1])**2)
        min_distance_index = np.argmin(distances)
        index_ref = list(range(min_distance_index + 3, min_distance_index + self.N + 3))
        index_ref = [num if num <= (self.num_points-1) else (self.num_points-1) for num in index_ref]
        x_ref_k = self.x_track[index_ref]
        y_ref_k = self.y_track[index_ref]

        Trajectory_k, Control_k, J_k = self.MPCFormulation(G0, g0, G1, g1, current_x_EV, x_ref_k, y_ref_k)
        Trajectory_k = Trajectory_k.full( )
        Control_k = Control_k.full( )
        
        return Control_k[:, 0], Trajectory_k, J_k.full( ), Occupancy_SV0_aug, Occupancy_SV1_aug,  theta0_k, y0_k, U_hat0_k, theta1_k, y1_k, U_hat1_k
    
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
        lam0 = opti.variable(4, N)
        s0   = opti.variable(N, 1)
        lam1 = opti.variable(4, N)
        s1   = opti.variable(N, 1)

        G0 = opti.parameter(4, 2*N)
        g0 = opti.parameter(4, N)
        G1 = opti.parameter(4, 2*N)
        g1 = opti.parameter(4, N)
        Initial = opti.parameter(DEV, 1)
        
        x_ref   = opti.parameter(1, N)
        y_ref   = opti.parameter(1, N)

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

        x_error = x - x_ref
        y_error = y - y_ref 
 
        # collision-avoidance
        for k in range(N):
            p_point = X[0:2, k + 1]
            
            G0_point = G0[:, 2*k:2*k + 2]
            g0_point = g0[:, k]
            temp0    = G0_point.T@lam0[:, k]
            opti.subject_to((G0_point@p_point - g0_point).T@lam0[:, k] >= d_min)
            opti.subject_to(temp0[0]**2 + temp0[1]**2 <= 1)
            opti.subject_to(0 <= lam0[:, k])
            
            G1_point = G1[:, 2*k:2*k + 2]
            g1_point = g1[:, k]
            temp1    = G1_point.T@lam1[:, k]
            opti.subject_to((G1_point@p_point - g1_point).T@lam1[:, k] >= d_min)
            opti.subject_to(temp1[0]**2 + temp1[1]**2 <= 1)
            opti.subject_to(0 <= lam1[:, k])


        opti.subject_to(opti.bounded(0, s0, d_min))
        opti.subject_to(opti.bounded(0, s1, d_min))
        opti.subject_to(opti.bounded(v_low, v, v_up))
        opti.subject_to(opti.bounded(acc_low, acc, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + acc@Q2@acc.T +  Q3*(x_error@x_error.T + y_error@y_error.T)
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0,
                "ipopt.linear_solver": "ma57",
                "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [G0, g0, G1, g1, Initial, x_ref, y_ref], [X, U, J])

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
