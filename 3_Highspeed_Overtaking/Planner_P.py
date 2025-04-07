import numpy as np
import math
import casadi
from pytope import Polytope
import scipy.stats as stats
from scipy import interpolate
from numpy.linalg import matrix_power
import time

class Planner_P( ): # Proposed Approach
    def __init__(self, Params):
        
        self.T = Params['T']
        self.N = Params['N']
        self.N_set = Params['N_set']
        self.T_set = Params['T_set']
        self.l_veh = Params['l_veh']
        self.w_veh = Params['w_veh']
        self.w_lane = Params['w_lane']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.DEV = Params['DEV']
        self.DPM = Params['DPM']
        self.A_SV_set = Params['A_SV_set']
        self.B_SV_set = Params['B_SV_set']
        self.X_SV_Poly = Params['X_SV_Poly']
        self.infinity = Params['infinity']
        self.max_speed = Params['max_speed']
        self.v_ref = Params['v_ref']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.Q4 = Params['Q4']
        self.Q5 = Params['Q5']
        self.MPCFormulation = self.MPCFormulation( )
    
    def ReachableSet(self, current_x_SV, current_y_SV, samples):
        # Predict the reachable set of SV -- online
        T = self.T
        N = self.N
        N_set = self.N_set
        T_set = self.T_set
        l_veh = self.l_veh
        w_veh = self.w_veh
        w_lane = self.w_lane
        A_SV_set  = self.A_SV_set
        B_SV_set  = self.B_SV_set
        X_SV_Poly = self.X_SV_Poly
        
        low_ax = np.min(samples)
        up_ax  = np.max(samples)
        
        BU_SV_Poly = Polytope(np.array([[B_SV_set[0]*low_ax, B_SV_set[1]*low_ax], [B_SV_set[0]*up_ax, B_SV_set[1]*up_ax]]))
        
        Reachable_Set = list( )
        Reachable_Set.append(current_x_SV)
        coarse_length_min = np.array([None]*(N_set + 1))
        coarse_length_max = np.array([None]*(N_set + 1))
        coarse_length_min[0] = current_x_SV[0]
        coarse_length_max[0] = current_x_SV[0]
    
        G = np.zeros((4, 2*N))
        g = np.zeros((4, N))
        for t in range(1, N_set + 1):
            if t == 1:
                reachable_set_t = (A_SV_set@Reachable_Set[t - 1] + BU_SV_Poly)
            else:
                reachable_set_t = (A_SV_set*Reachable_Set[t - 1] + BU_SV_Poly) & X_SV_Poly
            Reachable_Set.append(reachable_set_t)
            vertex   = reachable_set_t.V
            vertex_x = vertex[:, 0]
            coarse_length_min[t] = np.min(vertex_x)
            coarse_length_max[t] = np.max(vertex_x)
        
        coarse_interval = np.linspace(0, N_set*T_set, N_set + 1)
        fx_min = interpolate.interp1d(coarse_interval, coarse_length_min, kind = 'quadratic')
        fx_max = interpolate.interp1d(coarse_interval, coarse_length_max, kind = 'quadratic')
        fine_interval   = np.linspace(0, N*T, N + 1)
        fine_length_min = fx_min(fine_interval)
        fine_length_max = fx_max(fine_interval)
        Robust_SV_Position = np.ones((4, N)) # leading vehicle with vehicle shape expanded, middle_x, middle_y, dx, dy
        
        for t in range(1, N + 1):
            if fine_length_min[t] > fine_length_max[t]:
                min_x = (fine_length_min[t] + fine_length_max[t])/2 - l_veh
                max_x = (fine_length_min[t] + fine_length_max[t])/2 + l_veh
            else:
                min_x = fine_length_min[t] - l_veh
                max_x = fine_length_max[t] + l_veh
                
            min_y = w_lane/2 - w_veh/2
            max_y = w_lane/2 + w_veh/2
            
            temp_poly = Polytope(np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]))
            G[:, 2*t-2:2*t] = temp_poly.A
            g[:, t-1]       = temp_poly.b.reshape(4, )
            Robust_SV_Position[:, t-1] = np.array([(max_x + min_x)/2, (min_y + max_y)/2, (max_x - min_x)/2, (max_y - min_y)/2])
        
        return G, g, Robust_SV_Position, low_ax, up_ax
    
    
    def Return(self, current_x_SV, current_y_SV, current_x_EV, samples):
        # Return planned trajectory of EV
        G, g, Robust_SV_Position_k, low_ax, up_ax = self.ReachableSet(current_x_SV, current_y_SV, samples)

        Trajectory_k, U_k, J_k = self.MPCFormulation(G, g, current_x_EV)
        Trajectory_k = Trajectory_k.full( )
        U_k          = U_k.full( )
        
        return U_k[:, 0], Trajectory_k,  Robust_SV_Position_k, low_ax, up_ax
    
    def MPCFormulation(self):
        # MPC problem for motion planning of EV
        N = self.N
        DEV = self.DEV
        max_speed = self.max_speed
        T = self.T
        w_lane = self.w_lane
        y_ref  = w_lane/2
        v_ref  = self.v_ref
        Q1 = self.Q1
        Q2 = self.Q2
        Q3 = self.Q3
        Q4 = self.Q4
        Q5 = self.Q5

        opti = casadi.Opti( )
        X = opti.variable(DEV, N + 1)
        U = opti.variable(2, N)

        delta   = U[0, :]
        eta     = U[1, :]
        lam = opti.variable(4, N)
        
        G = opti.parameter(4, 2*N)
        g = opti.parameter(4, N)
        Initial = opti.parameter(DEV, 1)
        
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], delta[k], eta[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, delta[k], eta[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, delta[k], eta[k])
            k4 = self.vehicle_model(X[:, k] + T*k3, delta[k], eta[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 
        
        y   = X[1, 1::]
        phi = X[2, 1::]
        v   = X[3, 1::]
        a   = X[4, 1::]
        y_error = y[-1] - y_ref # lateral position error
        v_error = v[-1] - v_ref # longitudinal speed error
        # collision-avoidance
        for k in range(N):
            G_point = G[:, 2*k:2*k + 2]
            g_point = g[:, k]
            p_point = X[0:2, k + 1]
            opti.subject_to((G_point@p_point - g_point).T@lam[:, k] >= 2.5)
            temp = G_point.T@lam[:, k]
            opti.subject_to((temp[0]**2 + temp[1]**2) <= 1)
            opti.subject_to(0 <= lam[:, k])

        opti.subject_to(opti.bounded(0, v, max_speed))
        opti.subject_to(opti.bounded(-3, a, 3))
        opti.subject_to(opti.bounded(-0.2, delta, 0.2))
        opti.subject_to(opti.bounded(w_lane/2, y, 7))
        
        J = delta@Q1@delta.T + eta@Q2@eta.T + y_error@Q3@y_error.T + v_error@Q4@v_error.T + phi@Q5@phi.T
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "mumps", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [G, g, Initial], [X, U, J])

    def  vehicle_model(self, w, delta, eta):
        # EV model
        l_f = self.l_f
        l_r = self.l_r

        x_dot   = w[3] 
        y_dot   = w[3]*w[2] + (l_r/(l_f + l_r))*w[3]*delta
        phi_dot = w[3]/(l_f + l_r)*delta
        v_dot   = w[4]
        a_dot   = eta
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot, a_dot)
        
        
                
        
        
        
        
        
        
       

    
