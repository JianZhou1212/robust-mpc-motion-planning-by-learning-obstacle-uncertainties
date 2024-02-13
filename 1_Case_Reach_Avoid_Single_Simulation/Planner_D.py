import numpy as np
import casadi
from pytope import Polytope

class Planner_D( ):
    def __init__(self, Params):
        
        self.d_min = Params['d_min']
        self.T = Params['T']
        self.N = Params['N']
        self.l_veh = Params['l_veh']
        self.w_veh = Params['w_veh']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.DEV = Params['DEV']
        self.A_SV = Params['A_SV']
        self.B_SV = Params['B_SV']
        self.d_min = Params['d_min']
        self.d_margin = Params['d_margin']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.Q4 = Params['Q4']
        self.Q5 = Params['Q5']
        self.Q6 = Params['Q6']
        self.Q7 = Params['Q7']
        self.A_road = Params['A_road']
        self.b_road = Params['b_road']
        self.v_low = Params['v_low']
        self.v_up = Params['v_up']
        self.acc_low = Params['acc_low']
        self.acc_up = Params['acc_up']
        self.delta_low = Params['delta_low']
        self.delta_up = Params['delta_up']
        self.RefSpeed = Params['RefSpeed']
        self.RefPos = Params['RefPos']
        self.MPCFormulation = self.MPCFormulation( )
    
    def ReachableSet(self, current_x_SV):
        N = self.N
        A_SV  = self.A_SV
        B_SV  = self.B_SV
        
        G = np.zeros((4, 2*N)) 
        g = np.zeros((4, N))
        Reachable_Set = list( ) 
        Occupancy_SV = list( ) 
        x_before = current_x_SV
        Reachable_Set.append(x_before)
        for t in range(1, N + 1):
            x_next = A_SV@x_before
            x_before = x_next
            min_x = np.min(x_next[0] - self.d_margin) 
            max_x = np.max(x_next[0] + self.d_margin) 
            min_y = np.min(x_next[2] - self.d_margin) 
            max_y = np.max(x_next[2] + self.d_margin) 
            temp_poly = Polytope(np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])) 
            G[:, 2*t-2:2*t] = temp_poly.A
            g[:, t-1]       = temp_poly.b.reshape(4, )
                
            Reachable_Set.append(x_before)
            Occupancy_SV.append(temp_poly)
                
        return  G, g, Occupancy_SV
    
    def Return(self, current_x_SV, current_x_EV):
        RefSpeed = self.RefSpeed
        RefPos = self.RefPos
        phi = current_x_SV[2]
        current_x_SV = np.array([current_x_SV[0], current_x_SV[3]*np.cos(current_x_SV[2]), current_x_SV[1], current_x_SV[3]*np.sin(current_x_SV[2])])
        G, g, Occupancy_SV = self.ReachableSet(current_x_SV)
            
        # MPC-based planning
        RefXPos = RefPos[0]
        RefYPos = RefPos[1]
        RefPhi  = RefPos[2]
        Trajectory_k, Control_k, J_k = self.MPCFormulation(G, g, current_x_EV, RefSpeed, RefXPos, RefYPos, RefPhi)
        Trajectory_k = Trajectory_k.full( )
        Control_k = Control_k.full( )

        return Control_k[:, 0], Trajectory_k, J_k.full( ), Occupancy_SV
     
    def MPCFormulation(self):
        d_min = self.d_min
        N = self.N
        DEV = self.DEV
        T = self.T
        Q1 = self.Q1
        Q2 = self.Q2
        Q3 = self.Q3
        Q4 = self.Q4
        Q5 = self.Q5
        Q6 = self.Q6
        Q7 = self.Q7
        A_road = self.A_road
        b_road = self.b_road
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
        eta   = U[1, :]
        lam = opti.variable(4, N)
        s   = opti.variable(N, 1)
        
        G = opti.parameter(4, 2*N)
        g = opti.parameter(4, N)
        Initial = opti.parameter(DEV, 1)
        v_ref = opti.parameter( )
        x_ref = opti.parameter( )
        y_ref = opti.parameter( )
        phi_ref = opti.parameter( )
        
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], delta[k], eta[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, delta[k], eta[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, delta[k], eta[k])
            k4 = self.vehicle_model(X[:, k] + T*k3, delta[k], eta[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 
        x = X[0, 1::]
        y = X[1, 1::]
        phi = X[2, 1::]
        v = X[3, 1::]
        a = X[4, 1::]
        phi_error = phi[-1] - phi_ref
        v_error = v[-1] - v_ref # longitudinal speed error
        x_error = x[-1] - x_ref # longitudinal position error
        y_error = y[-1] - y_ref # lateral position error
        # collision-avoidance
        for k in range(N):
            p_point = X[0:2, k + 1]
            
            G_point = G[:, 2*k:2*k + 2]
            g_point = g[:, k]
            temp = G_point.T@lam[:, k]
            opti.subject_to((G_point@p_point - g_point).T@lam[:, k] >= d_min - self.d_margin - s[k])
            opti.subject_to(temp[0]**2 + temp[1]**2 == 1)
            opti.subject_to(0 <= lam[:, k])
            opti.subject_to(A_road@p_point <= b_road)

        opti.subject_to(opti.bounded(0, s, d_min))
        opti.subject_to(opti.bounded(-v_up, v, v_up))
        opti.subject_to(opti.bounded(acc_low, a, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + eta@Q2@eta.T + Q3*v_error@v_error.T + Q4*x_error@x_error.T + Q5*y_error@y_error.T + Q6*phi_error@phi_error.T + Q7*s.T@s
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [G, g, Initial, v_ref, x_ref, y_ref, phi_ref], [X, U, J])

    def  vehicle_model(self, w, delta, eta):

        l_f = self.l_f
        l_r = self.l_r
        
        beta = np.arctan(l_r/(l_f + l_r)*np.tan(delta))
        x_dot   = w[3]*np.cos(w[2] + beta) 
        y_dot   = w[3]*np.sin(w[2] + beta)
        phi_dot = w[3]/(l_r)*np.sin(beta)
        v_dot = w[4]
        a_dot = eta
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot, a_dot)