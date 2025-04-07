import numpy as np
import casadi

class ModelingSVTrue( ):
    def __init__(self, Params):
        self.N = Params['N']
        self.DSV = Params['DSV']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.T = Params['T']
        self.v_low = Params['v_low']
        self.v_up = Params['v_up']
        self.acc_low = Params['acc_low']
        self.acc_up = Params['acc_up']
        self.delta_low = Params['delta_low']
        self.delta_up = Params['delta_up']
        self.v_target = Params['v_target']
        self.x_track = Params['x_track']
        self.y_track = Params['y_track']
        self.num_points = Params['num_points']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.Q4 = Params['Q4']
        self.Q5 = Params['Q5']
        self.MPCFormulation = self.MPCFormulation( )
    
    def Return(self, current_x_SV, Direction):
        # The following to get the reference sequence for the SV from the track
        distances = np.sqrt((self.x_track - current_x_SV[0])**2 + (self.y_track - current_x_SV[1])**2)
        min_distance_index = np.argmin(distances)
        if Direction == 'clockwise': # front
            index_ref = list(range(min_distance_index + 1, min_distance_index + self.N + 1))
            index_ref = [num if num <= (self.num_points - 1) else num - self.num_points for num in index_ref]
        else: # back
            index_ref = list(range(min_distance_index - 1, min_distance_index -self.N - 1, -1))
            index_ref = [num if 0 <= num else num + self.num_points for num in index_ref]
        
        x_ref_k =  self.x_track[index_ref]
        y_ref_k = self.y_track[index_ref]

        # MPC-based trajectory planning for SV
        Trajectory_k, Control_k = self.MPCFormulation(current_x_SV, x_ref_k, y_ref_k)
        Trajectory_k = Trajectory_k.full( )
        Control_k = Control_k.full( )
        
        noise_delta = np.random.normal(0, 0.025, self.N)
        noise_acc = np.random.normal(0, 0.05, self.N)
        
        delta_uncertain = np.clip(Control_k[0, :] + noise_delta, self.delta_low, self.delta_up)
        acc_uncertain   = np.clip(Control_k[1, :] + noise_acc, self.acc_low, self.acc_up)
        
        Control_k[0, :] = delta_uncertain
        Control_k[1, :] = acc_uncertain
        
        for k in range(self.N):
            k1 = self.vehicle_model(Trajectory_k[:, k], delta_uncertain[k], acc_uncertain[k])
            k2 = self.vehicle_model(Trajectory_k[:, k] + self.T/2*k1, delta_uncertain[k], acc_uncertain[k])
            k3 = self.vehicle_model(Trajectory_k[:, k] + self.T/2*k2, delta_uncertain[k], acc_uncertain[k])
            k4 = self.vehicle_model(Trajectory_k[:, k] + self.T*k3, delta_uncertain[k], acc_uncertain[k])
            x_next = Trajectory_k[:, k] + self.T/6 * (k1 + 2*k2 + 2*k3 + k4)
            Trajectory_k[:, k + 1] == x_next

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

        v_low = self.v_low 
        v_up  = self.v_up 
        acc_low = self.acc_low 
        acc_up  = self.acc_up 
        delta_low = self.delta_low 
        delta_up  = self.delta_up

        opti = casadi.Opti( )
        X = opti.variable(DSV, N + 1)
        U = opti.variable(2, N)
        delta   = U[0, :]
        a       = U[1, :]
        
        Initial = opti.parameter(DSV, 1)
        x_ref = opti.parameter(1, N)
        y_ref = opti.parameter(1, N)
        
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
        v = X[3, 1::]
        x_error   = x - x_ref # long. position error
        y_error   = y - y_ref # lateral position error
        v_error   = v - self.v_target  # longitudinal terminal speed

        opti.subject_to(opti.bounded(v_low, v, v_up))
        opti.subject_to(opti.bounded(0.5*acc_low, a, 0.5*acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + a@Q2@a.T + Q3*v_error@v_error.T + Q4*x_error@x_error.T + Q5*y_error@y_error.T
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [Initial, x_ref, y_ref], [X, U])

    def  vehicle_model(self, w, delta, a):
        # EV model
        l_f = self.l_f
        l_r = self.l_r
        
        beta = np.arctan(l_r/(l_f + l_r)*np.tan(delta))
        x_dot   = w[3]*np.cos(w[2] + beta) 
        y_dot   = w[3]*np.sin(w[2] + beta)
        phi_dot = w[3]/(l_r)*np.sin(beta)
        v_dot = a
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot)
    
