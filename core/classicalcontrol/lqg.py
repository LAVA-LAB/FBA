import numpy as np
import matplotlib.pyplot as plt # Import Pyplot to generate plots

from ..commons import cm2inch

class lqg(object):
    
    def __init__(self, model, N, belief_mean0, belief_cov0):
        
        self.A = model['A']
        self.B = model['B']
        self.C = model['C']
        self.w = model['noise']['w_cov']
        self.v = model['noise']['v_cov']
        
        self.N = N
        self.x_dim = model['n']
        self.u_dim = model['p']
        
        self.x_hat = np.zeros((self.N, self.x_dim))
        self.x_hat[0, :] = belief_mean0
        
        self.cost = {'F': np.eye(self.x_dim), 'Q': 0.5*np.eye(self.x_dim), 'R': 10*np.eye(self.u_dim)}
        
        # Ricatti equations
        self.P_ricatti(belief_cov0)
        self.S_ricatti()   
        
        # Plot control direction
        self.plot([-20,-20], [20,20], [21,21], k=0)
        
    def _Kalman_gain(self, P):
        
        return P @ self.C.T @ np.linalg.inv(self.C @ P @ self.C.T + self.v)
        
    def P_ricatti(self, belief_cov0):
        
        self.P = {0: belief_cov0}
        self.L = {0: self._Kalman_gain(self.P[0])}
        
        for i in range(self.N):
            
            self.P[i+1] = self.A @ (self.P[i] - self.P[i]@self.C.T@ \
                np.linalg.inv(self.C@self.P[i]@self.C.T + self.v)@self.C@self.P[i]) @ self.A.T + self.w
                
            self.L[i+1] = self._Kalman_gain(self.P[i+1])
            
    def S_ricatti(self):
        
        self.S = {self.N: self.cost['F']}
        
        for i in range(self.N, 0, -1):
            
            self.S[i-1] = self.A.T @ (self.S[i] - self.S[i]@self.B@ \
                np.linalg.inv(self.B.T@self.S[i]@self.B + self.cost['R'])@self.B.T@self.S[i]) @ self.A + self.cost['Q']
                
        self.K = {}
                
        for i in range(0, self.N):
            
            self.K[i] = np.linalg.inv(self.B @ self.S[i+1] @ self.B + self.cost['R']) @ \
                self.B.T @ self.S[i+1] @ self.A
                
    def step(self, k, mean=None):
        
        if mean is None:
            x_hat = self.x_hat[k]
        else:
            x_hat = mean
        
        return -self.K[k] @ x_hat
    
    def update_belief(self, k, u, y):
        
        self.x_hat[k+1] = self.A@self.x_hat[k] + self.B@u + \
            self.L[k+1]@(y - self.C@(self.A@self.x_hat[k] + self.B@u))
            
    def plot(self, xmin, xmax, steps, k=0):
        
        fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))
        
        plt.xlabel('$x_1$', labelpad=0)
        plt.ylabel('$x_2$', labelpad=0)

        for y in np.linspace(xmin[1], xmax[1], steps[1]):        
            for x in np.linspace(xmin[0], xmax[0], steps[0]):
                
                state = np.array([x, y])
                
                u = self.step(k, mean=state).flatten()
                
                state_hat = self.A @ state + self.B @ u
                
                diff = state_hat - state
                
                plt.arrow(state[0], state[1], diff[0], diff[1], width=0.1, 
                          linewidth=0, head_width = 1, head_length = 1)
                
        # Set title
        ax.set_title("Optimal control direction under LQG", fontsize=8)
        
        # Set axis limits
        ax.set_xlim(xmin[0]-2, xmax[0]+2)
        ax.set_xlim(xmin[1]-2, xmax[1]+2)
        
        # Set tight layout
        fig.tight_layout()
                
        plt.show()
        