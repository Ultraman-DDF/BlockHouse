# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:05:41 2024

@author: Hanwen
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\liu70\Downloads\Blockhouse\merged_data.csv')

#%%
class PriceImpactModel:
    def __init__(self, alpha, sigma, Phi, beta, lam, p_list,gamma, T, N, dt,S,Q,dQ):
        
        self.alpha = alpha          # Accuracy of predictions
        self.sigma = sigma          # Volatility parameter
        self.Phi = Phi              # Mean reversion rate of signal f_t
        self.T = T                  # Total time horizon
        self.N = N                  # Number of time steps
        self.t_grid = np.linspace(0, T, N)
        self.beta= np.array(beta)   # Impact decay rate
        self.lam= np.array(lam)     # Impact level lambda.
        self.p_list= np.array(p_list) # Power 
        self.gamma=gamma
        self.dt=dt
        self.S=S
        self.Q=Q
        self.dQ=dQ
        


    

    def compute_ratio(self):
        PnL=0
        D=100
        N_imp = len(self.beta)
        lam=np.array(self.lam)
        sqrt_term = np.sqrt(1 + (2 * self.lam * self.beta) /(self.gamma*self.sigma**2))
        Cf = (self.alpha * (1 + self.beta / self.Phi)) /(self.gamma * self.sigma**2 * (sqrt_term+self.beta/self.Phi))
        Cj = (sqrt_term - 1)
        SR=0
        for d in range(D):
            # Initialize arrays
            f = np.zeros(self.N+1)
            Q = np.zeros(self.N+1)
            J_n = np.zeros((self.N+1, N_imp))
            J= np.zeros((self.N+1))


            # Set initial values
            f[0] = np.random.normal(0, 1)
            Q[0] = np.random.normal(0, 1)
            
            term1=0
            term2=0
            
            PnL=0
            RQ=0
            
            for i in range(N):
                f[i+1] = np.exp(-self.Phi * self.dt)*f[i] + np.sqrt(1-np.exp(-2*self.Phi * self.dt))*np.random.normal(0, 1)
                for n in range(N_imp):
                   J_n[i+1,n]= (J_n[i,n]+Cf*f[i+1]-Q[i])/(np.exp(self.beta * self.dt)+Cj)
                J[i+1] = np.sum(J_n[i+1,:])
                Q[i+1]=Cf*f[i+1]-Cj*J[i+1]
                term1=self.alpha*(1-np.exp(-self.beta*self.dt))*Q[i+1]*f[i]
                term2=self.lam*(1-np.exp(-self.beta*self.dt))*J[i+1]
                RQ += self.gamma*(self.sigma**2)*(Q[i+1]**2)*self.dt/2
                PnL += (term1 + term2)
                
            SR+=PnL/RQ
                
        
        return SR/D

    def simulate_OW(self):
        N_imp = len(self.beta)
        lam=np.array(self.lam)

        # Initialize arrays
        J_n = np.zeros((self.N+1, N_imp))
        J = np.zeros(self.N+1)
        I = np.zeros(self.N)
        P = np.zeros(self.N)

        # Set initial values
        for i in range(self.N):
            for n in range(N_imp):
               J_n[i+1,n]= np.exp(-self.beta[n] * self.dt)*(J_n[i,n]+self.dQ[i])
               
            J[i+1] = np.sum(J_n[i+1,:])
            I[i] = np.sum(lam * J[i+1])
            P[i] = self.S[i] + I[i]

        return self.t_grid, self.S[:], self.Q[:], P[:], I[:], J[1:]
    

    def simulate_AFS(self):
        N_imp = len(self.beta)
        lam=np.array(self.lam)

        # Initialize arrays
        J_n = np.zeros((self.N+1, N_imp))
        J = np.zeros(self.N+1)
        I = np.zeros(self.N)
        P = np.zeros(self.N)

        # Simulate the paths
        for i in range(self.N):
            for n in range(N_imp):
               J_n[i+1,n]= np.exp(-self.beta[n] * self.dt)*(J_n[i,n]+self.dQ[i])
               
            J[i+1] = np.sum(J_n[i+1,:])
            I[i] = np.sum(self.lam * np.sign(J_n[i+1,:]) * np.abs(J_n[i+1,:]) ** self.p_list)
            P[i] = self.S[i] + I[i]

        return self.t_grid, self.S[:], self.Q[:], P[:], I[:], J[1:]
    
    

#%%
S = np.array(data['mid_price'].values)
Q = np.array(data['Signed Volume'].cumsum().values)/1e6
dQ= np.array(data['Signed Volume'])/1e6


alpha =  1.67e-4     # Accuracy of predictions
sigma = 0.02     # Volatility parameter
Phi = 0.139  # Mean reversion rate of signal
N = len(Q)        # Number of time steps         
T = 1       # Total time horizon
dt= T/N     # stepsize
lam = [0.1]     # Impact levels
p_list = 0.8       # Power law exponents 
beta = [2]       # Impact decay rates
gamma=3         # risk aversion parameter.



model = PriceImpactModel(alpha, sigma, Phi, beta, lam, p_list,gamma, T, N, dt,S,Q,dQ)
#%%
"""
1. Construct and code the linear OW model and nonlinear AFS model, and visualize the distribution
of price impact based on the given data. (33 pt)

"""
#OW model

t_ow, S_ow, Q_ow, P_ow, I_ow, J_ow= model.simulate_OW()
plt.figure(figsize=(14, 6))
plt.suptitle('Obizhaeva and Wang (OW) Model')
plt.subplot(2, 2, 1)
plt.plot(t_ow, S_ow, label='Unaffected Price $S_t$')
plt.plot(t_ow, P_ow, label='Market Price $P_t$')
plt.title('Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(t_ow, J_ow)
plt.title('Time Decay impact $J_t$')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.subplot(2, 2, 3)
plt.plot(t_ow, Q_ow)
plt.title('Holdings $Q_t$')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.subplot(2, 2, 4)
plt.plot(t_ow, I_ow)
plt.title('Price Impact $I_t$')
plt.xlabel('Time')
plt.ylabel('Impact')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#%%
#AFS model


t_afs, S_afs, Q_afs, P_afs, I_afs, J_afs = model.simulate_AFS()

plt.figure(figsize=(14, 6))
plt.suptitle('Alfonsi, Fruth, and Schied (AFS) Model')


plt.subplot(2, 2, 1)
plt.plot(t_afs, S_afs, label='Unaffected Price $S_t$')
plt.plot(t_afs, P_afs, label='Market Price $P_t$')
plt.title('Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_ow, J_afs)
plt.title('Time Decay impact $J_t$')
plt.xlabel('Time')
plt.ylabel('Signal')

plt.subplot(2, 2, 3)
plt.plot(t_afs, Q_afs)
plt.title('Holdings $Q_t$')
plt.xlabel('Time')
plt.ylabel('Quantity')

plt.subplot(2, 2, 4)
plt.plot(t_afs, I_afs)
plt.title('Price Impact $I_t$')
plt.xlabel('Time')
plt.ylabel('Impact')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%
"""
2. Implement and code the optimal strategy with Linear Impact and visualize the Sharpe Ratio
plots in Section 6.2. (33 pt)

"""
alpha =  1.67e-4     
sigma = 0.02   
Phi = 0.139 
N = 5120        
dt= 0.01     
T = N*dt       
lam = [0.01]     
p_list = 0.5     
beta = [2]      
gamma=3         


fixed_beta = [2]
lambda_values = [0.01, 0.02, 0.03]
sr_lambda = []

for lam in lambda_values:
    model = PriceImpactModel(alpha=alpha, sigma=sigma, Phi=Phi, beta=fixed_beta, 
                             lam=[lam], p_list=p_list, gamma=gamma, T=T, N=N, dt=dt,S=S,Q=Q,dQ=dQ)
    SR = model.compute_ratio()
    sr_lambda.append(SR)


fixed_lambda = [0.01]
beta_values = [4, 5, 6, 7]
sr_beta = []

for beta_val in beta_values:
    model = PriceImpactModel(alpha=alpha, sigma=sigma, Phi=Phi, beta=[beta_val], 
                             lam=fixed_lambda, p_list=p_list, gamma=gamma, T=T, N=N, dt=dt,S=S,Q=Q,dQ=dQ)
    SR = model.compute_ratio()
    sr_beta.append(SR)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Sharpe Ratio vs Lambda
axs[0].plot(lambda_values, sr_lambda, marker='o', linestyle='-', color='b')
axs[0].set_title('Sharpe Ratio vs Lambda (Fixed Beta = 2)')
axs[0].set_xlabel('Lambda')
axs[0].set_ylabel('Sharpe Ratio')
axs[0].grid(True)

# Sharpe Ratio vs Beta
axs[1].plot(beta_values, sr_beta, marker='s', linestyle='-', color='r')
axs[1].set_title('Sharpe Ratio vs Beta (Fixed Lambda = 0.01)')
axs[1].set_xlabel('Beta')
axs[1].set_ylabel('Sharpe Ratio')
axs[1].grid(True)

plt.tight_layout()
plt.show()

#%%

"""
Implement and code the Deep Learning Algorithm in for discrete setting in Appendix C.2
and visualize the training loss for different network structures in Appendix C.2. (33 pt)

"""

INPUT_SIZE = 2       # (fn-1, J0_n-1)
HIDDEN_SIZES = [128, 32, 8]
OUTPUT_SIZE = 1      # Qn

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 1024
SEQUENCE_LENGTH = 5120
EPOCHS = 50


# Other constants
alpha =  1.67e-4
sigma = 0.02
Phi = 0.139  
N = 5120          
dt= 0.01    
T = N*dt       
lam = 0.01     
p_list = 0.5      
beta = 2       
gamma=3         

alpha_bar=alpha*(1-np.exp(-beta*dt))/(beta*dt)
# Scaling factors
SIGNAL_SCALE = 1e6
#%%
def generate_signals(batch_size, sequence_length):
    f = np.zeros(N+1)
    for i in range(sequence_length):
        f[i+1] = np.exp(-Phi * dt)*f[i] + np.sqrt(1-np.exp(-2*Phi * dt))*np.random.normal(0, 1)
    return f


def update_J0(J0_prev, Q_prev):
    exp_term = torch.exp(-beta * dt)
    J0 = exp_term * J0_prev - (1 - exp_term) * Q_prev
    return J0

class NetSimple(nn.Module):
    def __init__(self):
        super(NetSimple, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[2], OUTPUT_SIZE)
        )
    
    def forward(self, fn_1, J0_n_1):
        x = torch.stack((fn_1 * SIGNAL_SCALE, J0_n_1), dim=1)
        Qn = self.network(x).squeeze()
        return Qn

def train_net_simple():
    net = NetSimple()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    net.train()


    for epoch in range(EPOCHS):
        for batch in range(50):  # 50 batches per epoch
            signals = generate_signals(BATCH_SIZE, SEQUENCE_LENGTH)
            fn_1 = signals[:, :-1]
            J0_prev = torch.zeros(BATCH_SIZE)
            optimizer.zero_grad()
            total_objective = 0

            for t in range(SEQUENCE_LENGTH):
                Q_curr = net(fn_1[:, t], J0_prev)
                J0_curr = update_J0(J0_prev, Q_curr)
                J_curr = Q_curr + J0_curr

                # Compute terms of the objective function
                reward = alpha_bar * Q_curr * fn_1[:, t]
                cost = gamma * sigma**2 / 2 * Q_curr**2
                risk = lam * beta * torch.abs(J_curr)**(p_list + 1)
                objective = reward - cost - risk
                total_objective += objective.mean()

                # Update previous values
                J0_prev = J0_curr.detach()

            # Maximize total objective
            loss = -total_objective / SEQUENCE_LENGTH
            loss.backward()
            optimizer.step()

    return net


def evaluate_network(net, num_steps=N):
    net.eval()
    with torch.no_grad():
        signals = generate_signals(1, num_steps)
        fn_1 = signals[:, :-1].squeeze()
        J0_prev = torch.zeros(1)
        total_reward = 0
        total_cost = 0
        total_risk = 0

        for t in range(num_steps - 1):
            Q_curr = net(fn_1[t], J0_prev)
            J0_curr = update_J0(J0_prev, Q_curr)
            J_curr = Q_curr + J0_curr

            # Compute terms of the objective function
            reward = alpha_bar * Q_curr * fn_1[:, t]
            cost = gamma * sigma**2 / 2 * Q_curr**2
            risk = lam * beta * torch.abs(J_curr)**(p_list + 1)

            total_reward += reward.item()
            total_cost += cost.item()
            total_risk += risk.item()

            # Update previous values
            J0_prev = J0_curr

        total_objective = total_reward - total_cost - total_risk
        sharpe_ratio = total_objective / np.sqrt(total_risk)

    return sharpe_ratio
#%% 
net_simple = train_net_simple()
sr_simple = evaluate_network(net_simple)
print(f"NetSimple Sharpe Ratio: {sr_simple:.4f}\n")