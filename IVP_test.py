import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import tqdm

n = 100
L = 100 #Domain of size 100
dx = L/n #The size of the spatial step
nu = 180 #The advection coefficient 
x = np.arange(0,L,dx) #Setting

print(len(x))

def MOL_matrices(n,dx,diff_constant,adv_constant):#The construction of the Method Of Lines matrices for First order and Second Order
    SOM = np.zeros((n,n))
    FOM = np.zeros((n,n))

    np.fill_diagonal(SOM,-2)
    np.fill_diagonal(SOM[1:], 1)
    np.fill_diagonal(SOM[:, 1:], 1)
    SOM[0,-1] = 1
    SOM[-1,0] = 1


    np.fill_diagonal(FOM,-1)
    #np.fill_diagonal(FOM[1:], 1)
    np.fill_diagonal(FOM[:,1:],1)
    FOM[-1,0] = 1

    return (SOM*dx**(-2))*diff_constant,FOM*(dx**(-1))*adv_constant

def Gaussian_fit(x,sigma,mean,scale): #Fitting some Normal distribution to the U and W vectors

    return scale*10*((sigma*np.sqrt(2*math.pi))**(-1))*np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def noise_fit(x,noise_scale):
    #adding positive noise from 0
    return noise_scale*np.random.rand(len(x))


#u_init = Gaussian_fit(x,sigma=4,mean=L//2,scale=2)
#w_init = Gaussian_fit(x,sigma=4,mean=L//2,scale=2)
u_init = noise_fit(x,noise_scale=0.4)
w_init = noise_fit(x,noise_scale=0.4)

DU,DW = MOL_matrices(n,dx,diff_constant=1,adv_constant=nu) #First order and second order matrices!
dudt = np.zeros_like(u_init)
dudt = np.zeros_like(w_init)


def spatial_deriv(t,y_vector):#This is returning the spatial derivative
    dudt = DU@y_vector[:n] 
    dwdt = DW@y_vector[n:] #The w vector
    return np.concatenate([dudt, dwdt])
def solve_diff(fun,t_span,u_init,w_init,resolution):#This is solving over time (using solve_ivp)
    y_vector = np.concatenate([u_init,w_init])
    sol = solve_ivp(fun, t_span, y_vector,method='RK45', t_eval=np.linspace(t_span[0], t_span[1], resolution))
    u_solution = sol.y[:n, :]
    w_solution = sol.y[n:, :]
    return u_solution,w_solution


a = 2.2
b = 0.45

w_init = w_init + a
u_init = u_init + 1
def klaus_model(t,y_vector):
    dudt = np.multiply(y_vector[n:],np.square(y_vector[:n]))-b*y_vector[:n]+DU@y_vector[:n]
    dwdt = a-y_vector[n:]-np.multiply(y_vector[n:],np.square(y_vector[:n]))+DW@y_vector[n:]
    return np.concatenate([dudt,dwdt])

t_span = [0,200]
u_sol,w_sol = solve_diff(klaus_model,t_span,u_init,w_init,resolution = 10)
plt.figure()
plt.plot(x,u_sol[:,-1])
"""for i in range(10):
    plt.plot(x,u_sol[:,i])
"""
plt.ylim([0,20])
plt.show()

print(DW)