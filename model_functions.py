import numpy as np



class model:

    def __init__(self,timestep,spatial_step,total_time,total_space, param):
        
        #Note that the param = [a,b,nu]
        self.delta_t = timestep 
        self.total_time = total_time
        self.param = param
        self.delta_x = spatial_step

        self.nx = int(total_space/spatial_step) #The number of spatial steps
        self.nt = int(total_time/timestep) #Number of timesteps!

        #Note that param = [a,b] and we must have that a > 2b
        if param[0]< 2*param[1]:
            raise Exception('To be initially stable a must be larger')

        #Now filling in the DU and DW fixed matrices!

        DU = np.zeros((self.nx,self.nx))
        np.fill_diagonal(DU,-2*self.delta_x**-2) #This is the main diagonal
        np.fill_diagonal(DU[1:],self.delta_x**2) #This is the upper diagonal
        np.fill_diagonal(DU[:,1:],self.delta_x**2) #This is the lower diagonal
        #Boundary conditions
        DU[0,-1] = self.delta_x**2
        DU[-1,0] = self.delta_x**2
        #Filling in the DW matrix

        DW = np.copy(DU)
        np.fill_diagonal(DW,-self.param[2]*self.delta_x**-1)
        #The main diagonal
        np.fill_diagonal(DU[1:],self.param[2]*self.delta_x**-1)
        #The upper diagonal

        DW[0,-1] = self.param[2]*self.delta_x**-1 #Boundary conditions.

        self.DU = DU
        self.DW = DW

        #Setting up the following matrices
        self.KU_inv = np.linalg.inv(np.identity(self.nx) - 0.5*self.delta_t*self.DU)
        self.KW_inv = np.linalg.inv(np.identity(self.nx) - 0.5*self.delta_t*self.DW)

        self.MU = np.identity(self.nx) + 0.5*self.delta_t*self.DU
        self.MW = np.identity(self.nx) + 0.5*self.delta_t*self.DW





    def FG_func(self,u,w):
        #Note that u and w are N*1 vectors! and we are doing element wise multiplication. F = w*u*u - b*u, where * is element wise multiplication

        f = np.zeros((self.nx,1))
        g = np.zeros((self.nx,1))
        a_vec = np.ones((self.nx,1))*self.param[0] #A vector

        f = w*u*u - self.param[1]*u 
        g = a_vec - w - w*u*u

        return f,g

    def u_w_approx(self,u,w):
        #Identity matrix of size Nx by Nx

        

        u_approx = self.KU_inv@(self.MU@u+self.delta_t*self.FG_func(u,w)[0])
        w_approx = self.KW_inv@(self.MW@w+self.delta_t*self.FG_func(u,w)[1])

        return u_approx,w_approx
    
    
    def u_w_next(self,u,w):

        u_approx, w_approx =  self.u_w_approx(u,w)

        u_next = self.KU_inv@(self.MU@u+0.5*self.delta_t*(self.FG_func(u,w)[0]+self.FG_func(u_approx,w_approx)[0]))

        w_next = self.KW_inv@(self.MW@w+0.5*self.delta_t*(self.FG_func(u,w)[1]+self.FG_func(u_approx,w_approx)[1]))


        return u_next, w_next 
    


    

        