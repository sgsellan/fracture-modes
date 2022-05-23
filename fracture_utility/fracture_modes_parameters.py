class fracture_modes_parameters:
    def __init__(self,num_modes=10,d=1,max_iter=10,tol=1e-4,omega=0.01,verbose=False):
        self.num_modes = num_modes
        self.d = d
        self.max_iter = max_iter
        self.tol = tol
        self.omega = omega
        self.verbose = verbose