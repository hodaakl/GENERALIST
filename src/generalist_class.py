import torch 
import numpy as np
import time
from inference_fns import calc_deri , calc_loglikelihood , adaptive_newparams
from data_process_fns import Binarify 
# define the class 

class Generalist:
    def __init__(self, one_hot_data, k,  z_init =  torch.Tensor([]), t_init = torch.Tensor([])):
        """Initializing the instance 
        Input: 
            one_hot_data : of shape (number of categories, number of samples, number of features)
            k: user chosen latent dimension for the model
            z_init : torch tensor of shape (number of samples, k) specified by the user to initialize z 
            t_init : torch tensor of shape (number of categories, k, number of features) specified by the user to initialize theta
        Attributes: 
            self.nA : number of categories from the one hot encoded data
            self.nS : number of samples
            self.nP : number of features/ positions in a sequence
            self.sigmas : one_hot_data
            self.k : latent dimension of the model 
            self.z : model parameters z
            self.t : model paraemters theta
            self.pi: model probabilities """
        if  (len(one_hot_data.shape) != 3 or ((one_hot_data==0) | (one_hot_data==1)).all() == False):# raise error if data is not one hot encoded
            raise ValueError("Input data must be one hote encoded of shape (number of categories, number of samples, number of features)")
        ## 
        one_hot_data = torch.from_numpy(one_hot_data)
        self.nA, self.nS, self.nP = one_hot_data.shape
        self.sigmas = one_hot_data
        self.k = k
        ## Parameter initialization, if the user specified any of the parameters     
        init_shapez =  z_init.shape[0]
        init_shapet = t_init.shape[0]
        minlim = -1 ; maxlim = 1
        if init_shapez != 0: 
            z = z_init
        else: # if user didn't specify parameters then randomly initialize
            z = (minlim - maxlim)* torch.rand(self.nS,k) + maxlim ; z = z/torch.linalg.norm(z) 
        if init_shapet != 0:
            t = t_init
        else: # if user didn't specify parameters then randomly initialize
            t = (minlim - maxlim)* torch.rand(self.nA, k, self.nP) + maxlim; t = t/torch.linalg.norm(t)
        # calculate probability matrix according to the model
        exponent=  torch.einsum("nk,dkl -> dnl",z,t)
        pi = torch.exp( - exponent)/ torch.sum(torch.exp(-  exponent), axis =0 ).unsqueeze(0)
        self.z = z ; self.t = t ; self.pi = pi  
        
    def train(self, thresh = 1, alpha = .01,steps = int(10e7), optimizer ='adam',  use_gpu_if_avail = True, beta1 = .8, beta2 = .999 ,eps = 1e-8  ,zng = 10, tng = 10 , verbose = True, lambda_=0):
        """method to train model until the stopping criteria is met : |grad z | /|z| and |grad t |/|t| < 1 
         -- optimization uses adam algorithm
        Inputs: 
            thresh (default 1) : inference steps when |grad(Z)|/|Z| is less than thresh , same for theta  # threshold for stopping 
            alpha (default .01) : for the adaptive learning rate
            steps (default 10^7 ) : maximum number of allowed steps
            use_gpu_if_avail(default true): If GPU available the code defaults to using it. Set to False to use CPU anyway.
            beta1, beta2, eps: for adam optimization algorithm. 
            zng , tng : initialization of the gradient magnitude/parameter magnitude for the stopping criteria bng and eng < 1
        """
        device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu")
        sigmas = self.sigmas
        if verbose == True:
            print(f'running on {device}\none hot encoded data of size {sigmas.shape}')
        z, t = self.z , self.t 
        # sigmas = torch.from_numpy(sigmas) ; 
        sigmas = sigmas.to(device) 
        ### Initialize adam parameters and log likelihood array and gradient arrays
        mz = torch.zeros(z.shape) ; vz = torch.zeros(z.shape) ;mt = torch.zeros(t.shape) ;vt = torch.zeros(t.shape) 
        mz = mz.to(device) ; vz = vz.to(device) ; mt = mt.to(device) ; vt = vt.to(device)
        Larr = torch.zeros(steps)  ; Larr = Larr.to(device)  # initializing the log likelihood array 
        # zng_arr = torch.zeros(steps) ; zng_arr = zng_arr.to(device)# B norm grad
        # tng_arr = torch.zeros(steps) ; tng_arr = tng_arr.to(device)# E norn grad
        z = z.to(device) ; t = t.to(device)
        t0 = time.time()
        i=0             # counter
        while (zng > thresh) or (tng > thresh):
            if i==0:
                if verbose == True:
                    print('started inference...')
             
            der_z, der_t, exponent= calc_deri(z,t,sigmas,lambda_) # calculate derivative
            # update  parameters 
            if (optimizer=='adam' or optimizer =='yogi'):
                z, t , mz, mt, vz, vt= adaptive_newparams(z,t,mz,mt,vz,vt, der_z, der_t, beta1 =beta1 , beta2  =beta2, alpha = alpha, i = i, eps = eps, optimizer = optimizer)

            elif optimizer == 'reg':
                z = z + alpha*der_z
                t = t + alpha*der_t
            # calculate stopping criteria condition
            zng = torch.linalg.norm(der_z)/(torch.linalg.norm(z)); tng = torch.linalg.norm(der_t)/(torch.linalg.norm(t))
   
            Larr[i] = calc_loglikelihood(sigmas,exponent) 

            i=i+1
            # 
            if i==1:
                if verbose == True:
                    print(f'takes {time.time()-t0} seconds for the first step')    
            if i==steps:
                # if reached maximum steps 
                if verbose == True:
                    print('reached maximum steps, quitting inference -- consider increasing the max number of steps allowed')
                break    
        t1 = time.time()
        if verbose == True:
            print(f'inference is over\nstep on avg takes {(t1-t0)/i} seconds ')
        ### Truncate the arrays 
        Larr = Larr[:i]#; zng_arr= zng_arr[:i];  tng_arr=tng_arr[:i]
        ## pi 
        exponent= torch.einsum("nk,dkl -> dnl",z,t)
        pi = torch.exp(- exponent)/ torch.sum(torch.exp(- exponent), axis =0 ).unsqueeze(0)
        self.z  = z; self.t = t; self.pi = pi ;  self.Larr = Larr 
        if verbose == True:
            print(f'inference done in {(t1-t0)/60} minutes for k = {self.k}, in {i} steps')
        return 

    def generate(self, nGen, output_binary = False, output_params=False):
        """method to generate new data
        Input:
            nGen : number of generated samples 
            output_binary: default False ,the generated samples will be numerical unless argument is set to True; then
                they will be one hot encoded """
        list_idx = np.random.randint(0, self.nS, nGen )
        pi = self.pi[:,list_idx,:]
        ## make pi a numpy matrix 
        pi= pi.cpu() ; pi = pi.numpy()
        Gen_num = np.zeros((nGen, self.nP)) #initialize generated data matrix
        aa_idx = np.arange(self.nA) #define categories
        for sample in range(nGen): 
            for pos in range(self.nP): 
                p = pi[:,sample,pos] 
                Gen_num[sample, pos] = np.random.choice(aa_idx, 1, p = p)
        if output_binary ==True: 
            #binarify the output 
            output = Binarify(Gen_num )
        else: 
            output = Gen_num
        if output_params==True:
            output = output, self.z[list_idx,:] , pi
        return output