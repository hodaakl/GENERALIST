from os import set_blocking
import numpy as np
from data_process_fns import Binarify

class Generator: 
    def __init__(self, output_dir, k): 
        """Load Larr, KL, LB, latest_idx, Zmu, Zsigma, E""" 
        self.Larr = np.load(output_dir + f'/Larr_k{k}.npy' ) # this needs to change 
        self.latest_idx= np.where(self.Larr<0)[0].shape[0]
        self.Z = np.load(output_dir +  f'/Z_k{k}.npy')
        self.Theta = np.load(output_dir +  f'/T_k{k}.npy')



    def GenData(self, nGen, output_binary = False, output_params=False): 
        """Generates nGen samples from the inferred Z, theta"""
        Z = self.Z
        Theta = self.Theta
        nS = Z.shape[0]
        nA,k, nPos = Theta.shape
        list_idx = np.random.randint(0, nS, nGen )
        Z_ss = Z[list_idx,:]
        gamma = np.matmul(Z_ss,Theta)
        e_gamma = np.exp(-gamma)
        eg_sd = np.sum(e_gamma, axis = 0)
        pi = e_gamma/eg_sd
        # Generating sequences from probabilities 
        Gen_num = np.zeros((nGen, nPos))
        aa_idx = np.arange(nA)
        for sample in range(nGen): 

            for pos in range(nPos): 
                p = pi[:,sample,pos] 
                Gen_num[sample, pos] = np.random.choice(aa_idx, 1, p = p)

        if output_binary ==True: 
            #binarize the output 
            output = Binarify(Gen_num )
        else: 
            output = Gen_num

        if output_params==True:
            output = output, Z_ss , Theta, pi
        
        return output