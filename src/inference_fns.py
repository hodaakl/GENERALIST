import torch 
def calc_deri(z,t, sigmas, lambda_ = 0):
    """Get the derivative of the log likelihood with respect to z and theta
    input: Z 
            t (theta)"""
    # get pi
    g= torch.einsum("nk,dkl -> dnl",z,t)
    pi = torch.exp(- g)/ torch.sum(torch.exp(- g), axis =0 ).unsqueeze(0)
    # print(torch.sum(pi, axis = 0))
    error = pi - sigmas
    dLdz = torch.einsum("dkl,dnl -> nk",t.double(), error.double()) - 2*lambda_*z
    dLdt = torch.einsum("nk,dnl -> dkl",z.double(), error.double()) - 2*lambda_*t
    #
    return dLdz, dLdt , g

def calc_loglikelihood(sigmas,g):
    """calculates log likelihood likelihood given g - Z*theta and the data:sigmas """
    L = - torch.sum(torch.multiply(sigmas, g)) - torch.sum(torch.log(torch.sum(torch.exp(- g),axis = 0)))
    return L 

def adaptive_newparams(z,t,mz,mt, vz,vt, der_z, der_t, beta1 =.8 , beta2  = .999, alpha = .1, i = 1, eps = 1e-8, optimizer  = 'adam'):
    """Calculates the new parameters using the old parameters through ADAM algorithm"""
    mz = beta1*mz + (1-beta1)*der_z
    mt = beta1*mt + (1-beta1)*der_t
    if optimizer == 'adam':
        vz = beta2*vz + (1-beta2)*der_z**2
        vt = beta2*vt + (1-beta2)*der_t**2
    elif optimizer == 'yogi':
        vz = vz - (1-beta2)*torch.sign(vz - der_z**2)*der_z**2
        vt = vt - (1-beta2)*torch.sign(vt - der_t**2)*der_t**2
        pass
    ##### unbias 
    mzhat = mz/(1-beta1**(i+1))
    mthat  = mt/(1-beta1**(i+1))

    vzhat = vz/(1-beta2**(i+1))
    vthat = vt/(1-beta2**(i+1))
    z = z + alpha*mzhat/(torch.sqrt(vzhat) + eps)
    t = t + alpha*mthat/(torch.sqrt(vthat) + eps)
    return z,t , mz, mt, vz, vt