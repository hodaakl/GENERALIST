import numpy as np
import time
import os
import torch  # torch einsum function is much faster than numpy einsum. Install torch for fastest inference, otherwise , in the code below,  change torch. to np. and the code should run fine.
from inference_fns import  calc_deri, calc_loglikelihood, adaptive_newparams
from data_process_fns import Convert_fastaToNp , write_file , Numerify

def generalist(fasta_path, k , out_dir,  save_step = 100, thresh = 1, alpha = .01,steps = int(10e7), labels_inc = True): 
    """function that runs the inference -- optimization uses adam algorithm
        Input:
        fasta_path:  fasta file (msa) of protein family 
        k : latent dimension used in the model
        out_dir : output directory : if doesn't exist the function will create that directory
        save_step (default 100): save the parameters every save_step steps , in case the jobs got interrupted 
        thresh (default 1) : inference steps when |grad(Z)|/|Z| is less than thresh , same for theta  # threshold for stopping 
        alpha (default .01) : for the adaptive learning rate
        steps (default 10^7 ) : maximum number of allowed steps
        labels_inc (default true) : in the msa there are labels for sequences not only sequences

        Outputs: 
        Written as files in the output directory: 
            Z: written as numpy and torch array in the output directory
            Theta: written as numpy and torch array in the output directory
            details: saves details of the inference  
        
    """
    ## cpu or gpu 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('running on' , device)
    # define date
    # make the output folder if it does not exist
    isdir = os.path.isdir(out_dir) 
    if isdir ==False: 
        # if not , make output folder 
        os.mkdir(out_dir)
    # create the one hot encoded data : sigmas from the fasta file
    sigmas = Convert_fastaToNp(fasta_path)
    print(f'one hot encoded data of size {sigmas.shape}')
    nA,nS,nP = sigmas.shape # nA number of categories , nS : number of sequences, nP : length of the sequence (number o fpositions)
    sigmas = torch.from_numpy(sigmas) ; sigmas = sigmas.to(device) 
    # load the latest Z and Theta for that speific K if that specific K has run before 
    input_path_Z = out_dir + f'/Z_k{k}'
    input_path_T =  out_dir + f'/T_k{k}'
    filexists = os.path.exists(input_path_Z)
    if filexists==True:

        z = torch.load(input_path_Z, map_location=torch.device(device)) ; z = z.to(device)
        t = torch.load(input_path_T,map_location=torch.device(device) ) ; t = t.to(device)
    else: 
        z = torch.rand(nS, k) ; z = z.to(device)
        t = torch.rand(nA, k, nP) ; t = t.to(device)
    ### DEFINE FILENAMES
    filename = out_dir + f'/run_details.txt'
    filename_LL = out_dir + f'/Larr_k{k}'
    filename_Z = out_dir + f'/Z_k{k}'
    filename_T = out_dir + f'/T_k{k}'
    filename_Zgrad = out_dir + f'/Z_grad_k{k}'
    filename_Tgrad = out_dir + f'/T_grad_k{k}'
    #####
    # DEFINE SOME PARAMS 
    beta1 = .8      # for adaptive learning rate
    beta2 = .999    # for adaptive learning rate
    eps = 1e-8      # for adaptive learning rate
 
    i=0             # counter 
    bng = 10        # initializing stopping criteria
    eng = 10        # initializing stopping criteria
    save_step=100   # save step
    ###
    mz = torch.zeros(z.shape)  ; mz = mz.to(device)  # for adaptive learning rate
    vz = torch.zeros(z.shape)  ; vz = vz.to(device)  # for adaptive learning rate
    mt = torch.zeros(t.shape)  ; mt = mt.to(device) # for adaptive learning rate
    vt = torch.zeros(t.shape)  ; vt = vt.to(device) # for adaptive learning rate
    Larr = torch.zeros(steps)  ; Larr = Larr.to(device)  # initializing the log likelihood array 
    bng_arr = torch.zeros(steps) ; bng_arr = bng_arr.to(device)# B norm grad
    eng_arr = torch.zeros(steps) ; eng_arr = eng_arr.to(device)# E norn grad
    # 
  
    details = f'Adaptive algorithm\nwhile loop until norm(gradZ/T)/norm(Z/T)<{thresh}\nNon variational run \n USING ADAM ALGORITHM\n alpha = {alpha}\nk={k}\nsaving every {save_step} steps\nmaximum steps ={steps}'
    write_file(filename, details)
    t0 = time.time()

    while (bng > thresh) or (eng > thresh):
        if i==0:
            print('started inference..')
        # calculate derivative 
        der_z, der_t, g = calc_deri(z,t,sigmas)
        # change the parameters 
        z, t , mz, mt, vz, vt= adaptive_newparams(z,t,mz,mt,vz,vt, der_z, der_t, beta1 =beta1 , beta2  =beta2, alpha = alpha, i = i, eps = eps)
        # get the stopping criteria 
        bng = torch.linalg.norm(der_z)/torch.linalg.norm(z)
        eng = torch.linalg.norm(der_t)/torch.linalg.norm(t)
        # save the log likelihood and stopping criteria 
        Larr[i] = calc_loglikelihood(sigmas,g)
        bng_arr[i] = bng 
        eng_arr[i] = eng
        i=i+1 # add 1 to counter 
        # 
        if i==1:
            print(f'takes {time.time()-t0} seconds for the first step')    
        # if it is time to save 
        if i%save_step==0: 
            torch.save(Larr[:i], filename_LL )
            torch.save(z, filename_Z)
            torch.save(t, filename_T)
            torch.save(bng_arr[:i], filename_Zgrad)
            torch.save(eng_arr[:i], filename_Tgrad)
        if i==steps:
            # if reached maximum steps 
            print('reached maximum steps, quitting inference -- consider increasing the max number of steps allowed')
            break    
    t1 = time.time()
    print(f'Inference is over')
    print(f'step takes {(t1-t0)/i} seconds ')
    ### Truncate the arrays 
    Larr = Larr[:i]
    bng_arr= bng_arr[:i]
    eng_arr=eng_arr[:i]
    # save tensor arrays 
    torch.save(Larr, filename_LL )
    torch.save(z, filename_Z)
    torch.save(t, filename_T)
    torch.save(bng_arr, filename_Zgrad)
    torch.save(eng_arr, filename_Tgrad)
    ##### 
    # convert everything to a numpy array to save for loading from the analysis functions 
    # if device is gpu -- take it to cpu 
    if device.type == 'cuda': 
        # transfer the tensors to gpu to convert them to numpy 
        Larr = Larr.cpu() ; z = z.cpu() ; t = t.cpu() ; bng_arr = bng_arr.cpu() ; eng_arr = eng_arr.cpu()


    # convert to numpy array because the analyzer code loads numpy arrays not torch tensors  
    Larr = Larr.numpy()
    z = z.numpy()
    t = t.numpy()
    bng_arr = bng_arr.numpy()
    eng_arr = eng_arr.numpy()
    sigmas = sigmas.numpy()
    # SAVE THE NUMPY ARRAYS 
    np.save(filename_LL , Larr)
    np.save(filename_Z, z)
    np.save(filename_T, t)
    np.save(filename_Zgrad  , bng_arr)
    np.save(filename_Tgrad, eng_arr)
    print(f'Saved files for k = {k}')
    details = f'Inference ran on{device}\nAlgorithm took {(t1-t0)/60} minutes to run - ran for {i} steps\nAdaptive algorithm for Protein is\nUSING ADAM ALGORITHM\nwhile loop until norm(gradB/E)/norm(B/E) < {thresh}\nalpha = {alpha}\nk = {k}\nsaving every {save_step} steps\nmaximum steps = {steps}'
    write_file(filename, details)
    print(f'Updated details files')
    print(f'inference done in {(t1-t0)/60} minutes for k = {k}\n output files at directory {out_dir}')
    return 
