"""In this script you can run the function "generalist" -- optimization uses adam algorithm
    Input:
    fasta_path:  fasta file (msa) of protein family 
    k : latent dimension used in the model
    out_dir : output directory : if doesn't exist the function will create that directory
    save_step (default 100): save the parameters every save_step steps , in case the jobs got interrupted 
    thresh (default 1) : inference steps when |grad(Z)|/|Z| is less than thresh , same for theta  # threshold for stopping 
    alpha (default .01) : for the adaptive learning rate
    steps (default 10^7 ) : maximum number of allowed steps

    Outputs: 
    Written as files in the output directory: 
        Z: written as numpy and torch array in the output directory
        Theta: written as numpy and torch array in the output directory
        details: saves details of the inference  
"""
############################################ RUN INFERENCE #####################################################
from compile_generalist_fn import generalist
# necessary arguments for generalist
FastaFilePath  = '../data/msa_p53_unimsa.fa' # file path of the fasta file 
k= 2 # decide on the latent dimension for the model
output_directory = f'../p53_output_k{k}/' # the output directory where the parameters will be saved 
#This function should detect a gpu if available, otherwise runs on cpu
generalist(fasta_path = FastaFilePath, k = k, out_dir = output_directory) # 
# parameters are saved in the directory output_directory 
############################################ GENERATE NEW SEQUENCES ############################################
from loader_class import Generator
ngen = 1000
Gen_obj = Generator(output_directory, k = k)
generated_data = Gen_obj.GenData(ngen) #numpy array
print(generated_data.shape)
### save the generated dataset as a fasta file 
from data_process_fns import Conv_save_NpToFasta
Conv_save_NpToFasta(generated_data, path= output_directory + f'generated_set_k{k}.fa',  label = 'generated_seq', verbose = False , save = True, include_fish_sym = True)
