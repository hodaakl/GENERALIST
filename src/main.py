"""In this script you can run inference on model "generalist" -- optimization uses adam algorithm"""
############################################ ONE HOT ENCODE DATA #####################################################
from data_process_fns import Convert_fastaToNp
FastaFilePath  = '../data/msa_p53_unimsa.fa' # file path of the fasta file 
data_one_hot = Convert_fastaToNp(filepath = FastaFilePath, binary = True, labels_inc =True)
############################################ RUN INFERENCE #####################################################
from generalist_class import Generalist
k= 32 # decide on the latent dimension for the model
generalist = Generalist(data_one_hot,k )# parameters are saved in the directory output_directory 
generalist.train() #use_gpu_if_avail = True
############################################ GENERATE NEW SEQUENCES ############################################
ngen = 1000
generated_data = generalist.generate(nGen = ngen) # output_binary = False
print(generated_data.shape)
### save the generated dataset as numpy matrix 
file_name = f'generated_data.npy'
import numpy as np
np.save(file_name, generated_data)
### save the generated dataset as a fasta file 
# from data_process_fns import Conv_save_NpToFasta
# Conv_save_NpToFasta(generated_data, path=f'generated_set_k{k}.fa',  label = 'generated_seq', verbose = False , save = True, include_fish_sym = True)
