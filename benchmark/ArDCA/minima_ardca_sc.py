### import packages 
import numpy as np 
from DataProcess import Numerify
from prob_ardca_fns import get_stable_seq
from collections import defaultdict
## Load the BPT1 numerical matrix 
protein = 'BPT1'
data = np.load(f'/blue/pdixit/hodaakl/A4PSV/ProteinFamilies/sigmas_{protein}_unique.npy')
data = Numerify(data)
data = data.astype(int)
## LOAD J H AND P0 
## Load the npz files outputted from the Julia code for BPT1 protein 
path_to_dir = f'/blue/pdixit/hodaakl/A4PSV/ArDCA/EqWeights/'
J = np.load(f'{path_to_dir}J_{protein}_ardca_eqW.npz')
H = np.load(f'{path_to_dir}H_{protein}_ardca_eqW.npz')
p0= np.load(f'{path_to_dir}p0_{protein}_ardca_eqW.npz')
idx_entropic=  np.load(f'{path_to_dir}idxperm_{protein}_ardca.npz') - 1
#
natidx = np.load(f'/blue/pdixit/hodaakl/A4PSV/ArDCA/BPT1_indices_for_minima_search_5000_NOGAP.npy')
# 
L = data.shape[1]
#
stab_factor= 1000
batch_idx= 2
stab_lim = stab_factor*L
# initialize the probability vector and the stable sequence dictionary
prob_list = [] 
stable_seq_dict = defaultdict() 
m=0 
output_directory = f'/blue/pdixit/hodaakl/A4PSV/Results_Data/{protein}/seq_prob_and_local_minima_dir/ArDCA_minima/'
batch_size=500 #process 500 sequences each time the script runs
start_idx=batch_idx*batch_size ; end_idx = start_idx+batch_size
print(start_idx, end_idx)
print(f'stability limit = {stab_lim}')
print(f'batch  {batch_idx}')
for s_idx in natidx[start_idx:end_idx]:
    seq_init = data[s_idx,:] #initial sequence 
    result  = get_stable_seq(seq = seq_init, p0 = p0 , J = J, H = H, idx_entropic = idx_entropic, stab_lim =  stab_lim) #Find 
    if result == None:
        pass
    else:
        stableseq, p = result
        prob_list.append(p)
        stable_seq_dict[f'natidx_{s_idx}'] = stableseq

    # every 50 sequences save the probability and the sequences 
    print(stable_seq_dict)
    if m%50==0:
        path = f'{output_directory}stable_probability_stabfactor_{stab_factor}L_batchidx_{batch_idx}.npy'
        np.save(path, np.asarray(prob_list))
        path = f'{output_directory}stable_seqs_stabfactor_{stab_factor}L_batchidx_{batch_idx}.npy'
        np.save(path, stable_seq_dict)


    m+=1 
    print(f'{m}. done with natural sequence ')

## SAVE NOW
path = f'{output_directory}stable_probability_stabfactor_{stab_factor}L_batchidx_{batch_idx}.npy'
np.save(path, np.asarray(prob_list))
path = f'{output_directory}stable_seqs_stabfactor_{stab_factor}L_batchidx_{batch_idx}.npy'
np.save(path, dict(stable_seq_dict), allow_pickle = True)
