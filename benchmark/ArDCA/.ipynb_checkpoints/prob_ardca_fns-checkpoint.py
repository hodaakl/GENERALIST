import numpy as np
import time
### FOR ARDCA MODEL 
def sequence_prob_ardca_fn(sequence, p0, entropic_order, J, H, verbose = False):
    """Function that calculates the probability of a sequence using the ArDCA Model 
    sequence:   Sequence of interest , in numerical form where each element is in the set [0,20]
    p0 :        probability of different amino acids the least entropic site used as fist site in the learning, but need not be actual first site 
    entropic order: The ordering of the sites from least entropic to highest entropic used in learnig 
    J : The couplings in dimension [q,q,L,L] where q = 21 and L is the length of the sequence , these are not entropically ordered
    H : single site fields  """
    L = len(sequence)
    seqprob = 1
    #### loop over positions 
    for i in range(L):
        idx = entropic_order[i]
        aa = sequence[idx]
        
        ### 
        if i==0:
            probability_array = p0
        else:
            js = entropic_order[:i]
            ajs = sequence[js]
            Jsum_v = np.sum(J[:, ajs, idx, js], axis = 1)
            h_v = H[:, idx]
            prob_top = np.exp(h_v + Jsum_v ) #numerator 
            zsite = np.sum(prob_top) # parition function (dinomenator)
            probability_array = prob_top/zsite
        seqprob = seqprob*probability_array[aa]
        if verbose == True:
            print(f'sanity check \n sum probabilities is {np.sum(probability_array)} ')
    return seqprob


def get_stable_seq(seq, p0, J, H , idx_entropic, stab_lim , gap_index = 20 ):
    """Function that gets the local minima around a sequence by maximizing the probability through doing 
    mutations and a process of acceptance and rejection 
    INPUTS: 
        seq: Starting sequence
        p0:  probability of different AA at least entropic site - ardca parameter
        J:   couplings - ardca parameter
        H:   single site fields - ardca parameter
        idx_entropic: index values to indicate entropic order  - ardca 
        stab_lim: wait steps to call the sequence 'stable'
        gap_idx: index to avoid mutating to, gap index 
        
    OUTPUTS: 
        seq: 'stable' sequence
        p: proabability of that sequence """
    aa_idx = np.arange(0,gap_index)
    stab_count = 0 ; reg_count = 0 # initialize counters
    L = len(seq) #define length of sequence
    p = sequence_prob_ardca_fn(sequence = seq, p0 = p0, entropic_order = idx_entropic, J = J, H= H) #calc probability of that sequence
    ## while loop conditioned on stab_count
    t0 = time.time()
    while stab_count<stab_lim:
        mut_idx = np.random.randint(0,L) #pick mutation index
        aa_exist = seq[mut_idx] #find the amino acid that exists there 
        if aa_exist ==gap_index: # if there was a gap in that position
            print(f'mutating a gap to an amino acid')
            aa_possible = aa_idx # 
        else:
            aa_possible = np.delete(aa_idx, aa_exist) # all possible mutations to other amino acids
        mut_aa = np.random.choice(aa_possible) #pick a random amino acid for mutation
        seq_new = seq.copy() #new sequence , copy the og one
        seq_new[mut_idx] = mut_aa #mutate position mut_idx to amino acid mut_aa
        p_new = sequence_prob_ardca_fn(sequence = seq_new, p0 = p0, entropic_order = idx_entropic, J = J, H= H) #calc probability of that sequence
        if p_new > p: 
            # accept this new sequence
            stab_count=0 #reset stabilizer count
            p = p_new.copy()
            seq = seq_new.copy()
        else:
            stab_count +=1 # add to stabililizer count
        reg_count+=1 
        if stab_count ==stab_lim:
            print('sequence stabilized, breaking loop.')
            print(f' sequence took {time.time() - t0} to stabililize')
            # print('\n\n')
            output =  seq , p
            
            
        if (reg_count%1000==0 ):
            print(f'tried {reg_count} mutations ... stab_count = {stab_count}')
        if reg_count==int(100*stab_lim):
            print(f'tried {reg_count} mutations ... quitting loop')
            print(f' sequence took {time.time() - t0} to quit')
            output = None
            break
    return output


# THE FOLLOWING FUNCTIONS ASSUME THAT THE J H AND SEQUENCE AND P0 are in the correct order  
def CondProbForSite_fn(seq, site, J, H):
    """Function that calculates the conditional probability P(any amino acid at site| amino acids happening in previous sites)
    input:  seq (the sequence used to obtain the probability, relevant in determining which amino acids happened in previous sites)
            site (the site we want to calculate the conditional probability at )
            J , H (paramters from ArDCA code in Julia by Trinquire et. at 2021)
    output: an array of the dimension 21 giving the probabilities of any amino acid hapening at that site"""
    if site ==0:
        raise ValueError('for site =0 use p0, site can not be zero for this function')
    # an array of the previous positions
    js = np.arange(0,site)
    # sequences hapening before
    ajs = seq[js]
    # get the terms used in the probability , sum J and H
    Jsum_v = np.sum(J[:, ajs, site, js], axis = 1)
    h_v = H[:, site]
    # Get probability 
    prob_top = np.exp(h_v + Jsum_v ) #numerator 
    zsite = np.sum(prob_top) # parition function (dinomenator)
    prob = prob_top/zsite
    return prob
    
def SeqProbability_fn(seq, p0, J, H, verbose= False):
    """Function that calculates the probability of a sequence given the ARDCA model inferred parameters J and H 
    input:  seq : sequence to be scored 
            p0 : probabilities of first site -- obtained from ArDCA model 
            J , H : obtained from ArDCA model"""
    L = len(seq) # number of positions in the sequence 
    # for every sequence find the probability
    seqprob=1
    for position in range(L):
        # find out the amino acid happening there 
        aa = seq[position]
        if position ==0:
            prob_arr = p0
        else: 
            prob_arr = CondProbForSite_fn(seq, position, J, H)

        seqprob = seqprob*prob_arr[aa]
        if verbose==True:
            print(f'sanity check \n sum probabilities is {np.sum(prob_arr)} ')
    return seqprob