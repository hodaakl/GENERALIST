from statistics import mean
import numpy as np
import random
import time
from data_process_fns import Numerify, Binarify


def Calc_nBody_Corr(Data, Gen, n=2, nTest = 7000, mean_removed = True, weights = np.array([1])):
    """ Gets two arrays of n site correlation
        Input:  Data : original data in  categorical (integer), or binary form  
                Gen : generated data in  categorical (integer), or binary form 
                n: order of statistics 
                binary: default True, set false if data is integer. Function numerifies data anyway. 
                nTest:  default 7000, number of combinations of amino acids and positions tested
                mean_removed: default True. Calculates < delta sigma_i > < delta sigma_j > <delta sigma_k> 
        
        output: Array of two site correlation of data , Array of two site correlation of generated data
        """

    if np.max(Data)!=1: 
        # If data is numerical 
        BinData = Binarify(Data)
        NumData = Data
    else: 
        BinData =  Data
        NumData = Numerify(Data)

    if np.max(Gen)!=1: 
        BinGen = Binarify(Gen)
        # NumGen = Data
    else: 
        BinGen = Gen
        # NumGen = Numerify(Gen)
        
    nS, nPos = NumData.shape

    if (len(weights)>1) and (n==1):
        bb_w = np.tile(weights, [BinData.shape[0],BinData.shape[-1], 1])
        bb_w = np.transpose(bb_w , [0,2,1])
        BinData = bb_w * BinData
    freq_data = np.mean(BinData, axis = 1)    
    freq_gen = np.mean(BinGen, axis = 1)

    if n==1: 
        corr_arr_data = np.reshape(freq_data, (freq_data.shape[0]*freq_data.shape[1],))
        corr_arr_gen = np.reshape(freq_gen, (freq_gen.shape[0]*freq_gen.shape[1],))
    
    else:
        freq_matrix_data = np.transpose(np.tile(freq_data, [BinData.shape[1],1,1]),[1,0,2])
        freq_matrix_gen = np.transpose(np.tile(freq_gen, [BinGen.shape[1],1,1]),[1,0,2])

        # Create the delta matrix 
        DeltaMatrix_data  = BinData - freq_matrix_data 
        DeltaMatrix_gen  = BinGen - freq_matrix_gen 
        # initialize output arrays
        corr_arr_gen = np.zeros(nTest)
        corr_arr_data = np.zeros(nTest)
        # NumData.astype(int)
        for i in range(nTest): 
            # randomly pick a sample n positions 
            sam = np.random.randint(0,nS)
            # randomly pick a position set
            pos_vec = random.sample(list(np.arange(nPos)), n)
            # get the amino acid / category array in those positions in that sample
            aa_vec = NumData[sam,pos_vec]
            aa_vec = aa_vec.astype(int)
            if mean_removed == True:
                Mat_dat = DeltaMatrix_data
                Mat_gen = DeltaMatrix_gen

            else: 
                Mat_dat = BinData
                Mat_gen = BinGen

            if len(weights)>1:
                norm = np.sum(weights)
            else:
                norm = nS
            corr_arr_data[i] = np.sum(weights*np.prod(Mat_dat[aa_vec, :, pos_vec], axis = 0))/norm
            # corr_arr_data[i] = np.mean(weights*np.prod(Mat_dat[aa_vec, :, pos_vec], axis = 0))
            corr_arr_gen[i] = np.mean(np.prod(Mat_gen[aa_vec, :, pos_vec], axis = 0))
        
    return corr_arr_data, corr_arr_gen



def calculate_r_metric(data, gen,  nlist = np.arange(1,11), repititions = 100, m =20, verbose = 1, weights = np.array([1])): 
    """Calcualtes the r_n metric
    Inputs: Data  - Original categorical data in binary (one hot ended) or numerical form
            gen   - generated data from the model in binary (one hot ended) or numerical form
            nlist - list of n order statistics to compare 
            repititions - number of times the pearson coeff correlation will be averages
            m - parameter that specified the number of most frequent combinations to use
            in calculating the statistics  """
    if len(data.shape)>2:
        data = Numerify(data)
        
    if len(gen.shape)>2:
        gen = Numerify(gen)

    # data must be numerified and gen as well 
    nPos = data.shape[1]
    # initialize the output list
    r_n_list = []
    # for each n body correlation in the list
    for n in nlist:
        t0 = time.time()
        r =0
        for rep in range(repititions):
            # choose randomly a position set 
            parr = random.sample(list(np.arange(nPos)), n)
            # get the categories / amino acids at those positions from all the data
            DataComb = data[:,parr]
            # get the unique combinations 
            DataCombUnq = np.unique(DataComb, axis = 0)
            # initialize the frequency array for all those unique combinations
            FreqArr = np.zeros(DataCombUnq.shape[0])
            FreqArr_corrected = np.zeros(DataCombUnq.shape[0])
            # for each unique combination
            for i in range(DataCombUnq.shape[0]):
                # define the combination / word
                SpecSam = DataCombUnq[i,:]
                # get the count of that "word" 
                s = np.sum(DataComb == SpecSam, axis = 1 )
                # print(s.shape)
                idx_true = np.where(s == n)[0]
                ## extract those indices from the weights 
                ## is this the right way to do this 
                FreqArr[i] = len(np.where(s == n)[0])
                if len(weights)>1: 
                    subw = weights[idx_true]
                    ### Check that this is right 
                    FreqArr_corrected[i] = np.sum(subw)
                else: 
                    FreqArr_corrected = FreqArr

            ## this is absolutely not needed, I can get the freq
            # from the generated sequences with the data 
            # and after getting the indices of the most frequent 
            # I can use those indices from the data and the generated sequences 
            #####   
            # get the most frequent m combinations indices 
            ### SHOULDN"T I DO THIS WITH FREQUENCY CORRECTED???
            idx_mf = np.argsort(FreqArr)[-m:]
            # normalize to get the frequency
            FreqArr = FreqArr/np.sum(FreqArr)
            # use only the relevant indices
            Frequencies = FreqArr_corrected[idx_mf]
            # those are the highest m frequencies . 
            ## now get those frequencies in the generated data ! 
            counter = 0
            GenFreq = np.zeros(Frequencies.shape)
            for i in idx_mf:
                # define the combination / word 
                word = DataCombUnq[i,:]
                # get the count
                s = np.sum(gen[:,parr] == word, axis = 1 )
                # get the frequency
                GenFreq[counter] = len(np.where(s == n)[0])/gen.shape[0]
                counter +=1 
            # get the pearson correlation
            r+= np.corrcoef(Frequencies, GenFreq)[0,1]
        # average the pearson correlation 
        r_n = r/repititions
        # append to the list
        r_n_list.append(r_n)
        if verbose ==1: 
            print(time.time() - t0, 'seconds to finish n = ', n)
    return r_n_list


def calc_ham_arr(data, gen , same_data = False): 
    """Calculates the normalized minimum hamming distance for each sequences in the gen dataset to the dataset
    if both datasets are the same, the code removes the examined sequences from the second dataset so that 
    we don't end up with an array of zeros with no information

    Inputs: data: matrix 1 
            gen: matrix 2 
    outputs: an array of minimum normalized hamming distances of length gen.shape[0] """
    if len(gen.shape)==1:
        gen=np.reshape(gen,(1, gen.shape[0]))
        
    if len(data.shape)==1:
        data=np.reshape(data,(1, data.shape[0]))
        
        
    if len(data.shape)>2:
        data = Numerify(data)
    if len(gen.shape)>2:
        gen = Numerify(gen)
    # print(len(np.where(data !=gen)[0]) )
    nS,nP = data.shape
    ngen,_ = gen.shape

    Closest_arr = np.zeros(ngen)

    
    if same_data==True: 
        print('two datasets are the same, removing the examined sequence from comparison')
        for i in range(nS):
            seq = data[i]
            mat = np.tile(seq, [nS-1,1])
            mat2 = np.delete(data, i, axis = 0)
            diff_count =  nP - np.sum(mat == mat2, axis = 1)
            Closest_arr[i] = np.min(diff_count)
            
        
    else:        
        for i in range(ngen):
            gen_sample = gen[i, :]
            gen_matrix = np.tile(gen_sample, [nS,1])
            diff_count =  nP - np.sum(gen_matrix == data, axis = 1)
            Closest_arr[i] = np.min(diff_count)

    return Closest_arr/nP

def cal_ham_randompairs(dataset1, dataset2, number_of_pairs = 1000):
    """Calculates the hamming distance between randomly selected pairs between two protein datasets
    input:  dataset 1 
            dataset 2
            number of pairs 
    output : hamming distance array for the number_of_pairs randomly chosen pairs"""
    ## numerify the data if binary 
    if np.max(dataset1)==1:  
        dataset1 = Numerify(dataset1)
    if np.max(dataset2)==1:  
        dataset2 = Numerify(dataset2)
    # define the number of samples in each dataset , and the number of positions
    n1 , n2 , nP = dataset1.shape[0] , dataset2.shape[0] ,  dataset1.shape[1]
    idxn1 = np.random.randint(0,n1, size = number_of_pairs)
    idxn2 = np.random.randint(0,n2, size = number_of_pairs)
    ## the two new datasets to compare -- picking out pairs 
    dataset1 = dataset1[idxn1, : ]
    dataset2 = dataset2[idxn2, : ]
    ## hamming distance between the both datasets 
    ham_arr = np.sum(dataset1 == dataset2, axis = 1)/nP
    # ## ## ## 
    return ham_arr

def calc_weights_potts(dat, thresh = .8):
    """Calculates the weights that are used to correct the MSA statistics when compared with the DCA model 
    Inputs: dat = numerical/Binary (one hot encoded) matrix for MSA , original data
            thresh = Threshold that will determine the count. 
    Outputs: weights """
    # If the matrix is 3D convert it to 2D Numerical
    if len(dat.shape)>2: 
        dat = Numerify(dat)
    # initialize weights vector - as long as the MSA (number of sequences)
    weights = np.zeros(dat.shape[0])
    # define the number of sequences nS, number of positions nP 
    nS,nP = dat.shape
    # for each sequence count the number of sequences within the threshold
    for i in range(nS): 
        # define the sequence
        seq = dat[i,:]
        # repeat the sequence to get a matrix 
        mat = np.tile(seq, [nS-1,1])
        # delete the sequence from the data
        mat2 = np.delete(dat, i, axis = 0)
        # compare the two matrices to get fraction similarity
        fr_sim =  np.sum(mat == mat2, axis = 1)/nP
        # count the number within the threshold set
        weights[i] = len(np.where(fr_sim>thresh)[0])
    ## zeros should be 1 
    idx = np.where(weights==0)[0]
    weights[idx] = np.ones(len(idx))
    return 1/weights 
def calc_r_new(Data, Gen, mean_removed = True, nlist = np.arange(1,11), repititions = 100, m =20, verbose = True, weights = np.array([1])):
    """ Gets rm array (r20 is a calculation described in paper: F. McGee, S. Hauri, Q. Novinger, S. Vucetic, R. M. Levy, V. Carnevale, and A. Haldane, The Generative Capacity of Probabilistic Protein Sequence Models, Nat Commun 12, 6302 (2021).)
        Input:  Data : original data in  categorical (integer), or binary form  
                Gen : generated data in  categorical (integer), or binary form 
                mean removed : obtains the mean removed statistics if True (default: True)
                nlist: order of statistics list (default  np.arange(1,11))
                repititions : number of repititions for each n (default 100)
                verbose : prints the time it takes for each n calculation (default1)
                weights : weights matrix for the potts model generated sequences (default not the potts model ) 
                            of length = number of sequences in the OG MSA 
                            its elements = 1/count of similar sequences , where the similairty threshold is set for .8 by default
                            in the function that calculates the weights 
        output: r list 
        """
    ### Obtain numerical and binary (one hot encoded ) representations of the data 
    if np.max(Data)!=1: 
        # If data is numerical 
        BinData = Binarify(Data)
        NumData = Data
    else: 
        BinData =  Data
        NumData = Numerify(Data)

    if np.max(Gen)!=1: 
        BinGen = Binarify(Gen)
    else: 
        BinGen = Gen
    # nS > number of sequences in OG data , nPos > number of positions in that protein    
    nS, nPos = NumData.shape
    # defining the single cite frequencies to calculate delta sigma_i
    freq_data = np.mean(BinData, axis = 1)    
    freq_gen = np.mean(BinGen, axis = 1)
    freq_matrix_data = np.transpose(np.tile(freq_data, [BinData.shape[1],1,1]),[1,0,2])
    freq_matrix_gen = np.transpose(np.tile(freq_gen, [BinGen.shape[1],1,1]),[1,0,2])
    # Create the delta matrix  delta sigma 
    DeltaMatrix_data  = BinData - freq_matrix_data 
    DeltaMatrix_gen  = BinGen - freq_matrix_gen 
    # initialize r array 
    r_mean = [] 
    r_std_of_mean  = []
    # interate over order of stats 
    for n in nlist:
        # track time 
        t0 = time.time()
        # initialize pearson coefficient of correlation
        r = 0
        # repeat for repitition times  
        rarr = np.array([])
        # rep = 0
        rep_count=0
        while rep_count<repititions:  
            # randomly choose a position set of length n 
            pos_vec = random.sample(list(np.arange(nPos)), n)
            # get the categories / amino acids at those positions from all the data
            DataComb = NumData[:,pos_vec]
            # get the unique combinations 
            DataCombUnq = np.unique(DataComb, axis = 0)
            # initialize the correlation arrays 
            corr_arr_data = np.zeros(DataCombUnq.shape[0])
            corr_arr_gen = np.zeros(DataCombUnq.shape[0])
            # for each unique combination 
            for i in range(DataCombUnq.shape[0]):
                # define the combination/word
                aa_vec = DataCombUnq[i,:]
                aa_vec = aa_vec.astype(int)
                # if mean removed use the delta matrix 
                if mean_removed == True:
                    # if n =1 , do not use the delta matrix 
                    if n>1:
                        Mat_dat = DeltaMatrix_data
                        Mat_gen = DeltaMatrix_gen
                    else:
                        Mat_dat = BinData
                        Mat_gen = BinGen
                else: 
                    Mat_dat = BinData
                    Mat_gen = BinGen
                # get the mean removed frequency of that word combination 
                if len(weights)>1:
                    norm = np.sum(weights)
                else:
                    norm = nS
                corr_arr_data[i] = np.sum(weights*np.prod(Mat_dat[aa_vec, :, pos_vec], axis = 0))/norm
                corr_arr_gen[i] = np.mean(np.prod(Mat_gen[aa_vec, :, pos_vec], axis = 0))
            ## now sort this to get the most frequent k words
#             if verbose==1:
#                 if mean_removed==False:
#                     print('Sanity Check: sum of frequencies of unique combinations  ')
#                     print(np.sum(corr_arr_data))
            idx_mf = np.argsort(corr_arr_data)[-m:]
            # add to the pearson coefficient of correlation
            rn = np.corrcoef(corr_arr_data[idx_mf], corr_arr_gen[idx_mf])[0,1]
            # if rn is nan do not use in pearson correlation average 
            if np.isnan(rn)==False:
                rarr = np.concatenate((rarr, np.array([rn])), axis = 0)
                # r= r+ rn
                rep_count = rep_count +1 
            else: 
                if verbose ==True:
                    print('encountering a runtime error -- resulting in nan, ignoring that particular position combination')
        ## get the mean of the pearson correlations 
        r_mean.append(np.mean(rarr))
        # get the standard diviation of the mean 
        r_std_of_mean.append(np.std(rarr)/repititions)
        if verbose==True:
            print(f'{time.time()-t0} for finishing n = {n}')
    return r_mean , r_std_of_mean

# def calculate_r20_Modified(data, gen, repititions, nlist): 
# #     import time
#     # data must be numerified and gen as well 
#     n_msa_seq, nPos = data.shape
#     r20list = []
#     for n in nlist:
#         # print('working on n = ', n)
#         t0 = time.time()
# #         repititions = 100
#         r =0
#         for rep in range(repititions):
#             parr = random.sample(list(np.arange(nPos)), n)
#             DataComb = data[:,parr]
# #             print(DataComb.shape)
#             DataCombUnq = np.unique(DataComb, axis = 0)
# #             print(DataCombUnq.shape)
#             nuni = DataCombUnq.shape[0]

#             ## create a 3D tensor of both DataCombUniq and DataComb to compare their entries and get the highest freq words
#             DataCombRep = np.broadcast_to(DataComb, (nuni, n_msa_seq, n))
#             DataCombUniRep = np.broadcast_to(DataCombUnq, (n_msa_seq, nuni, n))
#             DataCombUniRep = np.transpose(DataCombUniRep, [1,0,2])
#             s = np.sum((DataCombRep==DataCombUniRep), axis =2 )
#             FreqArr = np.sum(s == np.ones(s.shape)*n, axis = 1)/n_msa_seq
#             idx_mf = np.argsort(FreqArr)[-20:]
#             FreqArr = FreqArr/np.sum(FreqArr)
#             Freq_data = FreqArr[idx_mf]
#             DataCombUnq = DataCombUnq[idx_mf,:]
#             DataCombUniRep = np.broadcast_to(DataCombUnq, (gen.shape[0], DataCombUnq.shape[0], n))
#             DataCombUniRep = np.transpose(DataCombUniRep, [1,0,2])
#             DataCombRep = np.broadcast_to(gen[:,parr], (DataCombUnq.shape[0], gen.shape[0], n))
#             s = np.sum((DataCombRep==DataCombUniRep), axis =2 )
#             FreqArr_gen = np.sum(s == np.ones(s.shape)*n, axis = 1)/gen.shape[0]
#             r+= np.corrcoef(FreqArr_gen, Freq_data)[0,1]




#         r20 = r/repititions
#         r20list.append(r20)
#         print(time.time() - t0, 'seconds to finish n = ', n)
#     return r20list


