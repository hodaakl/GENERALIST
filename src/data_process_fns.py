# from PerformanceFunctions import  Numerify
import numpy as np
def Numerify(binary_3d_data ):
    """takes in 3D Binary data of the form nA X nS X nP and gives integer data nS X nP for which the values are of range(nA)"""
    # Check that input is binary 
    if np.max(binary_3d_data)==1: 
        idd = np.where(binary_3d_data==1)
        output = np.zeros((binary_3d_data.shape[1], binary_3d_data.shape[2]))
        output[idd[1], idd[2]] = idd[0]
    else: 
        print('Input already categorical (numerical) not binary, output = input')
        output = binary_3d_data
    
    return output


def Binarify(matrix): 
    """Takes categorical integer data of shape nS, nPos and spits out the nA, nS, nPos binary matrix of protein sequences 0,1
    nA = number of categories 
    """
    # Check that the entries are numerical and not categorical. 
    if np.max(matrix)==1: 
        print('Matrix Already Binary: output is the input')
        output = matrix
    else:
        # nA = int(np.max(matrix) +1 )#number of categories 
        nA = 21
        nS, nPos = matrix.shape
        output = np.zeros((nA, nS, nPos))
        for i in range(nA): 
            pos_idx = np.where(matrix == i* np.ones((nS,nPos)))
            output[ i, pos_idx[0], pos_idx[1]] = np.ones(len(pos_idx[0]))
    return output
def Convert_fastaToNp(filepath, binary = True, labels_inc =True): 
    """  Takes a file path and convert the fasta file of protein sequences to a numpy array 
    Input:  filepath  (path to the fasta file)
            Seq_len (expected sequence length : sometimes the fasta file is written with a space at the end which causes issues, this input takes care of it)
            binary (outputs a one hot encoded matrix , set to False to get the numerical encoding of amino acids in a 2D shape)
            labels_inc (are the labels included in the fasta file? every sequence will be proceeded by >some_label, otherwise set to False.)
    Output: 3D matrix of length, one hot encoded (21, nSequences, nPositions) if binary = True. 
            2D matric of numbers ranging between [0,20] (nSequences, nPositions) if binary =False. 
    """
    AA_Letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-' ]
    nAA = len(AA_Letters) 
    # load the fasta file 
    with open(filepath) as f:
        lines = f.readlines()

    nlen = len(lines)
    if labels_inc==True:
        
        idx = np.arange(1,nlen,2)
    else: 
        idx = np.arange(1,nlen)

    
    if lines[idx[0]][-1] =='\n':
        Seq_len = len(lines[idx[0]]) -1
    else:
        Seq_len = len(lines[idx[0]])
    Data_base = np.empty((len(idx), Seq_len), dtype = str)
    k=0
    for i in idx: 
        Seq = lines[i][:Seq_len]
        Data_base[k] = np.asarray(list(Seq))
        k+=1
    sigmas = np.zeros((nAA,Data_base.shape[0], Data_base.shape[1]))
    for a in range(nAA): 
        AA = AA_Letters[a]
        idx = np.where(Data_base == AA)
        sigmas[a,idx[0] ,idx[1]] = np.ones(idx[0].shape)

    sanch = len(np.where(np.sum(sigmas, axis = 0) != 1)[0])
    if sanch!= 0: 
        raise ValueError('sigmas do not sum upto 1 on axis = 0, something is wrong!')    
    if binary == True: 
        out = sigmas 
    else: 
        out = Numerify(sigmas)
    return out 

def convert(s):
    """ Converts a list of letters (amino aicds) into one string 
    Input: List of letters, each element in a string 
    Output: one string of all letters"""
   # initialization of string to ""
    new = ""
    # traverse in the string 
    for x in s:
        new += x 
    # return string 
    return new

def Conv_save_NpToFasta(Numerical_Data, path='',  label = 'seq', verbose = False , save = True, include_fish_sym = True): 
    """"Saves data of numerical sequences to a fasta file
    Input: Numerical_Data :shape = [number of sequences , number of positions ]] 
            path : pathtofile.fa
    Output: saved fasta file """

    AA_Letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-' ]
    AA_arr = np.asarray(AA_Letters)
    if save==True:
        file1 = open(path ,"a")
    ## If data is binary (3D) Numerify it . 
    if len(Numerical_Data.shape)>2:
        Numerical_Data = Numerify(Numerical_Data)
    
    if len(Numerical_Data.shape)==1:
        Numerical_Data = np.reshape(Numerical_Data, (1,Numerical_Data.shape[0]))
    dim = len(Numerical_Data.shape)
    if dim>1:
        # saving a fasta file with multiple sequences 
        for i in range(Numerical_Data.shape[0]): 
            if include_fish_sym==True:
                fl = f'>{label}\n'
            else:
                fl = f'{label}\n'
            int_arr = Numerical_Data[i,:].astype(int) 
            seq = AA_arr[int_arr]
            sl = list(seq)
            sl_one_str = convert(sl)
            sl_one_str = f'{sl_one_str}\n'
            if save==True:
                file1.write(fl)
                file1.write(sl_one_str)
    else: 
        fl = f'>{label}{i}\n'
        int_arr = Numerical_Data.astype(int) 
        seq = AA_arr[int_arr]
        sl = list(seq)
        sl_one_str = convert(sl)
        sl_one_str = f'{sl_one_str}'
        if save==True:
            file1.write(fl)
            file1.write(sl_one_str)
        
    
    if save ==True:
        file1.close()
    if verbose == True:
        print(f'done with save = {save}')
    
    return sl_one_str

# import numpy as np
def write_file(filename, details): 
    """Write details of the run:: 
        Add alpha, reps, steps, and any relevant details about the run.
        filename: Output folder """
    f= open(filename,"w+")
    f.write(details)
    f.close()
    return print('wrote details file')