import random
import numpy as np

'''
A library to calculate the negative of the hamiltonian of the sequences outputted by the method referred to as adabmDCA:
Muntoni, A.P., Pagnani, A., Weigt, M. et al. adabmDCA: adaptive Boltzmann machine learning for biological sequences. BMC Bioinformatics 22, 528 (2021). https://doi.org/10.1186/s12859-021-04441-9
'''

'''
Reads standard .dat output file from the adabmDCA method and saves J and h into seperate numpy arrays

input: 
 - fileName: name of data file
 - protein: string, name of protein for labeling purposes
 
'''
def calc_J_and_h(fileName, protein):
    file = open(fileName, 'r') # open(f'Potts/{protein}/Parameters_conv_{protein}_nw_20000.dat', 'r')
    count_J = 0
    count_h = 0
    oops_count = 0
    x = 0
    y = 0

    for line in file.readlines():
        if(line[0] == 'J'):
            x+=1
        if(line[0] == 'h'):
            y+=1

    file.close()
    file = open(fileName, 'r') # open(f'Potts/{protein}/Parameters_conv_{protein}_nw_20000.dat', 'r')

    ij_arr = np.zeros((x, 5))
    i_arr = np.zeros((y,3))

    for line in file.readlines():
        line_arr = line.split(' ')
        
        if line_arr[0] == 'h':
            line_arr[3] = line_arr[3].replace('\n', '')
            i_arr[count_h] = line_arr[1:]
            count_h += 1
        elif line_arr[0] == 'J':
            line_arr[5] = line_arr[5].replace('\n', '')
            if(count_J >= x - 1):
                print(line_arr)
            ij_arr[count_J] = line_arr[1:]
            count_J += 1
        else:
            print('------------ Error! -------------')
            break

        if(count_J%5000 == 0):
            print(count_J)

    np.save(f'Potts/{protein}/J_arr_{protein}', ij_arr)
    np.save(f'Potts/{protein}/h_arr_{protein}', i_arr)

'''
Calculates negative of the Hamiltonian of arrays sorted as the function above outputs

inputs:
 - J_arr: the array of the couplings for the protein
 - h_arr: the array of the fields for the protein
 - seq: the sequence to calculate the probability of
 
outputs:
 - nHamiltonian: the negative of the hamiltonian
'''
def calc_n_hamiltonian(J_arr, h_arr, seq):
    H_h = 0
    H_j = 0
    x = 0

    for j in range(0, len(seq)):
        if(j>0):
            x += 50-(j-1)
        H_h += h_arr[j, seq[j]]

        for k in range(j + 1, len(seq)):
            n  = (x + (k-j-1))*(21**2)
            a = 21*seq[j]
            b = seq[k]
            # if(n + a + b == 562275):
            #     print('j =', j, 'i =', i)
            H_j += J_arr[n + a + b , 4]

    nHamiltonian = H_h + H_j

    return nHamiltonian

'''
An alternative method for calculating the hamiltonian. This does not depend on the order of the array. It is more general but it also takes significantly more time and for adabmDCA produces the same results

inputs:
 - J_arr: the array of the couplings for the protein
 - h_arr: the array of the fields for the protein
 - seq: the sequence to calculate the probability of
 
outputs:
 - nHamiltonian: the negative of the hamiltonian
'''
def calc_n_hamiltonian_2(J_arr, h_arr, seq):
    H_h = 0
    H_j = 0
    J_pos = J_arr[:,:-1]
    x = 0
    for i in range(len(seq)):
        H_h += h_arr[i, seq[i]]
        
        for j in range(i + 1, len(seq)):
            
            pos = np.array([i, j, seq[i], seq[j]])
            indx = np.where(np.all(J_pos==pos,axis=1))         
            if(len(indx) == 0 or len(indx) > 1):
                print('----------------------- Error -------------------')
                return -5000000000
            
            idx = int(indx[0])
            H_j += J_arr[idx, 4]
            
    nHamiltonian = H_h + H_j
    
    return nHamiltonian

'''
find_min_seq(J_arr, h_arr, seq, stop_num): Function to find sequence with a local minima in energy from a given starting sequence

inputs:
 - J_arr: the array of the couplings for the protein
 - h_arr: the array of the fields for the protein
 - seq_i: sequence we would like to start from
 - stop_num: stop_num * length of sequence is how long the algorithm will search for a mutation with a higher probability

returns:
 - seq: the sequence that we converge on
 - prob: the probability of the sequence we converge on
 - seq_i: the original sequence
 - prob_i: the probability of the original sequence
'''
def find_min_seq(J_arr, h_arr, seq_i, stop_num):
    n = 20
    L = len(seq_i)

    count = 0 # count of how many steps we have gone without finding a better sequence
    count_tot = 0

    seq = list(seq_i); seq_changed = list(seq_i) # Initialize the sequence_i and sequence changed

    prob_i = calc_prob(J_arr, h_arr, seq); prob = prob_i # Get the probability of the initial sequence, set equal to current prob

    while(count < stop_num * L):
        pos = random.randint(0, L - 1) # pick random position to mutate
        aa = random.randint(0, n - 1) # random ammino acid to mutate to

        seq_changed[pos] = aa # change sequence

        prob_f = calc_prob(J_arr, h_arr, seq_changed) # calc prob of the mutated sequence

        # If sequence is better accept it
        if prob_f > prob:
            seq = list(seq_changed)
            prob = prob_f
            count = 0
        # If not reject it
        elif prob >= prob_f:
            seq_changed = list(seq)
            count += 1
        else:
            print('Error')
            break
        count_tot += 1
        
        # prob_arr = np.append(prob_arr, prob_i) # prob of current sequence
        
        # if count % 10000 == 0 and count != 0:
        #     print('count =', count)

    return seq, prob, seq_i, prob_i, count_tot


'''
find_min_from_rand_seq(J_arr, h_arr, seq_arr, stop_num): Function to find sequence with a local minima in energy from a randomly chosen starting sequence

inputs:
 - J_arr: the array of the couplings for the protein
 - h_arr: the array of the fields for the protein
 - seq_arr: array of the sequences in the MSA
 - stop_num: stop_num * length of sequence is how long the algorithm will search for a mutation with a higher probability

returns:
 - seq: the sequence that we converge on
 - seq_arr[n]: the original sequence
 - n: index of original sequence
 - prob_arr: array of the probability of the current sequence at every step
'''
def find_min_from_rand_seq(J_arr, h_arr, seq_arr, stop_num):

    AA_Letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    ''' Note that I use x = list(y) when setting the arrays equal to one another. I learned the hard way that not doing it like this
    actually sets them to the same address in memory which is no bueno'''

    count = 0 # count of how many steps we have gone without finding a better sequence
    
    n = random.randint(0, seq_arr.shape[0] - 1) # Pick random sequece
    seq = list(seq_arr[n]); seq_changed = list(seq_arr[n]) # Initialize the sequence_i and sequence changed

    prob_i = calc_prob_2(J_arr, h_arr, seq) # Get the probability of the initial sequence

    while(count < stop_num * seq_arr.shape[1]):
        pos = random.randint(0, seq_arr.shape[1] - 1) # pick random position to mutate
        aa = random.randint(0, len(AA_Letters) - 1) # random ammino acid to mutate to

        seq_changed[pos] = aa # change sequence
        

        prob_f = calc_prob_2(J_arr, h_arr, seq_changed) # calc prob of the mutated sequence

        # If sequence is better accept it
        if prob_f > prob_i:
            seq = list(seq_changed)
            prob_i = prob_f
            count = 0
        # If not reject it
        elif prob_i >= prob_f:
            seq_changed = list(seq)
            count += 1
        else:
            print('Error')
            break
        
        # prob_arr = np.append(prob_arr, prob_i) # prob of current sequence
        
        # if count % 10000 == 0 and count != 0:
        #     print('count =', count)

    return seq, seq_arr[n], n, np.array(prob_f)


def hamming_distance(Data, min_seq):
    count = 0
    ham_dis = np.zeros(Data.shape[0])

    for seq in Data:
        for i in range(len(seq)):
            if(min_seq[i] != seq[i]):
                # print(i, 'is different in', count)
                ham_dis[count] += 1
        count += 1
        # print(ham_dis)
        # if(count%5000 == 0):
        #     print('sample count', str(count))

        # print(ham_dis.size)

    min_ham_dis = np.min(ham_dis)
    return min_ham_dis