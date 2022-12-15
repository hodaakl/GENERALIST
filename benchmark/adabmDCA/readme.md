For comparison with GENERALIST, we run the adabmDCA [code](https://github.com/anna-pa-m/adabmDCA) accompanying the paper 

[Muntoni, A.P., Pagnani, A., Weigt, M. et al. adabmDCA: adaptive Boltzmann machine learning for biological sequences. BMC Bioinformatics 22, 528 (2021). https://doi.org/10.1186/s12859-021-04441-9](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04441-9#citeas)

Ran with the options
```
-I -s 1000 -n 20 -m 100 -z 1 -u 0.005 -v 0.005 -P -L -t 20 -e 40 -i 100000 -l 0.0 -S
```

### What's in this folder?

- HamiltonianFunctions.py
    We train adabmDCA with the BPT1 MSA present in ../Data/ and use the adabmDCA model generated sequences and run various tests to compare to GENERALIST.

    After training the model, we extract the following parameters: single site fields ``h``, two site couplings ``J``. These are stored in arrays that put ``J`` and ``h`` in the order 

    ---
    > $h = [h_1, h_2, ..., h_L]$  and $J = [J_{12}, J_{13}, ..., J_{1L}, J_{23}, J_{24} ... J_{(L-1)L}]$

    The asssociated hamiltonian of a sequence can be calculated as: 

    $ -\sum_{i} h_i(s_i) - \sum_{i < j} J_{i_j}(s_i, s_j) $
    
    where $s_i$ is the amino acid present at the $i^{th}$ position in the sequence. 

    The functions in HamiltonianFunctions.py:
    - Read the output file from adabmDCA into arrays ordered as shown above
    - Calculate the negative of the hamiltonian according to the above equation, using the given ordering of fields and couplings stored in the array, or a search for the exact entry. Both methods produce the same results
    - Run the local minima search, to find the locally optimum sequence starting from a natural sequence. 
    - Run the local minima search, to find the locally optimum sequence starting from a random sequence. 
