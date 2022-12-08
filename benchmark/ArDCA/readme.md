For comparison with GENERALIST, we run the ArDCA [code](https://github.com/pagnani/ArDCA) accompanying the paper 

[Trinquier J, Uguzzoni G, Pagnani A, Zamponi F, Weigt M. Efficient generative modeling of protein sequences using simple autoregressive models. Nat Commun. 2021;12: 5800. doi:10.1038/s41467-021-25756-4](https://www.nature.com/articles/s41467-021-25756-4)

### What is in this folder?

- ardca_julia_run.ipynb 

    We train ArDCA model on all the MSAs present in ../Data/ and use the ardca model generated sequences and run various tests to compare to GENERALIST. 

    After training the model, we extract the following parameters: single site fields ``H``, two site couplings ``J``, the vector corresponding to the how the positions are permuted ``idxperm``, and the probability of amino acids at the initial site in the
    sequence given the chosen ordering ``p0``, and save them to use them in the probability calculation. 

- prob_ardca_fns.py

    Functions to get the probability of a sequence given the parameters of the ArDCA model   ``H``, ``J`` and the arrays ``idxperm`` and ``p0`` .

- minima_ardca_sc.py

    The script that runs the local minima search, to find the locally optimum sequence starting from a natural sequence. 
