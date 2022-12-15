# Source code for GENERALIST 
### Required python packages 
- torch # to run inference , parameters are saved as torch tensors
- numpy # to load and manipulate data , parameters are saved as numpy arrays
- os    # to create the output folder if it doesn't exist
- time  # to calculate how log the inference takes

### Description of the files 
- main.py

    Script that runs generalist algorithm given a fasta file , the latent dimension chosen by the user and the output directory. This script creates an output folder if it doesn't already exist and saves the inferred distribution parameter arrays (Z, Theta) in it. 
    
    Then it uses those parameters to geenrate new sequences.

- data_process_fns.py

    Includes functions that transfer the fasta file into a numerical matrix or a one-hot encoded tensor to run the inference on, and includes supporting functions like that used to write to a txt file. 

- inference_fns.py

    Includes the gradient functions, the calculation of the likelihood, and the update of paramters using adam algorithm.

- compile_generalist_fn.py

    Includes the full function that runs the inference. Loaded in main.py.

- performance_fns.py

    Includes the funcitons used to assess the generated sequences by assessing the statistics and the hamming distance between training and generated dataset.

- loader_class.py

    Includes Generator Class , this uses the files saved in the output directory and generated sequences sampled from the learned distributions. 