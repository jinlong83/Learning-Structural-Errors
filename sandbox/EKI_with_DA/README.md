# Example of Lorenz 63 for joint learning of parameters and the initial condition 

## Source codes
1. Ensemble Kalman inversion
2. Numerical simulator of Lorenz 63 system

## Scripts 
1. main_nn.py: the main script for learning Ultradian via DA + EKI
2. main_nn_sparse.py: the main script for learning Ultradian via DA + EKI w/ L1 penalty

## Conda environment
1. Install conda (replace link with most recent)
   1. wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   2. bash Anaconda3-2023.09-0-Linux-x86_64.sh
   3. Refresh terminal
   4. conda update conda
   5. rm Anaconda3-2023.09-0-Linux-x86_64.sh
2. Set up environment
   1. `conda env create -f ekida_simple.yml`