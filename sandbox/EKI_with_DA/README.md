# Example of Lorenz 63 for joint learning of parameters and the initial condition 

## Source codes
1. Ensemble Kalman inversion
2. Numerical simulator of Lorenz 63 system

## Scripts 
1. main.py: the main script for the joint learning of L63 with EKI
1. main_ultradian.py: the main script for the joint learning of ultradian model with EKI
2. plot_params.py: plotting the results of learned parameters and the initial conditions
3. plt_G.py: plotting the results of observation data

## Conda environment
1. Install conda (replace link with most recent)
   1. wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   2. bash Anaconda3-2023.09-0-Linux-x86_64.sh
   3. Refresh terminal
   4. conda update conda
   5. rm Anaconda3-2023.09-0-Linux-x86_64.sh
2. Set up environment
   1. `conda create --name ekida python=3.10`
   2. `conda activate ekida`
   3. `conda install pytorch torchvision -c pytorch`
   4. `conda install -c anaconda tqdm scipy pandas seaborn matplotlib`