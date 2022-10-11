# Quantum Process Tomography via Genetic Algorithms and Neural Networks amo anna

[![!](https://img.shields.io/badge/Genetic-Algorithms-orange)]() [![!](https://img.shields.io/badge/Neural-Networks-blue)]()


This repository contains the numerical codes to reproduce the results reported in the paper: 'Retrieving complex polarization transformations via optimized quantum process tomography'
submitted to ...

With the Jupyter Notebooks available in the repository, it is possible to reconstruct $SU(2)$ processes both via Machine Learning (ML) and Genetic Algorithms (GAs).

In detail the repo is organized as follows:

  ###  GAs Code
  ____
  1. Genetic_Reconstruction_of_Single_Processes.ipynb is the notebook containing the tools required to reproduce the experiment on single $SU(2)$ Trasformations reconstructed via GAs.
  2. GA_Maps_Reconstruction.ipynb contains the code for reconstructing the 73X73 images reported in the above paper.

  ###  ML Code
  ____
  1. The folder 'models' contains all the trained Neural Networks for different numbers of input measurements (3, 4, 5, 6)
  2. NN_Reconstruction_of_Single_Processes.ipynb is the notebook containing all the tools and instructions necessary to perform the tomographic reconstruction of single $SU(2)$ processes via Neural Networks.
  3. NN_Maps_Reconstruction.ipynb, just as for GAs, is the notebook containing the code to reproduce the final images of the paper.
  ###  Data
  ____
  All the data used in the paper are reported in the repo.
  
  1. The folder 'dataset_gassian' contains the data for reporducing single $SU(2)$ Transformation
  2. The folder 'experimental_data_73' contains the data for reconstructing the three images 73x73 shown in the paper

For adapting the code to your particular problem you should simply adapt your data in the same format of these you can find in this repo.

All the Notebook inserted in this repo can be executed directily on Binder by means of the following Tag: 
