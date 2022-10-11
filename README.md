# Quantum Process Tomography via Genetic Algorithms and Neural Networks

[![!](https://img.shields.io/badge/Genetic-Algorithms-orange)]() [![!](https://img.shields.io/badge/Neural-Networks-blue)]()


This repository contains the numerical codes to reproduce the results reported in the paper: 'Retrieving complex polarization transformations via optimized quantum process tomography'
submitted to ...

With the Jupyter Notebooks available in the repository, it is possible to reconstruct $SU(2)$ processes both via Neural Networks (NN) and Genetic Algorithms (GAs).

In detail the repo is organized as follows:

  ###  GAs Code
  ____
  1. Genetic_Reconstruction_of_Single_Processes.ipynb is the notebook containing the tools required to reproduce the experiment on single $SU(2)$ Trasformations reconstructed via GAs.
  2. GA_Maps_Reconstruction.ipynb contains the code for reconstructing the 73X73 images reported in the above paper.

  ###  ML Code
  ____
  1. The folder 'models' contains the trained Neural Network (.h5 and.json files) for 6 input measurements.
  2. NN_Reconstruction_of_Single_Processes.ipynb is the notebook containing all the tools and instructions necessary to perform the tomographic reconstruction of single $SU(2)$ processes via Neural Networks.
  3. NN_Maps_Reconstruction.ipynb, just as for GAs, is the notebook containing the code to reproduce the final images of the paper.
  ###  Data
  ____
  All the data used in the paper are reported in the repo.
  
  1. The folder 'dataset_gassian' contains the data for reporducing single $SU(2)$ Transformation
  2. The folder 'experimental_data_73' contains the data for reconstructing the three images 73x73 shown in the paper

To adapt the code to your particular problem you should simply convert your data to the same format used in this repo.

All the Notebook inserted in this repo can be executed directily on Binder by means of the following Tag: 
