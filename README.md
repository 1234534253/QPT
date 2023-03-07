# Quantum Process Tomography via Genetic Algorithms and Neural Networks

[![!](https://img.shields.io/badge/Genetic-Algorithms-orange)]() [![!](https://img.shields.io/badge/Neural-Networks-blue)]()


This repository contains the numerical codes to reproduce the results reported in the paper: 'Retrieving space-dependent polarization transformations via near-optimal quantum process tomography'. 


With the Jupyter Notebooks available in the repository, it is possible to reconstruct $SU(2)$ processes via a Neural Network (NN) and a Genetic Algorithm (GA), as detailed in the paper. 

Notebooks can be executed online via [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/1234534253/QPT.git/HEAD)

The repository is organized as follows:

  ###  GA Code
  ____
  1. The folder 'utils' contains the functions ('GA_utils.py') used within the GA.
  2. Genetic_Reconstruction_of_Single_Processes.ipynb is the notebook containing the tools required to reconstruct the synthetic experiments on single $SU(2)$ Trasformations via GAs.
  2. GA_Maps_Reconstruction.ipynb is the notebook to reconstruct the experimental 73X73 images reported in the paper via GAs.

  ###  NN Code
  ____
  1. The folder 'models' contains the trained NN (.h5 and.json files) for 6 input measurements.
  2. NN_Reconstruction_of_Single_Processes.ipynb is the notebook containing all the tools to reconstruct the synthetic experiments on single $SU(2)$ Trasformations via NNs.
  3. NN_Maps_Reconstruction.ipynb is the notebook to reconstruct the experimental 73X73 images reported in the paper via NNs.
  
  ###  Data
  ____
  All the data used in the paper are reported in the repo.
  
  1. The folder 'dataset' contains synthetic outcomes of polarimetric measurements associated with 1000 $SU(2)$ Transformation. The folder contains the theoretical parameters of each transformation as well. The subfolders '0', '1', '2', '5' contain the same set of 6 polarimetric measurements, simulating different levels of experimental noise. 
  2. The folder 'experimental_data' contains the intensity data of the three experimental 73x73 images reported in the paper

To adapt our codes to your particular experimental conditions, simply arrange your dataset into the format used in this repo. 
