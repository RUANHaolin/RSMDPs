# Robust Satisficing MDPs


This repository contains supporting material for the paper [Robust Satisficing MDPs](https://doi.org/????) by H. Ruan, S. Zhou, Z. Chen and C. P. Ho.


## Content

This repository includes

1. Instance data for experiments "Improvements on Percentiles" and "Target-Oriented Feature".
1. Source codes for all our experiments.


### Data files

For both the experiments "Improvements on Percentiles" and "Target-Oriented Feature", the instance data files are located in the folder [exp_models](exp_models). For the former experiment, the random instances generated under the environments *river swim*, *machine replacement* and *grid world* are stored in the folders [exp1_samples_RiSw](exp_models/exp1_samples_RiSw), [exp1_samples_MR](exp_models/exp1_samples_MR) and [exp1_samples_GW](exp_models/exp1_samples_GW), respectively. For the latter one, the instances under the three environments are stored in [exp2_samples_RiSw](exp_models/exp2_samples_RiSw), [exp2_samples_MR](exp_models/exp2_samples_MR) and [exp2_samples_GW](exp_models/exp2_samples_GW), respectively.
	


### Source codes
All our experiments are implemented in C++. The folder [exp_models](exp_models) contains all the source code files for the experiments "Improvements on Percentiles" and "Target-Oriented Feature", and the folder [exp_algorithms](exp_algorithms) contains those for the experiment "Scalability of Different Algorithms".

## Data Formats

### Improvements on Percentiles

The folders [exp1_samples_RiSw](exp_models/exp1_samples_RiSw) [exp1_samples_MR](exp_models/exp1_samples_MR) and [exp1_samples_GW](exp_models/exp1_samples_GW) contain the .csv files that record the estimated transition kernels associated with different lengths of trajectories, where the lengths are as indicated by the file names (e.g., "SamSiz11MaxSte10.csv" corresponds to the transition kernel estimated based on a trajectory with length $11\times 10=110$, and "SamSiz3MaxSte100.csv" corresponds to the one with length $3 \times 100=300$).

### Target-Oriented Feature

For the *river swim* environment, the folder [exp2_samples_RiSw](exp_models/exp2_samples_RiSw) contains two folders---[train](exp_models/exp2_samples_RiSw/train) and [test](exp_models/exp2_samples_RiSw/test). The .csv file that records the true transition kernel is in the former folder, while the files for the polluted kernels are in the latter folder. The files for the *machine replacement* and the *grid world* environments are in the folders [exp2_samples_MR](exp_models/exp2_samples_MR) and [exp2_samples_GW](exp_models/exp2_samples_GW), respectively.






