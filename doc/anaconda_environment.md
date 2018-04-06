## Environment
In this project, we used currently Ubuntu 14.04(Linux), Anaconda3 (with python3.5, below anaconda 4.2.0), CUDA 8.0 (with Cudnn 7.0.5).

### 1. Setting Anaconda environment
After installing [Anaconda](https://www.continuum.io/downloads), it is nice to create a [conda environment](http://conda.pydata.org/docs/using/envs.html)
so you do not destroy your main installation in case you make a mistake somewhere:
```bash
conda create --name vqa-kdmcl python=3.5
```
Now you can switch to the new environment in your terminal by running the following (on Linux terminal):
```bash
source activate vqa-kdmcl
```

### 2. Required Packages

#### install python packages through conda
This project requires several Python packages to be installed.<br />
You can install the packages by typing:
```bash
conda install nb_conda numpy scipy jupyter matplotlib pillow nltk tqdm pyyaml seaborn scikit-image scikit-learn h5py
conda install -c conda-forge colorlog 
conda install -c conda-forge coloredlogs
```

#### install python packages through pip
Curerntly not use pip-based packages
