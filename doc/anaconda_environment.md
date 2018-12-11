## Environment
In this project, we used currently Ubuntu 14.04(Linux), Anaconda3 (with python3.5, below anaconda 4.2.0), CUDA 8.0 (with Cudnn 7.0.5).

### 1. Setting Anaconda environment
After installing [Anaconda](https://www.continuum.io/downloads), it is nice to create a [conda environment](http://conda.pydata.org/docs/using/envs.html)
so you do not destroy your main installation in case you make a mistake somewhere:
```bash
conda create --name mclkd python=3.5
```
Now you can switch to the new environment in your terminal by running the following (on Linux terminal):
```bash
source activate mclkd 
```

### 2. Required Packages

#### Pytorch
```bash
conda install -y pytorch=0.3.1 torchvision=0.2.0 cuda80 -c soumith
```

#### TensorFlow for tensorboard (python 3.5)
We install CPU-version of TensorFlow since we use only tensorboard.
```bash
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.7.0-cp35-cp35m-linux_x86_64.whl
````

#### Python packages through conda
This project requires several Python packages to be installed.<br />
You can install the packages by typing:
```bash
conda install -y nb_conda numpy scipy jupyter matplotlib pillow nltk tqdm pyyaml seaborn scikit-image scikit-learn h5py
conda install -y -c conda-forge colorlog coloredlogs
```

#### Python packages through pip
Curerntly not use pip-based packages
