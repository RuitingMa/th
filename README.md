# Ensemble-based Low Precision Inference in Software 


### Description 
Implemented based on Luca Mocerino's [Binary-Neural-Networks-PyTorch-1.0](https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-1.0), this repository implements the proposed design in my Honours project.



The project is organized as follows:

  - **models** folder contains CNN models (simple mlp, Network-in-Network, LeNet5, etc.)
  - **classifiers/{type}_classifier.py** contains the test and train procedures; where type = {bnn, xnor, dorefa}
  - **models/{type}_layers.py** contains the binarylayers implementation (binary activation, binary conv and fully-connected layers, gradient update);  where type = {bnn, xnor, dorefa}
  - **yml** folder contains configuration files with hyperparameters
  - **main.py** represents the entry file
  - **ensemble.py** represents the implementation of the proposed ensemble, containing its test and training procedures
  - **BENN.py** represents the ensemble voting algorithms described in the thesis treaty which were used to unify the individual the votes of the members.

Please note that so far, only XNOR-LeNet and XNOR-NiN on MNIST and CIFAR-10 have been modified for integration in the ensmeble. Other model architectures have been left in the repository for integration in the future.
### Installation

All packages are in *requirement.txt*
Install the dependencies:

```sh
pip install -r requirements.txt
```
### Basic usage
```sh
$ python main.py app:{yml_file}
```
### Example 
Network-in-Network on CIFAR10 dataset. All hyper parameters are in .yml file. 
```sh
$ python main.py app:yml/nin_cifar10.yml
```




