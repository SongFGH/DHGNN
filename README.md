# DHGNN: Dynamic Hypergraph Neural Networks
## Introduction
DHGNN is a kind of neural networks modeling dynamically evolving hypergraph structures.
## Citation
## Installation
The code has been tested with Python 3.6, CUDA 9.0 on Ubuntu 16.04. To run the code, you may also need some of the following Python packages, which can be installed via pip or conda:
- absl-py (0.7.0)
- astor (0.7.1)
- backports.weakref (1.0rc1)
- bleach (1.5.0)
- certifi (2016.2.28)
- cycler (0.10.0)
- decorator (4.1.2)
- gast (0.2.2)
- grpcio (1.18.0)
- h5py (2.7.0)
- html5lib (0.9999999)
- Keras-Applications (1.0.6)
- Keras-Preprocessing (1.0.5)
- kiwisolver (1.0.1)
- Markdown (2.6.9)
- matplotlib (3.0.2)
- networkx (1.11)
- numpy (1.15.4)
- pandas (0.20.3)
- Pillow (5.3.0)
- pip (9.0.1)
- protobuf (3.6.1)
- pyparsing (2.3.1)
- python-dateutil (2.6.1)
- pytz (2017.2)
- PyYAML (3.13)
- scikit-learn (0.19.0)
- scipy (1.1.0)
- setuptools (36.4.0)
- six (1.11.0)
- tensorboard (1.12.2)
- tensorflow (1.3.0)
- tensorflow-gpu (1.12.0)
- tensorflow-tensorboard (0.1.5)
- termcolor (1.1.0)
- thop (0.0.31.post1909230639)
- torch (0.4.1)
- torchvision (0.2.1)
- Werkzeug (0.12.2)
- wheel (0.29.0)
## Experiment
## Usage
## License
Our code is released under MIT License (see LICENSE file for details).
### Cora
- **Setting**: standard setting, 140 train, 500 valid, 1000 test   
- **state of the art**: LGCN 83.3%   
- **DHGNN_knn**: 83.8% (max)  
    - **config_cora.yaml**
- **DHGNN_cluster**: 83.8% (max)
    - **config_cora_cluster.yaml**
### ModelNet40
- **Setting**: same as HGNN  
- **HGNN**: 96.7%  
- **DHGNN_knn**: 97.6%  
    - **config_modelnet40.yaml**
### Weibo
- **Setting**: 4690 train, 400 valid, 500 test (10 times random)  
- **Bi-MHG**: 90.0%  
- **DHGNN_knn**: 91.8%  (mean)
    - **config_weibo.yaml**
- **DHGNN_cluster**: 91.0% (mean)
    - **config_weibo_cluster.yaml**
