# DHGNN: Dynamic Hypergraph Neural Networks
## Introduction
DHGNN is a kind of neural networks modeling dynamically evolving hypergraph structures.
## Citation
## Installation
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
