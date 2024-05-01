# Addressing Machine Learning Problems in the
Non-Negative Orthant  🚀


This repository contains an implementation of the manuscript "*[Addressing Machine Learning Problems in the
Non-Negative Orthant](https://ieeexplore.ieee.org/document/10510233)*" by Ioannis Tsingalis and Constantine Kotropoulos.

## Usage
### 1. Requirements
The requirements are in the requirements.yml file. 


### 2. Download Dataset
You can download the datasets from [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and [here](https://github.com/zalandoresearch/fashion-mnist) and extract them to the folder dataset (see directory structure below).


### 3. Train and Test

```angular2
python run_softMaxOur.py --task_name fashion
```

```angular2
python run_softMaxGenop.py --task_name fashion
```

## Reference
If you use this code in your experiments please cite this work by using the following bibtex entry:

```
@ARTICLE{10510233,
  author={Tsingalis, Ioannis and Kotropoulos, Constantine},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Addressing Machine Learning Problems in the Non-Negative Orthant}, 
  year={2024},
  pages={1-15},
  doi={10.1109/TETCI.2024.3379239}
  }
```