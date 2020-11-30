# InstanceSelection

InstanceSelection is a Python module for reducing number of instances in datasets used in classification problems.
The module is implemented as part of an engineering project.

# Instalation
TODO

# Usage
## Data loading and preparation
The first step is to load and prepare data using DataPreparation:
```
    data = DataPreparation('iris')
```
## Instance selection with selected algoritm
For all algorithms required parameter is instance of DataPreparation. Then you can reduce instances and prepare raport.
```
    alg = ENN(data, k=5)
    alg.reduce_instances()
```
## Creating raport
After reduction with selected algorithm you can create raport:
```
    rap = Raport(data, alg.red_data, alg.red_lab)
    rap.print_raport(c_type = 'knn')
```

# Results of raporing
TODO

