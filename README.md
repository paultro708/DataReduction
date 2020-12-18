# InstanceSelection

InstanceSelection is a Python module for reducing number of instances in datasets used in classification problems.
The module is implemented as part of an engineering project.

# Instalation
```
    pip install InstanceReduction
```

# Usage
## Import modules
```
    from InstanceReduction.Raport import Raport
    from InstanceReduction.DataPreparation import DataPreparation
    from InstanceReduction.Reduction.DROP1 import DROP1

```
## Data loading and preparation
The first step is to load and prepare data using DataPreparation:
```
    data = DataPreparation('iris')
```
## Instance selection with selected algoritm
For all algorithms required parameter is instance of DataPreparation. Then you can reduce instances and prepare raport.
```
    alg = DROP1(data, k=3)
    alg.reduce_instances()
```
## Creating raport
After reduction with selected algorithm you can create raport:
```
    rap = Raport(data, alg.red_data, alg.red_lab)
    rap.print_raport(c_type = 'knn')
```

# Results of raporting
```
=============
Classifier:   knn
=============
Raport for original dataset
Count of instances:  105
                 precision    recall  f1-score   support

    Iris-setosa     1.0000    1.0000    1.0000        19
Iris-versicolor     1.0000    1.0000    1.0000        13
 Iris-virginica     1.0000    1.0000    1.0000        13

       accuracy                         1.0000        45
      macro avg     1.0000    1.0000    1.0000        45
   weighted avg     1.0000    1.0000    1.0000        45

Cohen's Kappa: 1.00
===
Training time:  0.0008822999999997805
Predicting time:  0.003322799999999848

Raport for reduced dataset
Count of instances:  21
                 precision    recall  f1-score   support

    Iris-setosa     1.0000    1.0000    1.0000        19
Iris-versicolor     0.7647    1.0000    0.8667        13
 Iris-virginica     1.0000    0.6923    0.8182        13

       accuracy                         0.9111        45
      macro avg     0.9216    0.8974    0.8949        45
   weighted avg     0.9320    0.9111    0.9090        45

Cohen's Kappa: 0.86
===
Training time:  0.0006775000000001086
Predicting time:  0.0024793999999999095

Reduction factor: 80.00 %
```