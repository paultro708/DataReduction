from InstanceReduction.DataPreparation import DataPreparation
from InstanceReduction.Reduction.DROP1 import DROP1
from InstanceReduction.Reduction.ENN import ENN
from InstanceReduction.Reduction.MSS import MSS
from InstanceReduction.Reduction.PSC import PSC
from InstanceReduction.Reduction.ICF import ICF
import pytest
"""
File with constant parameters used in tests
"""

#parametres for DataPreparation

#all values arrays above depends on "names" datasets
names = ['iris', 'pendigits']#, 'letter']
len_all = [150, 10992]#, 20000]
n_classes = [3, 10]# 26]
n_features = [4, 16]#, 16]

#other params - not depending on dataset name
wrong_names = ['iriss', 'xg', '0']
wrong_name_types = [False, True, 0, 12, dict(), set(), tuple(), []]
wrong_sep_types = [False, True, 0, 12, dict(), set(), tuple(), []]

#parapetres for Reduction classes
basic_reduction = [ENN, DROP1, MSS, PSC, ICF]
# k_param_ok = [1, 2, 3, 10, maxk]
k_praram_wrong = [0, -1, 0.34, 'wrong', False]
to_large_k = [105, 7695]#, 14000]

classifiers = ['knn', 'svm', 'naive_bayers', 'decision_tree','neural_network']

def tuples(x :list, y: list) -> list:
    """Function merging two list into tuples - used in @pytest.mark.parametreize 

    Args:
        x (list): list with same shape
        y (list): list with same shape

    Returns:
        list of tuples
    """
    return [(x[i], y[i]) for i in range(0, len(x))] 
