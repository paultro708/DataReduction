from abc import ABCMeta, abstractmethod, ABC
from time import process_time
import numpy as np
from .. import DataPreparation


class _Reduction(metaclass = ABCMeta):
    """
    Abstract class respresenting instance reduction method.
    """  
    @staticmethod
    def prepare_reduced(inst):
        """Method convering reduced dataset to expected format

        Args:
            inst: insatnce of reduction algorithm
        """
        red = []
        for i in inst.red_data:
            red.append(i.tolist())
        inst.red_lab = np.array(inst.red_lab)
        inst.red_data = np.array(red) 
    
    @abstractmethod
    def reduce_instances(self, return_time = False):
        """
        Abstract method - main in class. Runs reduction algorithm.
        """
        pass



