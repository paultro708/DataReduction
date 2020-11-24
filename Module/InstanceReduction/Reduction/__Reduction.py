from abc import ABCMeta, abstractmethod, ABC
from time import process_time

from .. import DataPreparation

# import repackage
# repackage.up()

# #dictionary of names of reduction methods
# dataset_path = {"iris": create_path_csv(datasets_folder, 'iris'),
#                 "glass": create_path_csv(datasets_folder, 'glass'),
#                 "letter": create_path_csv(datasets_folder, 'letter'),
#                 "liver": create_path_csv(datasets_folder, 'liver'),
#                 "pendigits": create_path_csv(datasets_folder, 'pendigits'),
#                 "spambase": create_path_csv(datasets_folder, 'spambase'),
#                 "segment": create_path_csv(datasets_folder, 'segment'),
#                 "satimage": create_path_csv(datasets_folder, 'satimage'),
#                 "yeast": create_path_csv(datasets_folder, 'yeast')
#                 }

class __Reduction(metaclass = ABCMeta):
    """
    Abstract class respresenting instance reduction method.
    """   

    def __init__(self, data: DataPreparation):
        """
        Initialize data for reduction
        """
        self.data = data

    @abstractmethod
    def reduce_instances(self, return_time = False):
        """
        Abstract method - main in class. Runs reduction algorithm.
        """
        pass



