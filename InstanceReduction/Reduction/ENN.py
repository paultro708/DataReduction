from ._Reduction import _Reduction
from ..DataPreparation import DataPreparation
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from time import process_time

class ENN(_Reduction):
    """
    Class representing ENN algorithm. It reduces especially noise instances. 

    Attributes:
        data (DataPreparation): instance of :DataPreparation class containing prepared dataset for ENN alhorithm application
        k (int): number of nearest neighbours takein into account during reduction
        red_data: array with selected data from training subset
        red_lab: array with labels for :red_data
    """

    def __init__(self, data: DataPreparation, k: int=5):
        """Constructor of ENN class. Edited Nearest Neighbours algorith removes instances which label dusagrees with majority class in k nearest neighbours set.

        Args:
            data (DataPreparation): instance of :DataPreparation class containing prepared dataset for ENN alhorithm application
            k (int, optional): number of nearest neighbours takein into account during reduction. Defaults to 5.

        Raises:
            TypeError: when given parameter does not have apriopriate type
            ValueError: when given :k value is less than 1 or greater than number of instances in reducting dataset.
        """
        if not isinstance(data, DataPreparation):
            raise TypeError('Atribute \'data\' must be DataPreparation instance')
        self.data = data
        self.k = k
        self.red_data = []
        self.red_lab = []

        if type(k) != int:
            raise TypeError('k atribute must be integer value')
        elif k < 1:
            raise ValueError('k atribute must have value not less than 1')
        elif k >= len(self.data.data_all_train):
            raise ValueError('k atribute must have value less than number of instances in dataset')

    @staticmethod
    def find_majority_class_knn(red_lab, point, neigh):
        """Function for k nearest neighbors check the majority class and returns it

        Args:
            red_lab: 1d array or list with labels 
            point: point for which will checking k neighbours
            neigh: fitted, sklearn K nearest neighbours classifier

        Returns:
            majority: index of majority class in k nearest neighbours of :point
        """
        # get indexes of nearest neighbours for given point
        indexes = neigh.kneighbors([point], return_distance = False)
        classes = []
        for idx in indexes:
            classes.append(red_lab[idx])
        #check labels of indexes and choose majority
        majority = Counter(classes[0]).most_common(1)[0][0]
        return majority
    

    def reduce_instances(self, return_time = False):
        """Main function in class that uses algorithm ENN on training dataset and create arrays with selected instances and labels 

        Args:
            return_time (bool, optional): . Defaults to False.

        Returns:
            end - start (float): algorithm execution time in seconds 
        """
        print('Reducing the dataset using the ENN algorithm...')
        self.red_data = self.data.data_all_train
        self.red_lab = self.data.data_label_train
        start = process_time()
        n_instances = len(self.data.data_label_train)

        #create array with zeros, ones will be represent instances to remove
        flag_data = np.zeros(n_instances)

        #create array with idexes of data to remove
        remove_id=[]

        #prepare model
        neigh = NearestNeighbors(n_neighbors = self.k).fit(self.red_data)
        
        for idx in range(n_instances):
            instance_class = self.red_lab[idx]
            # flag instance to remove if it's label disagrees with majority class of k nearest neighbours:
            if instance_class != self.find_majority_class_knn(self.red_lab, self.red_data[idx], neigh):
                flag_data[idx] = 1
                remove_id.append(idx)

        #remove flaged instances
        self.red_data = np.delete(self.red_data, remove_id, axis = 0)
        self.red_lab = np.delete(self.red_lab, remove_id, axis = 0)

        end = process_time()

        if return_time:
            return end - start


