from ._Reduction import _Reduction
from ..DataPreparation import DataPreparation
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from time import process_time
from scipy.spatial import distance
from time import process_time

class MSS(_Reduction):
    """
    Class representing Modified Selective Subset algorithm. It selects all instances that is nearest to different class than same. 
    It starts with indexes sorted by distance of nearest enemy.

    Args:
        _Reduction: abstract class of reduction algorithm

    Attributes:
        data (DataPreparation): DataPreparation instance with original dataset 
        red_data: reduced dataset
        red_lab: labels of reduced dataset
        k (int):parameter for k nearest neighbourhood
        train: training normalized datased
    """

    def __init__(self, data: DataPreparation, k: int=3):
        """Initialization if MSS algorithm.

        Args:
            data (DataPreparation): instance of prepared dataset
            k (int, optional): count of nearest neighbours. Defaults to 3.

        Raises:
            TypeError: if type of parameter is not apriopriate
            ValueError: if value of :k: is less than 1 or grater than number of instances in dataset 
        """
        if not isinstance(data, DataPreparation):
            raise TypeError('Atribute \'data\' must be DataPreparation instance')
        self.data = data
        self.k = k
        if type(k) != int:
            raise TypeError('k atribute must be integer value')
        elif k < 1:
            raise ValueError('k atribute must have value not less than 1')
        elif k >= len(self.data.data_label_train):
            raise ValueError('k atribute must have value less than number of instances in dataset')
        self.red_data = []
        self.red_lab = []

    @staticmethod
    def group_neigh_enemies(labels, index:int, sort):
        """Function grouping indexes of points with same label and enemies - points with different label.
        Atributes:
        :labels: - array of label for each point in dataset
        :index: - index of point
        :dist_arr: - 1d array of indexes sorted by distance beetween :index:
        Return arrays:
        :neigh: - array of points with same label as point with :index:
        :enemy: - array of points with diferent label than point with :index:

        Args:
            labels: array with labels of instances in dataset
            index (int): index of instance
            sort: array of indexes sorted by distance

        Returns:
            list: list of indexes with neighbours sorted by distance
            ist: list of indexes with enemies sorted by distance
        """
        #init empty arrays
        neigh = []
        enemy = []
        for i in sort:
                if labels[index] == labels[i]:
                    neigh.append(i)
                else:
                    enemy.append(i)

        return neigh, enemy


    def reduce_instances(self, return_time = False):
        """Main function in class reducing instances woth MSS algoritm.

        Args:
            return_time (bool, optional): if True retuns processing time in seconds. Defaults to False.

        Returns:
            float: processing time in seconds
        """
        print('Reducing the dataset using the MSS algorithm...')
        start = process_time()
        
        self.red_data = []
        self.red_lab = []
        #normalization for creation distance array
        self.train =  self.data.normalize(self.data.data_all_train)[0]
        
        #create 2d array for dataset with distances between pairs 
        dist_arr = distance.cdist(self.train, self.train)
        n_ins = len(self.data.data_all_train)
        tmp = np.arange(n_ins) #array of original indexes
        nearest_enemy = []
        nearest_enemy_dist = []
    

        #create array with indexes of nearest enemy
        for i in range(n_ins):
            #sort by distance
            sort = np.argsort(dist_arr[i])
            #create sorted array with indexes of neighbours with same label and enemies - with different label
            neigh, enemy = self.group_neigh_enemies(self.data.data_label_train, i, sort)
            #add index of nearest enemy to array 
            nearest_enemy.append(enemy[0])
            nearest_enemy_dist.append(dist_arr[i][enemy[0]])

        #indexes sorted by nearest enemy distance
        sort = np.argsort(nearest_enemy_dist)
        s = sort
        added = []
        for i in sort:
            add = False
            for j in sort:
                if j in s and dist_arr[i][j] < nearest_enemy_dist[j]:
                    s = np.delete(s, np.where(s==j))
                    add = True
            
            if add and i not in added:
                self.red_data.append(self.data.data_all_train[i])
                self.red_lab.append(self.data.data_label_train[i])
                added.append(i)

        super().prepare_reduced(self)
        end = process_time()

        if return_time:
            return end - start
