from DataPreparation import DataPreparation
from InstanceReduction import InstanceReduction
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from time import process_time
from scipy.spatial import distance

class KNN(InstanceReduction):
    """
    Class representing ENN algorithm. It reduces especially noise instances.
    """

    def __init__(self, data: DataPreparation, k=5):
        self.data = data
        self.k = k
        self.red_data = self.data.data_all_train
        self.red_lab = self.data.data_label_train

    def reduce_instances(self, return_time = False):
        print('Dzieje sie magia KNN')
        start = process_time()
        ################
        #create 2d array for dataset with distances between pairs 
        dist_arr = distance.cdist(self.data.data_all_train, self.data.data_all_train)
        n_ins = len(self.data.data_all_train)
        tmp = np.arange(n_ins) #array of original indexes

        for i in range(n_ins):
            #sort by distance
            sorted = np.argsort(dist_arr[i])
            neigh = []
            enemy = []
            #create sorted array with indexes of neighbours with same label and enemies - with different label
            """
            TODO what with neighbours and enemies
            """
            for j in sorted:
                if self.data.data_label_train[i] == self.data.data_label_train[j]:
                    neigh.append(j)
                else:
                    enemy.append(j)

        ################
        end = process_time()

        if return_time:
            return end - start

