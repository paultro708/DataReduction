from DataPreparation import DataPreparation
from InstanceReduction import InstanceReduction
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class ENN(InstanceReduction):
    """
    Class representing ENN algorithm. It reduces especially noise instances.
    """

    def __init__(self, data: DataPreparation, k=5):
        self.data = data
        self.k = k
        self.red_data = self.data.data_all_train
        self.red_lab = self.data.data_label_train

    def find_majority_class_knn(self, point):
        """
        Function for k nearest neighbors check the majority class and returns it
        """
        neigh = NearestNeighbors(n_neighbors = self.k).fit(self.red_data)
        indexes = neigh.kneighbors([point], return_distance = False)
        classes = []
        for idx in indexes:
            classes.append(self.red_lab[idx])
        #check labels of indexes and choose majority
        return Counter(classes[0]).most_common(1)[0][0]
    

    def reduce_instances(self):
        print('Dzieje sie magia ENN')
        
        for idx in range(len(self.red_data)):
            instance_class = self.red_lab[idx]
            if (instance_class != self.find_majority_class_knn(self.red_data[idx])):
                self.red_data = np.delete(self.red_data, idx, axis=0)
                self.red_lab = np.delete(self.red_lab, idx, axis=0)
                #TODO test

               

        # # return np_red_data, np_red_col
        # self.red_data = np_red_data
        # self.red_lab = np_red_data

