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
    
    def prepare_reduced_set(self):
        """
        Function prepare reduced dataset grouped by label for using in classificators
        """

        reduced_labels = []
        tmp = []
        for i in range(self.data.n_classes):
            for id in reduced_set[i]:
                reduced_labels.append(list(self.data.class_dict)[i])
                tmp.append(id.tolist())

        np_red_data = np.array(tmp)
        np_red_label = np.array(reduced_labels)

        return np_red_data, np_red_label

    def reduce_instances(self):
        print('Dzieje sie magia ENN')
        n_instances = len(self.data.data_label_train)

        #create array with zeros, ones will be represent instances to remove
        flag_data = np.zeros(n_instances)

        #create array with idexes of data to remove
        remove_id=[]
        
        for idx in range(n_instances):
            instance_class = self.red_lab[idx]
            if (instance_class != self.find_majority_class_knn(self.red_data[idx])):
                flag_data[idx] = 1
                remove_id.append(idx)

        #remove flaged instances
        self.red_data = np.delete(self.red_data, remove_id, axis = 0)
        self.red_lab = np.delete(self.red_lab, remove_id, axis = 0)

        # self.red_data=np.array(self.red_data)
        # self.red_lab=np.array(self.red_lab)
        # for idx in range(n_instances):

            #if (flag_data[idx] == 1):    
                # self.red_data = np.delete(self.red_data, idx, axis=0)
                # self.red_lab = np.delete(self.red_lab, idx, axis=0)
                
                #TODO test

               

        # # return np_red_data, np_red_col
        # self.red_data = np_red_data
        # self.red_lab = np_red_data

