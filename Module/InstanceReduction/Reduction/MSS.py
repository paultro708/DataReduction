from ._Reduction import _Reduction
from .. import DataPreparation
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from time import process_time
from scipy.spatial import distance
import datetime

class MSS(_Reduction):
    """
    Class representing ENN algorithm. It reduces especially noise instances.
    """

    def __init__(self, data: DataPreparation, k=5):
        self.data = data
        self.k = k
        self.red_data = []
        self.red_lab = []

    @staticmethod
    def group_neigh_enemies(labels, index, sort):
        """
        Function grouping indexes of points with same label and enemies - points with different label.
        Atributes:
        :labels: - array of label for each point in dataset
        :index: - index of point
        :dist_arr: - 1d array of indexes sorted by distance beetween :index:
        Return arrays:
        :neigh: - array of points with same label as point with :index:
        :enemy: - array of points with diferent label than point with :index:
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

    def prepare_reduced(self):
        red = []
        lab = []
        # for i in range(len(self.red_lab)):
        #     red.append(self.red_lab[i].tolist())
            #lab.append(self.red_lab[i].tolist())
        for i in self.red_data:
            red.append(i.tolist())
        self.red_lab = np.array(self.red_lab)
        self.red_data = np.array(self.red_data)



    def reduce_instances(self, return_time = False):
        print('Dzieje sie magia KNN')
        start = process_time()
        ################
        #create 2d array for dataset with distances between pairs 
        dist_arr = distance.cdist(self.data.data_all_train, self.data.data_all_train)
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
            #add index of nearest enemy to aray 
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

        self.prepare_reduced()
        ################
        end = process_time()

        if return_time:
            return str(datetime.timedelta(seconds=(end - start))) 
