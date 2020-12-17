import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from array import *

class _NNGraph:
    """
    Class creating neighbourhood graph. Calculates arrays of nearest neighbours, enemies and assotiates.
    After creating graph can be done prediction.

    Attributes:
        dist_arr: array with calculated distance between all pairs of instances in dataset
        sort_id: array containing for each instance indexes of instances sorted by distance to it
        self.neigh = array containing for each instance indexes of intances with same label sorted by distance 
        self.enemy = array containing for each instance indexes of intances with different label sorted by distance 
        self.assot = array containing for each instance indexes of intances for which :neigh: contains it
    """

    @staticmethod
    def group_neigh_enemies(labels, index: int, sort):
        """Function grouping indexes of points with same label and enemies - points with different label.

        Args:
            labels: array of label for each point in dataset
            index (int): index of point
            sort ([type]): 1d array of indexes sorted by distance beetween :index:

        Raises:
            ValueError: if :labels: or :sort: is different type than list or np.nparray or is empty
            TypeError: if given parameter has wrong type

        Returns:
            array('I'): array of points with same label as point with :index:
            array('I'): array of points with diferent label than point with :index:
        """

        for l in [labels, sort]:
            if type(l) == np.ndarray and (l.ndim != 1 or l.size == 0):
                raise ValueError('\'{}\' must be 1d not empty numpy array'.format(l))
            elif type(l) == list and len(l) == 0:
                raise ValueError('\'{}\' must be not empty list'.format(l))
            
            elif type(l) not in [np.ndarray, list]:
                raise TypeError('\'{}\' must be list or numpy array'.format(l))
        if type(index) != int:
            raise TypeError('index must be integer value')
        
        #init empty arrays
        neigh = array('I')
        enemy = array('I')

        for i in sort:
            if labels[index] == labels[i]:
                neigh.append(i)
            else:
                enemy.append(i)

        return neigh, enemy
        
    @staticmethod
    def predict(sort_id, class_dict:dict, labels, k:int, without = None):
        """Function creating prediction of label by k nearest neighbours

        Args:
            sort_id ([type]): [description]
            class_dict (dict): [description]
            labels ([type]): [description]
            k (int): [description]
            without (int, optional): index which is ommited during prediction. Defaults to None.

        Returns:
            string: predicted label
        """
        #dict with labels and count
        lab = list(class_dict.keys())
        n_cl = dict.fromkeys(lab, 0)


        for n in range(1, k+1):
            if n == without:
                break
            l = labels[sort_id[n]]
            n_cl[l] = n_cl[l] + 1
        
        #return label with max count
        return max(n_cl, key=n_cl.get)



    def create_graph(self, data, labels, k = None, n_enem = False):
        """Man function in class calculating arrays of

        Args:
            data: array with dataset 
            labels: array of labels for each instance in dataset
            k (int, optional): parameter determining count of nearest instances taking into account. Defaults to None.
            n_enem (bool, optional): parameter precises if arrays should be limited to nearest enemy. Defaults to False.
        """
        
        #create 2d array for dataset with distances between pairs 
        self.dist_arr = distance.cdist(data, data)
        n_ins = len(data)

        #if k is not given, take int account all instances
        if k == None:
            k = n_ins

        # #init arrays
        self.sort_id = []
        self.neigh = []
        self.enemy = []
        self.assot = []

        for i in range(n_ins):
            self.assot.append([])
            #create array with indexes of nearest enemy
            #sort by distance
            self.sort_id.append(np.argsort(self.dist_arr[i]))

        for i in range(n_ins):
            #create sorted array with indexes of neighbours with same label and enemies - with different label
            n, e = self.group_neigh_enemies(labels, i, self.sort_id[i][:k+1])
            if n_enem:
                self.enemy.append(e[:3])
                self.neigh.append(n[1:(e[0]+1)])
            else:
                self.neigh.append(n[1:k+1])
                self.enemy.append(e[:k])
            
            #add i to assotiates:
            for n in self.neigh[i]:
                self.assot[n].append(i)