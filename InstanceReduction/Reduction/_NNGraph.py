import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

class _NNGraph:
    """
    Class representing neigbourhood. TODO docstrings
    """

    # def __init__(self, data):
    #     self.data = data

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
        neigh = []
        enemy = []

        for i in sort:
            if labels[index] == labels[i]:
                neigh.append(i)
            else:
                enemy.append(i)

        return neigh, enemy
        
    @staticmethod
    def predict(sort_id, class_dict, labels, k, without = None):
        """
        Create prediction of label by k nearest neighbours
        Return true or false if 
        :without: index without which prediction is made
        """
        #dict with labels and count
        lab = list(class_dict.keys())
        n_cl = dict.fromkeys(lab, 0)


        for n in range(1, k+1):
            if n == without:
                break
            l = labels[sort_id[n]]
            n_cl[l] = n_cl[l] + 1
        
        #return label withmax
        return max(n_cl, key=n_cl.get)



    def create_graph(self, data, labels, k = None, n_enem = False):
        
        #create 2d array for dataset with distances between pairs 
        self.dist_arr = distance.cdist(data, data)
        n_ins = len(data)

        if k == None:
            k = n_ins

        # #init arrays
        self.sort = []
        self.sort_id = []
        self.neigh = []
        self.enemy = []
        self.assot = []

        for i in range(n_ins):
            self.assot.append([])
            #create array with indexes of nearest enemy
            # #sort by distance
            self.sort_id.append(np.argsort(self.dist_arr[i]))

        for i in range(n_ins):
            # #create sorted array with indexes of neighbours with same label and enemies - with different label
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