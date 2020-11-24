import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

class _NNGraph:
    """
    Class representing neigbourhood 
    """

    def __init__(self, data):
        self.data = data

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

    def predict(self, index, class_dict, labels, k, without = None):
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
            l = labels[self.sort_id[index][n]]
            n_cl[l] = n_cl[l] + 1
        
        #return label withmax
        return max(n_cl, key=n_cl.get)



    def create_graph(self, data, labels):
        
        #create 2d array for dataset with distances between pairs 
        self.dist_arr = distance.cdist(data, data)
        n_ins = len(data)

        # #init arrays
        # self.sort = np.array([])
        # self.sort_id = np.array([])
        # self.neigh = np.array([])
        # self.enemy = np.array([])
        # self.assot = np.array([])
        #init arrays
        self.sort = []
        self.sort_id = []
        self.neigh = []
        self.enemy = []
        self.assot = []

        # for i in range(n_ins):
        #     self.sort = np.append(self.sort, np.array(list))
        #     self.sort_id = np.append(self.sort_id, np.array(list))
        #     self.neigh = np.append(self.neigh, np.array(list))
        #     self.enemy = np.append(self.enemy, np.array(list))
        #     self.assot = np.append(self.assot, np.array(list))
    

        # #create array with indexes of nearest enemy
        # for i in range(n_ins):
        #     # #sort by distance
        #     # self.sort_id[i] = np.argsort(self.dist_arr[i])

        #     # #create sorted array with indexes of neighbours with same label and enemies - with different label
        #     # self.neigh[i], self.enemy[i] = self.group_neigh_enemies(labels, i, self.sort_id[i])
            
        #     # #add i to assotiates:
        #     # for n in self.neigh[i]:
        #     #     l = self.assot[n]
        #     #     l.append(i)
        #     #     self.assot[n] = l
        #     np.appenf

        #     for i in range(n_ins):
        #         self.sort = np.append(self.sort, np.array(list))
        for i in range(n_ins):
            self.assot.append([])
        #create array with indexes of nearest enemy
        for i in range(n_ins):
            # #sort by distance
            self.sort_id.append(np.argsort(self.dist_arr[i]))

            # #create sorted array with indexes of neighbours with same label and enemies - with different label
            n, e = self.group_neigh_enemies(labels, i, self.sort_id[i])
            self.neigh.append(n)
            self.enemy.append(e)
            
            # #add i to assotiates:
            for n in self.neigh[i]:
                self.assot[n].append(i)