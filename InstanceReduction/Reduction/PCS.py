from ..DataPreparation import DataPreparation
from ._Reduction import _Reduction
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
from collections import defaultdict

from matplotlib import pyplot as plt

import time
from time import process_time

class PCS(_Reduction):

    def __init__(self, data: DataPreparation, r: int = 3):
        if not isinstance(data, DataPreparation):
            raise TypeError('Atribute \'data\' must be DataPreparation instance')
        if type(r) != int:
            raise TypeError('r atribute must be integer value')
        elif r < 1:
            raise ValueError('r atribute must have value not less than 1')
    
        self.data = data
        self.n_clusters = r*data.n_classes
        if self.n_clusters >= len(self.data.data_label_train):
            raise ValueError('The multiple of the number of classes cannot be greater than the number of instances')

        self.red_data = []
        self.red_lab = []

    @staticmethod
    def create_clusters(data, number_of_clusters=20):
        """
        Function creating :n_clusters: clusters from data
        Return array of labels of created clusters from 0 to :number_of_clusters: -1
        """

        # creating clusters using k-means algorithm
        # AgglomerativeClustering(n_clusters=number_of_clusters)#
        clust = KMeans(n_clusters=number_of_clusters, random_state=42)
        clust.fit(data)

        return clust.labels_

    @staticmethod
    def group_id_by_cluster(clusters):
        """
        Function grouping indexes of data by indexes of clusters
        Return dictionary
        """
        # creating list grouping idexes of training data grouped by cluster label
        clusters_with_id = defaultdict(list)
        for idx, cluster in enumerate(clusters):
            clusters_with_id[cluster].append(idx)

        return clusters_with_id

    @staticmethod
    def find_id_of_nearest_point(data_all, indexes, point):
        """
        Function finding index of point in data, nearest to given point. Using for find nerest point of mean in homogeniuos cluster
        """
        # id of nearest, for now the first
        id = indexes[0]
        # minimal distance, for now - the first distance
        min_dist = distance.euclidean(point, data_all[id])

        for i in indexes:
            data = data_all[i]
            dist = distance.euclidean(point, data)
            if min_dist > dist:
                min_dist = dist
                id = i

        return id

    @staticmethod
    def find_nearest_instance(element, indexes_of_data, data_all):
        """
        Function return index of nearest instance to given
        element - index of instance 
        indexes_of_data - indexes of instances from which have to find nearest to element
        data_all - array of all instances to get data of selected index
        """
        point = data_all[element]
        # first temporary index
        idx = 0
        # minimal distance, for now - the first distance
        min_dist = distance.euclidean(point, data_all[idx])
        for i in indexes_of_data:
            if i == element:
                break
            data = data_all[i]
            dist = distance.euclidean(point, data)
            if min_dist > dist:
                min_dist = dist
                idx = i

        return idx

    @staticmethod
    def find_majority_class(number_of_classes, classes_with_indexes):
        """
        Function return index of majority class
        """
        max = len(classes_with_indexes[0])
        majority_class = 0

        for i in range(number_of_classes):
            count = len(classes_with_indexes[i])
            if max < count:
                max = count
                majority_class = i

        return majority_class

    def mean_point_in_cluster(self, data_all, indexes):
        """
        Function calculating mean point in cluster.
        data_all - training dataset
        indexes - array of indexes of cluster form training dataset
        """
        count_of_values = len(indexes)
        if count_of_values == 0:
            raise Exception('Count of indexes in cluster can not be equal 0!')
        sum = 0

        # dimesionality of point
        if data_all.ndim == 1: count_of_features=1
        else: count_of_features = data_all[0].shape[0]
        mean_point = np.array([])

        if count_of_features == 1:
            sum = 0
            for index in indexes: sum += data_all[index]
            return sum/count_of_values
        else: 
            for feature in range(count_of_features):
                sum = 0
                for index in indexes:
                    actual_data = data_all[index]
                    sum += actual_data[feature]
                mean_point = np.append(mean_point, sum/count_of_values)

        return mean_point

    def group_cluster_by_class(self, cluster):
        """
        Function creates array with indexes in cluster grouped by class label
        """
        # initialize array with 0 occurrence of each class
        classes_with_indexes = []

        # initialize array
        for i in range(self.data.n_classes):
            classes_with_indexes.append([])

        for instance_id in cluster:
            # checking label of instance
            class_label_of_instance = self.data.data_label_train[instance_id]
            # add to array for class label
            classes_with_indexes[self.data.class_dict[class_label_of_instance]].append(instance_id)

        return classes_with_indexes

    def check_homogenious(self, n_classes, cluster):
        """
        Function checking if the cluster is homogenious or not
        Return True if is, False if not.
        """
        grouped_cluster = self.group_cluster_by_class(cluster)

        is_homogeniuos = True
        count_of_classes_in_cluster = 0
        for i in range(n_classes):
            if(len(grouped_cluster[i]) > 0):
                count_of_classes_in_cluster += 1
        if (count_of_classes_in_cluster > 1):
            is_homogeniuos = False

        return is_homogeniuos

    def prepare_reduced_set(self, reduced_set):
        """
        Function prepare reduced dataset grouped by label for using in classificators
        reduced_set - dataset grouped by label
        Return:
        np_red_data - uninterrupted array of instances
        np_red_label - array of labels
        TODO: remove repeated values?
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

    def clustering_reduction(self, clusters_with_id, data_all_train):
        """
        The main function of clustering reduction module
        :param clusters_with_id: - indexes of instances from training dataset grouped by indexes of clusters
        :param data_all_train: - training dataset
        :returns: np_red_data - reduced dataset received as a result
        np_red_col - labels of reduced dataset
        """
        classes_with_indexes = []
        # create empty reduced dataset
        reduced_set = []

        # init arrays dimensionality
        for i in range(self.data.n_classes):
            classes_with_indexes.append([])
            reduced_set.append([])

        # for each cluster
        for i in range(self.n_clusters):
            # for each instance in cluster
            for instance_id in clusters_with_id[i]:
                class_label_of_instance = self.data.data_label_train[instance_id]
                classes_with_indexes[self.data.class_dict[class_label_of_instance]].append(
                    instance_id)

            # checking if the cluster is homogenious
            is_homogeniuos = self.check_homogenious(self.data.n_classes, clusters_with_id[i])

            if (is_homogeniuos):
                # find index of majority class - in this case only one possible
                cm = self.find_majority_class(
                    self.data.n_classes, classes_with_indexes)
                # find mean point in cluster
                mean_point = self.mean_point_in_cluster(
                    data_all=self.train, indexes=clusters_with_id[i])
                # print(mean_point)
                # find index of intance located in cluster nearest to mean point
                accept_id = self.find_id_of_nearest_point(
                    data_all=self.train, indexes=clusters_with_id[i], point=mean_point)
                # print(accept_id)

                # add instance within the class 9to reduced set
                reduced_set[cm].append(self.train[accept_id])

            else:
                # majority class in cluster
                cm = self.find_majority_class(
                    self.data.n_classes, classes_with_indexes)
                # print(cm)

                # for each instance in other classes find nearest instance to checked from majority class and belonging class
                # add instances to reduced set
                for class_id in range(self.data.n_classes):
                    if class_id == cm:
                        break
                    for el in classes_with_indexes[class_id]:
                        # nearest form majority class
                        nearest_of_majority_class = self.find_nearest_instance(
                            element=el, indexes_of_data=classes_with_indexes[cm], data_all=self.train)
                        reduced_set[cm].append(
                            self.train[nearest_of_majority_class])
                        # nearest from belonging class
                        nearest_of_actual_class = self.find_nearest_instance(
                            element=el, indexes_of_data=classes_with_indexes[class_id], data_all=self.train)
                        # reduced_set = np.append(reduced_set, data_all_train[nearest_of_actual_class])
                        reduced_set[cm].append(
                            self.train[nearest_of_actual_class])

            # reset array
            classes_with_indexes = []
            for j in range(self.data.n_classes):
                classes_with_indexes.append([])

        np_red_data, np_red_col = self.prepare_reduced_set(reduced_set)
        return np_red_data, np_red_col

    def reduce_instances(self, return_time = False):
        print('Reducing with PCS algorithm ...')
        start = process_time()

        #normalize data
        self.train, self.weights = self.data.normalize(self.data.data_all_train)

        # create clusters
        clusters = self.create_clusters(data = self.train, number_of_clusters = self.n_clusters)

        #group clusters 
        clusters_with_id = self.group_id_by_cluster(clusters)

        #apply main part of algorithm
        self.red_data, self.red_lab = self.clustering_reduction(clusters_with_id, self.train)

        # reverse normalize 
        self.red_data = self.data.reverse_normalize(self.red_data, self.weights)

        end = process_time()
        if return_time:
            return end - start
        
