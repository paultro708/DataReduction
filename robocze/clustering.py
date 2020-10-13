import time
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets

from collections import defaultdict

number_of_classes = 3

# spliting dataset
data_all, data_label = datasets.load_iris(return_X_y=True)
data_all_train, data_all_test, data_label_train, data_label_test = train_test_split(
    data_all, data_label, test_size=0.3)

plt.scatter(data_all[:, 0], data_all[:, 2], c=data_label)


t = time.process_time()
# knn classify - measure time and accuracy for original train dataset
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_all_train, data_label_train)
accuracy = knn.score(data_all_test, data_label_test)
elapsed = time.process_time() - t

print("Time:  ", elapsed)
print('Accuracy:  ', accuracy)

# creating clusters kmeans
number_of_clusters = 3 * number_of_classes
clust = KMeans(n_clusters=number_of_clusters)
clust.fit(data_all_train)
# print label of clusters
print(clust.labels_)


# create empty reduced dataset
reduced_set = np.array([])
# adding value to reduced dataset
reduced_set = np.append(reduced_set, 1)


# create array to group cluster intsnaces
clusters = np.array([])
for i in clust.labels_:
    clusters = np.append(clusters, [i, data_all_train[i]])



# creating list grouping idexes of training data grouped by cluster label
clusters_with_id = defaultdict(list)
for idx, cluster in enumerate(clust.labels_):
    clusters_with_id[cluster].append(idx)

# clusters_with_id[key-czyli label of cluster :)]
# clusters_with_id[6]

# utowrzenie tablicy n wartości, gdzie n to liczba klas, w której bedzie trzymana liczba wystapień danej klasy
# initialize array with 0 occurrence of each class
classes_count = np.array([])
for i in range(number_of_classes):
    classes_count = np.append(classes_count, 0)


"""
Narazie tylko sprawdzanie który klaster jest homogeniczny który nie
TODO dla homogenicznych wyliczać średnią, srpawdzać, która z instancji jest najbliżej średniej (odleglość euklidesowa)
i tą najblizszą dodawać do tabvlicy zredukowanego zbioru danych
TODO dla niehomogenicznych obliczac odległości do tych z innej klasy, ewentualnie na początek pozostawić cały niehomogeniczny klaster i przetestować
TODO podzielić na funkcje
TODO przemyśleć sposób ładowania i rozdzielania danych labeli.. ewentualnie jakie wymagania do datasetu danego do programu?
"""
# dla każdego klastra
for i in range(number_of_clusters):
    # zliczamy liczbę wystąpień każdej z klas
    # dla każdego id wystepującej danej w klastrze sprawdzić klasę i zwiększyć liczbę wystąpień
    for instance_id in clusters_with_id[i]:
        class_label_of_instance = data_label_train[instance_id]
        classes_count[class_label_of_instance] += 1
    print(classes_count)

    # reset classes counter
    classes_count = np.array([])
    for i in range(number_of_classes):
        classes_count = np.append(classes_count, 0)


# wartość średnia wszystkich instancji w klastrze
"""
data_all - dane, do których odnosza się indexy
indexes - indexy z data_all, których mamy policzyć średnią
"""

def mean_in_cluster(data_all, indexes):
    count_of_values = len(indexes)
    sum = 0

    # liczba wszystkich wymiarów wzgledem których ma być sumowane
    count_of_features = data_all[0].shape[0]

    mean = np.array([])
    
    for feature in range(count_of_features):
        for index in indexes:
            actual_data = data_all[index]
            sum += actual_data[feature]
        mean = np.append(mean, sum/count_of_values)
        sum = 0

    return mean


#clusters_with_id[key-czyli label of cluster :)]
#clusters_with_id[6]

#utowrzenie tablicy n wartości, gdzie n to liczba klas, w której bedzie trzymana liczba wystapień danej klasy
#initialize array with 0 occurrence of each class
classes_count = np.array([])
for i in range(number_of_classes):
    classes_count = np.append(classes_count, 0)

#dla każdego klastra
for i in range(number_of_clusters):
    #zliczamy liczbę wystąpień każdej z klas
    #dla każdego id wystepującej danej w klastrze sprawdzić klasę i zwiększyć liczbę wystąpień
    for instance_id in clusters_with_id[i]:
        class_label_of_instance = data_label_train[instance_id]
        classes_count[class_label_of_instance]+=1
    print(classes_count)

    is_homogeniuos = True
    count_of_classes_in_cluster = 0
    for i in range(number_of_classes):
        if(classes_count[i] > 0):
            count_of_classes_in_cluster+=1

    if (count_of_classes_in_cluster > 1):
        is_homogeniuos = False

    if (is_homogeniuos):
        #TODO mean
        print('homogenious')
        #obliczyć średnia wszystkich wartości instancji w klastrze w postaci obiektu-
        #dla każdej instancji obliczyć odległość euklidesową od tej średniej i zapisać w jakiejś pomocniczej np.array?
        #wybrać 
        actual_ids=clusters_with_id[i]
        print(mean_in_cluster(data_all_train, actual_ids))
        

        
    #reset classes counter     
    classes_count = np.array([])
    for i in range(number_of_classes):
        classes_count = np.append(classes_count, 0)