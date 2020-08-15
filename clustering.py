import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets


number_of_classes = 3

#spliting dataset
data_all, data_label= datasets.load_iris(return_X_y=True)
data_all_train, data_all_test, data_label_train, data_label_test = train_test_split(data_all, data_label, test_size=0.3)

plt.scatter(data_all[:,0], data_all[:,2],c=data_label)

from sklearn.neighbors import KNeighborsClassifier
import time

t = time.process_time()
#knn classify - measure time and accuracy for original train dataset
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_all_train,data_label_train)
accuracy = knn.score(data_all_test,data_label_test)
elapsed = time.process_time() - t

print("Time:  ", elapsed)
print('Accuracy:  ', accuracy)

#creating clusters kmeans
number_of_clusters = 3* number_of_classes
clust= KMeans(n_clusters=number_of_clusters)
clust.fit(data_all_train)
#print label of clusters
print(clust.labels_)


#create empty reduced dataset
reduced_set = np.array([])
#adding value to reduced dataset
reduced_set= np.append(reduced_set,1)


#create array to group cluster intsnaces
clusters = np.array([])
for i in clust.labels_:
    clusters = np.append(clusters, [i, data_all_train[i]])


rom collections import defaultdict

#creating list grouping idexes of training data grouped by cluster label
clusters_with_id = defaultdict(list)
for idx, cluster in enumerate(clust.labels_):
    clusters_with_id[cluster].append(idx)

#clusters_with_id[key-czyli label of cluster :)]
#clusters_with_id[6]

#utowrzenie tablicy n wartości, gdzie n to liczba klas, w której bedzie trzymana liczba wystapień danej klasy
#initialize array with 0 occurrence of each class
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
#dla każdego klastra
for i in range(number_of_clusters):
    #zliczamy liczbę wystąpień każdej z klas
    #dla każdego id wystepującej danej w klastrze sprawdzić klasę i zwiększyć liczbę wystąpień
    for instance_id in clusters_with_id[i]:
        class_label_of_instance = data_label_train[instance_id]
        classes_count[class_label_of_instance]+=1
    print(classes_count)
        
    #reset classes counter     
    classes_count = np.array([])
    for i in range(number_of_classes):
        classes_count = np.append(classes_count, 0)
        