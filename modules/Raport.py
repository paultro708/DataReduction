from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import time

classifiers = {'knn': KNeighborsClassifier(), 
                'svm': svm.SVC(), 
                'naive_bayers': GaussianNB(), 
                'decision_tree': DecisionTreeClassifier(),
                'neutral_network': MLPClassifier()}

classify_metrics = {"accuracy": accuracy_score,
                    "f1": f1_score,
                    "precision": precision_score,
                    "recall": recall_score}


class Raport():

    def __init__(self, original, reduced_data, reduced_label, c_type = 'all'):
        self.original = original
        self.reduced_data = reduced_data
        self.reduced_label = reduced_label
        self.c_type = c_type

    def raport_classify(self, original_set, original_labels, reduced_set, reduced_labels, test_set, test_labels, c_type = 'all'):
        """
        TODO special classifier
        Function generating raport 

        :original_set: original training dataset
        :original_labels: labels of classes in original trainign dataset
        :reduced_set: reduced training dataset
        :reduced_labels: labels of classes in reduced training dataset
        :test_set: testing dataset
        :test_labels: labels of classes in testing dataset, expected classification results
        :c_type: name of classifier, optional attribute, if not given raport is generated for all classifiers
                        Possible values: 'knn': KNeighborsClassifier(), 
                                            'svm': svm.SVC(), 
                                            'naive_bayers': GaussianNB(), 
                                            'decision_tree': DecisionTreeClassifier(),
                                            'neutral_network': MLPClassifier().
        """
        #TODO fix with fit classifier 
        try:
            if c_type == 'all':
                for c_t in classifiers:
                    self.raport_classify(original_set, original_labels, reduced_set, reduced_labels, test_set, test_labels, c_t)
            else:
                #select classifier
                classifier = classifiers[c_type]
                #train with original dataset and time measure
                start = time.clock()
                classifier.fit(original_set, original_labels)
                end = time.clock()
                training_time = end - start

                #make predictions and time measure
                start = time.clock()
                predict = classifier.predict(test_set)
                end = time.clock()
                prediction_time = end - start

                #create confusion matrix
                plot = plot_confusion_matrix(classifier, test_set, test_labels, cmap=plt.cm.Blues)
                title = "Confusion matrix\n original data - classifier: " + str(c_type)
                plot.ax_.set_title(title)
                plot.figure_.savefig(".\plots\\" + c_type + ' - original data')  

                #print raport with metrics for original training data
                print('=============')
                print("Classifier:  ", c_type)
                print('=============')
                print("Raport for original dataset")
                print('Count of instances: ', len(original_labels))
                for metric in classify_metrics:
                    if (metric == 'accuracy'):
                        print(metric, ": ", classify_metrics[metric](test_labels, predict))
                    else: 
                        print(metric, ": ", classify_metrics[metric](test_labels, predict, average=None))
                print('===')
                print("Training time: ", training_time)
                print("Predicting time: ", prediction_time)


                #same for reduced training dataset
                classifier = classifiers[c_type]
                start = time.clock()
                classifier.fit(reduced_set, reduced_labels)
                end = time.clock()
                training_time = end - start
                start = time.clock()
                predict = classifier.predict(test_set)
                end = time.clock()
                prediction_time = end - start

                plot = plot_confusion_matrix(classifier, test_set, test_labels, cmap=plt.cm.Blues)
                title = "Confusion matrix\n reduced data - classifier: " + str(c_type)
                plot.ax_.set_title(title)
                plot.figure_.savefig(".\plots\\" + c_type + ' - reduced data')  

                print("\nRaport for reduced dataset")
                print('Count of instances: ', len(reduced_labels))
                for metric in classify_metrics:
                    if (metric == 'accuracy'):
                        print(metric, ": ", classify_metrics[metric](test_labels, predict))
                    else: 
                        print(metric, ": ", classify_metrics[metric](test_labels, predict, average=None))
                    
                print('===')
                print("Training time: ", training_time)
                print("Predicting time: ", prediction_time, "\n")
        except KeyError:
            print('Choose existing classifier!')

    def print_raport(self):
        self.raport_classify(self.original.data_all_train, self.original.data_label_train, self.reduced_data, self.reduced_label, self.original.data_all_test, self.original.data_label_test, c_type = self.c_type)
    
   
# raport_classify(data_all_train, data_label_train, np_red_data, np_red_col, data_all_test, data_label_test, 'knn')
    