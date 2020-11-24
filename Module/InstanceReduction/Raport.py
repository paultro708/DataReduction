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

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report 

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

    # def draw_plots(self, col1, col2):
    #     #prepare labels
    #     orig = []
    #     for i in self.original.data_label_train: #range(len(data.data_all_train)):
    #         orig.append(self.original.class_dict[i])
    #     orig = np.array(orig)
    #     red = []
    #     for i in self.reduced_data: #range(len(data.data_all_train)):
    #         red.append(self.original.class_dict[i])
    #     red = np.array(red)

    #     plt.scatter(data.data_all_train[:,0], data.data_all_train[:,1], c = orig) #,c=data.data_label_train)
    #     plt.savefig(".\\plots\\original.png")
    #     # plt.scatter(reduction.red_data[:, 0], reduction.red_data[:, 1], c=red)#,c=reduction.red_lab)
    #     # plt.savefig(".\\plots\\reduced.png")
    @staticmethod
    def create_confusion_matrix(classifier, test_set, test_labels, title, filename, filepath, show = False, save = True):
        plot = plot_confusion_matrix(classifier, test_set, test_labels, cmap=plt.cm.Blues)
        #title = "Confusion matrix\n reduced data - classifier: " + str(c_type)
        plot.ax_.set_title(title)
        if save:        
            plot.figure_.savefig(filepath + filename)
        if show:
            plt.show()

    def raport(self, original_set, original_labels, reduced_set, reduced_labels, test_set, test_labels, c_type = 'all'):
        if (c_type !='all') and (c_type not in classifiers):
            raise Exception("Classifier type not exist in available set!")
        else:
            if c_type == 'all':
                for c_t in classifiers:
                    self.raport(original_set, original_labels, reduced_set, reduced_labels, test_set, test_labels, c_t)

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
                title = "Confusion matrix\n original data - classifier: {}".format(str(c_type))
                self.create_confusion_matrix(classifier, test_set, test_labels, title, "Original " + str(c_type),  ".\plots\\", True)

                #print raport with metrics for original training data
                print('=============')
                print("Classifier:  ", c_type)
                print('=============')
                print("Raport for original dataset")
                print('Count of instances: ', len(original_labels))
                print(classification_report(test_labels, predict)) 
                print("Cohen's Kappa: {:.2f}".format(cohen_kappa_score(test_labels, predict)))
                print('===')
                print("Training time: ", training_time)
                print("Predicting time: ", prediction_time)

                #same for reduced training dataset
                classifier = classifiers[c_type]
                #train
                start = time.clock()
                classifier.fit(reduced_set, reduced_labels)
                end = time.clock()
                training_time = end - start
                #predict
                start = time.clock()
                predict = classifier.predict(test_set)
                end = time.clock()
                prediction_time = end - start

                #create confusion matrix
                title = "Confusion matrix\n reduced data - classifier: {}".format(str(c_type))
                self.create_confusion_matrix(classifier, test_set, test_labels, title, "Reduced " + str(c_type),  ".\plots\\", True)

                print("\nRaport for reduced dataset")
                print('Count of instances: ', len(reduced_labels))
                print(classification_report(test_labels, predict))
                print("Cohen's Kappa: {:.2f}".format(cohen_kappa_score(test_labels, predict)))
                print('===')
                print("Training time: ", training_time)
                print("Predicting time: ", prediction_time, "\n")
                print('Reduction factor: {:.2f} %'.format((len(original_labels) - len(reduced_labels))/len(original_labels)*100))
                print('===')

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
        pass

    def print_raport(self, confusion_matrix = None):
        # confusion_matrix = {filepath: "x", show = False, save = True}
        self.raport(self.original.data_all_train, self.original.data_label_train, self.reduced_data, self.reduced_label, self.original.data_all_test, self.original.data_label_test, c_type = self.c_type)
    
   
# raport_classify(data_all_train, data_label_train, np_red_data, np_red_col, data_all_test, data_label_test, 'knn')
    