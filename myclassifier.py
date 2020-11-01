import sys
from sklearn import datasets
import random
from scipy.spatial import distance

def euclidean_distance(a,b):
    return distance.euclidean(a,b)

class myKNN():
    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, data):
        predictions = list()
        for row in data:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euclidean_distance(row, self.data[0])
        best_index = 0
        for i in range(1, len(self.data)):
            dist = euclidean_distance(row, self.data[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.labels[best_index]

iris = datasets.load_iris()

data = iris.data
labels = iris.target

from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size= .2)

print("Traning set has {} elements. Test set has {} elements.".format(len(data_train), len(data_test)))

try:
    user_selected_clssifier = list(sys.argv[1])
except:
    user_selected_clssifier = ["dt", 'knn', 'myknn']

for classifier_type in user_selected_clssifier:
    if(classifier_type == 'dt'):
        from sklearn import tree
        classifier = tree.DecisionTreeClassifier()
        print("Decision Tree Classifier")
    elif(classifier_type == 'knn'):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier()
        print("K Nearest Neighbors Classifier")
    elif(classifier_type == 'myknn'):
        classifier = myKNN()
        print("myKNN classifier")
    else:
        print("Unkown Classifier")
        continue

    classifier.fit(data_train, labels_train)
    predictions = classifier.predict(data_test)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(labels_test, predictions))
