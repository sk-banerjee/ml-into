import sys
from sklearn import datasets

iris = datasets.load_iris()

data = iris.data
labels = iris.target

from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size= .5)

try:
    classifier_type = sys.argv[1]
except:
    classifier_type = "dt"

if(classifier_type == 'dt'):
    from sklearn import tree
    classifier = tree.DecisionTreeClassifier()
    print("Decision Tree Classifier")
elif(classifier_type == 'knn'):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier()
    print("K Nearest Neighbors Classifier")
else:
    print("Unkown Classifier")
    exit()

classifier.fit(data_train, labels_train)
predictions = classifier.predict(data_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, predictions))
