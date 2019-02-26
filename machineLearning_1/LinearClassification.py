#!/usr/bin/env python

import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
#clf.fit(X_train, y_train)

#############################################
#############################################
#############################################

#	Datasets
from sklearn import datasets

iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

print(X_iris.shape, y_iris.shape)
print(X_iris[0], y_iris[0])

#############################################
#############################################
#############################################

#	1st ML-Method:	LINEAR CLASSIFICATION

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

# Retrieve dataset with only first 2-attributes
X, y = X_iris[:, :2], y_iris

# Separate the dataset into 2 datsets:	1) Training set, 2) Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=33)
clf.fit(X_train, y_train)

print(X_train.shape, y_train.shape)

#	Standardizing/Modifying dataset features (i.e. via:	FEATURE-SCALING)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt

colors = ['red', 'greenyellow', 'blue']

###################################
###	xrange = undeclared	###
###################################
for i in range(len(colors)):
  xs = X_train[:, 0][y_train == i]
  ys = X_train[:, 1][y_train == i]
  plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

#	Implementation of SGDClassifier-Method for LINEAR-CLASSIFICATION
# Imports SGDClassifier()
from sklearn.linear_modelsklearn._model import SGDClassifier
# Create Classifier-Object
clf = SGDClassifier()
# Initialize Classifier-Object parameters
clf.fit(X_train, y_train)

# Print LINEAR-BOUNDARY-Coefficients	(for each class: 0, 1, 2)
print(clf.coef_)

# Print Y-INTERCEPT			(for each class: 0, 1, 2)
print(clf.intercept_)

# Convert classification for 3-classes (0, 1, 2) into 3 Binary-Classifications (3 hyperplane-lines for distinguishing each classification)
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
  axes[i].set_aspect('equal')
  axes[i].set_title('Class ' + str(i) + ' versus the rest')
  axes[i].set_xlabel('Sepal Length')
  axes[i].set_ylabel('Sepal Width')
  axes[i].set_xlim(x_min, x_max)
  axes[i].set_ylim(y_min, y_max)
  sca(axes[i])
  plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap = plt.cm.prism)
  ys = (-clf.intercept_[i] - Xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
  plt.plot(xs, ys, hold = True)

# Use predict-method to determine the class (i.e. 0, 1, 2) of a new sample (i.e. Sepal Width = 4.7, Sepal Length = 3.1)
###	NORMALIZE!!!!
print(clf.predict(scaler.transform([[4.7, 3.1]])))
# The command above will combine the result of the 3 binary-classifiers and select the class in which it is most confident. Selects Boundary-Line with longest distance to the instance (4.7, 3.1)
# Confirm the result from the above command, using decision_function() classifier
print(clf.decision_function(scalar.transform([[4.7, 3.1]])))

#############################################
#############################################
#############################################

#	Evaluating results from Classifier (i.e. Accuracy, Precision, Recall, F1-Score)

# Test ACCURACY on Training-Set	(THIS IS NOT WHAT WE WANT TO DO.	DO !USE Training-Set for testing ACCURACY)
from sklearn import metrics
y_train_pred = clf.predict(X_train)
print(metrics.accuracy_score(y_train, y_train_pred))

# Check ACCURACY on Testing/Evaluation-Set
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# EVALUATION-Functions
print(metrics.classification_report(y_test, y_pred, target_names = iris.target_names))
	
# Another EVALUATION:	CONFUSION-MATRIX	-	Shows the number of class-instances predicted to be in a specific class, and illustrates what types of errors are made by classifier
print(metrics.confusion_matrix(y_test, y_pred))

#	K-FOLD CROSS-VALIDATION
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline

# Create a COMPOSITE-ESTIMATOR made by a pipeline of the Standardization-Model & Linear-Model
clf = Pipeline([
  ('scaler', StandardScaler()),
  ('linear_model', SGDClassifier())
])

# Create a K-FOLD CROSS-VALIDATION ITERATOR of K=5 (K = # of Folds)
cv = KFold(X.shape[0], 5, shuffle = True, random_state = 33)

# Number by default used = Number returned by Score-Method of the ESTIMATOR (ACCURACY)
scores = cross_val_score(clf, X, y, cv = cv)
print(scores)
# The 2 commands above will return an array with K number of scores

# Calculate: MEAN, STANDARD-ERROR
from scipy.stats import sem
def mean_score(scores):
  return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

print(mean_score(scores))
