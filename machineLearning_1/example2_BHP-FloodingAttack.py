#!/usr/bin/env python

import sys
import scipy
import numpy
import matplotlib
import pandas#										### Used for EXPLORATORY/DESCRIPTIVE/DATA-VIZUALIZATION statistics
import sklearn

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('##############################################################################')
print('')

###	1.	LOAD DATA:
#########################
#	1.1:	Import Library Modules/Functions/Objects
#import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#	1.2:	Load Dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00404/OBS-Network-DataSet_2_Aug27.arff"
names = ['node-num.', 'Utilised Bandwidth Rate', 'Packet Drop Rate', 'Full_Bandwidth', 'Avg_Delay_Time_Per_Sec', 'Percentage_Of_Lost_Pcket_Rate', 'Percentage_Of_Lost_Byte_Rate', 'Packet Received Rate', 'of Used_Bandwidth', 'Lost_Bandwidth', 'Packet Size_Byte', 'Packet_Transmitted', 'Packet_Received', 'Packet_lost', 'Transmitted_Byte', 'Received_Byte', '10-Run-AVG-Drop-Rate', '10-Run-AVG-Bandwidth-Use', '10-Run-Delay', 'Node Status', 'Flood Status', 'Class']#	### Specifies Column-Names
dataset = pandas.read_csv(url, names=names)
print('##############################################################################')
print('')

###	2.	SUMMARIZE DATA:
##############################
#	2.1:	Dimensions of the dataset
print('SHAPE(ROWS, COLUMNS):', dataset.shape)
#	2.2:	Peek at the data itself
print('HEAD(20):')
print(dataset.head(20))
print('')
#	2.3:	Statistical summary of all attributes
print('STATISTICAL SUMMARY FOR EACH COLUMN/ATTRIBUTE:')
print(dataset.describe())
print('')
#	2.4:	Breakdown of the data by the class variable
print(dataset.groupby('class').size())
print('##############################################################################')
print('')

###	3.	DATA VISUALIZATION:
##################################
#	3.1:	Univariate Plots
#			BOX & WHISKER PLOTS
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
#			HISTOGRAM PLOT
dataset.hist()
plt.show()

#	3.1:	Multivariate Plots
#			SCATTER-PLOT MATRIX
scatter_matrix(dataset)
plt.show()

print('##############################################################################')
print('')

###	4.	MODELING ALGORITHMS:
###################################
#	4.1	Separate/Create a Validation-Dataset
#			Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
# TRAINING DATA (100 - 20% = 80%):	X_train, Y_train
# TESTING/VALIDATION DATA (20%):	X_validation, Y_validation 
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#	4.2	Test Harness
#			K-Fold (K = 10) Cross-Validation (Estimate ACCURACY)
seed = 7
scoring = 'accuracy'

#	4.3	BUILDING MODELS
#			LINEAR Algorithms: Logistic Regression (LR), Linear Discriminant Analysis (LDA)
#			NON-LINEAR Algorithms:	K-Nearest Neighbors (KNN), Classification and Regression Trees (CART), Gaussian Naive Bayes (NB), Support Vector Machines (SVM)
models = []
print('MODEL EVALUATIONS:	ACCURACY')
models.append(('Logistic Regression (LR)			', LogisticRegression()))
models.append(('Linear Discriminant Analysis (LDA)		', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbors (KNN)			', KNeighborsClassifier()))
models.append(('Classification and Regression Trees (CART)	', DecisionTreeClassifier()))
models.append(('Gaussian Naive Bayes (NB)			', GaussianNB()))
models.append(('Support Vector Machine (SVM)			', SVC()))
# evaluate each model in turn
results = []
names = []

from sklearn.model_selection import KFold, cross_val_score

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#	4.4	SELECTING BEST MODEL
#			Compare Algorithms:	Plotting model-evaluation results & comparing the SPREAD & MEAN-ACCURACY
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
print('##############################################################################')
print('')

###	5.	MAKING PREDICTIONS:
##################################
#					Take most accurate model to determine the model's accuracy on the validation set

#					Run the model on the VALIDATION/TESTING-Set, summarizing results as FINAL-ACCURACY SCORE
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
