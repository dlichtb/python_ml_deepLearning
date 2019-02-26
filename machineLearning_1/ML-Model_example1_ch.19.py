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
# Load libraries
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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
# Load dataset
filename = '____.csv'
names = ['', '', '', '']
dataset = read_csv(filename, names=names)
print('##############################################################################')
print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	2.	SUMMARIZE DATA:
##############################
#	2.1:	Dimensions of the dataset
print('SHAPE(ROWS, COLUMNS):', dataset.shape)
#	2.2:	Data-types of each attribute
set_option('display.max_rows', 500)
print('ATTRIBUTE DATA-TYPES:')
print(dataset.dtypes)
print('')
#	2.2:	Peek at the data itself
set_option('display.width', 100)
print('HEAD(20):')
print(dataset.head(20))
print('')
#	2.3:	Summarize ATTRIBUTE-DISTRIBUTION
#			- Change precision to 3 places
set_option('precision', 3)
print(dataset.describe())
print('##############################################################################')
print('')
#	2.4:	Breakdown of the data by the CLASS variable:	Class Distribution
print(dataset.groupby(60).size())

#	##############################################################################
#	##############################################################################
#						OR					
#	##############################################################################
#	##############################################################################

#	from pandas import read_csv
#	filename = '____.csv'
#	names = ['', '', '', '']
#	data = read_csv(filename, names = names)
#	class_counts = data.groupby('class').size()
#	print(class_counts)

print('##############################################################################')
print('')
#	2.5:	Statistical summary of all attributes	Statistical Summary(Attribute-x) = Count, Mean, Std.Dev, Min.Value, 25th Percentile, 50th Percentile, 75th Percentile, Max.Value
print('STATISTICAL SUMMARY FOR EACH COLUMN/ATTRIBUTE:')#set_option('precision', 1)
print(dataset.describe())
print('')

#	2.6:	Taking a look at the correlation between all of the numeric attributes
#					CORRELATIONS

# Assess where 'LSTAT' has highest |%|-correlation to an output-variable
#	set_option('precision', 2)
#	print(dataset.corr(method = 'pearson'))

#	##############################################################################
#	##############################################################################
#						OR					
#	##############################################################################
#	##############################################################################

#	PAIRWISE PEARSON CORRELATION:
#		from pandas import read_csv
#		from pandas import set_option
#		filename = '____'
#		names = ['', '', '', '', ''}# Attribute/Column Names
#		data = read_csv(filename, names = names)
#		set_option('display width', 100)
#		set_option('precision', 3)
#		correlations = data.corr(method = 'pearson')
#		print(correlations)

#print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	3.	DATA VISUALIZATION:
##################################
#		3.1:	Univariate/Unimodal Plots

#				i)	Attribute-based HISTOGRAMS
	
#							HISTOGRAM PLOT
#dataset.hist()
#plt.show()

dataset.hist(sharex = False, sharey = False, xlabelsie = 1, ylabelsize = 1)
pyplot.show()

#	from matplotlib import pyplot
#	from pandas import read_csv
#	filename = '____.csv'
#	names = ['', '', '', '']
#	data = read_csv(filename, names = names)
#	data.hist()
#	pyplot.show()

print('##############################################################################')
print('')

#				ii)	Density-Plots to determine Attribute-Distributions
#						Attribute-based DENSITY-PLOT Distributions

dataset.plot(kind = 'density', subplots = True, layout(8,8), sharex = False, legend = False, fontsize = 1)
pyplot.show()

#	from matplotlib import pyplot
#	from pandas import read_csv
#	filename = '____.csv'
#	names = ['', '', '', '']
#	data = read_csv(filename, names = names)
#	dataset.plot(kind = 'density', subplots = True, layout(3,3), sharex = False, legend = False, fontsize = 1)
#	pyplot.show()

print('##############################################################################')
print('')

#				iii)	BOX & WHISKER PLOTS

dataset.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False, fontsize = 1)
pyplot.show()

#	from matplotlib import pyplot
#	from pandas import read_csv
#	filename = '____.csv'
#	names = ['', '', '', '']
#	data = read_csv(filename, names = names)
#	dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, fontsize = 1)
#	pyplot.show()

print('##############################################################################')
print('')
##plt.show()

#	3.4:	SKEW for UNIVARIATE-DISTRIBUTIONS
# Skew/Attribute
from pandas import read_csv
filename = '___'
names = ['', '', '', '']
data = read_csv(filename, names = names)
skew = data.skew()
print(skew)
print('##############################################################################')
print('')

#		3.2:	Multivariate/Multimodal Plots:		- Intersections between variables

#				i)	SCATTER-PLOT MATRIX

#							- Represents relationship between 2-variables as a 2-Dimm-dot
#							- A series/sequence of scatter-plots for multiple variable-pairs = Scatter-Plot Matrix
#	from matplotlib import pyplot
#	from pandas import read_csv
#	import numpy
#	filename = '___.csv'
#	names = ['', '', '', '']
#	data = read_csv(filename, names = names)
#	scatter_matrix(dataset)
#	pyplot.show()


scatter_matrix(dataset)
pyplot.show()
#print('##############################################################################')
#print('')


#################################################################################################
#################################################################################################
#################################################################################################

###	4.	EVALUATING ALGORITHMS:
#####################################

#		4.1:	Isolate VALIDATION/TESTING-Set

#				a)	Create VALIDATION/TESTING-Set

# SLIT-OUT (Validation / Testing set)
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)

#		4.2:	Configure TEST-HARNESS to use K(10)-FOLD CROSS-VALIDATION on ML-Models

#				[a)	Build ML-Models]  >>  [b)	Build 5 ML-Models: Predicting species from Flower-Measurements/Attributes]  >>  [c)	Select best ML-Model]

# SPOT-CHECK ML-Models/Algorithms
models = []
models.append(( ' LR 		' , LogisticRegression()))
models.append(( ' LDA 		' , LinearDiscriminantAnalysis()))
models.append(( ' KNN 		' , KNeighborsClassifier()))
models.append(( ' CART 		' , DecisionTreeClassifier()))
models.append(( ' NB 		' , GaussianNB()))
models.append(( ' SVM 		' , SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring= ' accuracy ' )
results.append(cv_results)
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

# output results to evaluate and select best ML-Model/Algorithm
print(msg)
print('##############################################################################')
print('')

#		5:	COMPARE ALGORITHMS:
##########################################

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle( ' Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#		5:	MAKE PREDICTIONS:
########################################

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print('##############################################################################')
print('')
