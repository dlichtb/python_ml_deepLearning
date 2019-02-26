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
import numpy
#from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
#import pandas
#from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
#from sklearn import model_selection
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC

#	1.2:	Load Dataset
#filename = ('____')
url = ''
#names = ['', '', '', '', '']# Other than CLASS-Attribute (last column) the variables do not have meaningful names, therefore no need to specify column-names
dataset = read_csv(url, header = None)
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']#	### Specifies Column-Names
#dataset = pandas.read_csv(url, names=names)
print('##############################################################################')
print('')

#################################################################################################
#################################################################################################
#################################################################################################

#	b.:	RESCALING DATA:
##############################

## Rescaling data between 0...1
#from pandas import read_csv
#from numpy import set_printoptions
#from sklearn.preprocessing import MinMaxScaler
#filename = '____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into INPUT/OUTPUT components
##
#X = array[:, 0 : 8]
#Y = array[:, 8]
#scaler = MinMaxScaler(feature_range = (0, 1))
#rescaledX = scaler.fit_transform(X)
##
## Summarize transformed data
##
#set_printoptions(precision = 3)
#print(rescaledX[0:5, :])
#print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	c.:	STANDARDIZING DATA:
##################################
## Standardize data (0 mean, 1 stdev)
#from sklearn.preprocessing import StandardScaler
#from pandas import read_csv
#from numpy import set_printoptions
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe - read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into Input/Output components
##
#X = array[:, 0:8]
#Y = array[:, 8]
#scaler = StandardScaler().fit(X)
#rescaledX = scaler.transform(X)
##
## Summarize transformed data
##
#set_printoptions(precision = 2)
#print(rescaledX[0:5,:])
#print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	d.:	NORMALIZING DATA:
################################
## NORMALIZE data (length of 1)
#from sklearn.preprocessing import Normalizer
#from pandas import read_csv
#from numpy import set_printoptions
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into Input/Output components
##
#X = array[:,0:8]
#Y = array[:, 8]
#scaler = Normalizer.fit(X)
#normalizedX = scaler.transform(X)
##
## Summarize transformed data
##
#set_printoptions(precision = 3)
#print(normalizedX[0:5,:])
#print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	e.:	BINARIZING DATA:
################################
## BINARIZING
#from sklearn.preprocessing import Binarizer
#from pandas import read_csv
#from numpy import set_printoptions
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into Input/Output components
##
#X = array[:,0:8]
#Y = array[:, 8]
#binarizer = Binarizer(threshold = 0.0).fit(X)
#binaryX = binarizer.transform(X)
##
## Summarize transformed data
##
#set_printoptions(precision = 3)
#print(binaryX[0:5,:])
#print('##############################################################################')
#print('')

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
#	2.4:	Breakdown of the data by the class variable:	Class Distribution
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
#print('STATISTICAL SUMMARY FOR EACH COLUMN/ATTRIBUTE:')
#set_option('precision', 1)
#print(dataset.describe())
#print('')

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

###	2.a:	FEATURE SELECTION:
#################################

#					1. UNIVARIATE SELECTION:
###############################################################

# FEATURE EXTRACTION with Univariate-Statistical-Test:	CHI-Squared for Classification
#
#from pandas import read_csv
#from numpy import set_printoptions
#from sklearn.feature_selection import SelectKBest
##
## CHI^2 TEST
##
#from sklearn.feature_selection import chi2
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into Input/Output components
##
#X = array[:,0:8]
#Y = array[:, 8]
##
## FEATURE EXTRACTION
##			- Using chi^2-test to test for non-Negative features/attributes/variables to select 4 best
#test = SelectKBest(score_func = chi2, k = 4)
#fit = test.fit(X, Y)
##
## Summarize scores
##
#set_printoptions(precision = 3)
#print(fit.scores_)
#features = fit.transform(X)
##
## Summarize Selected Features
##
#print(features[0:5,:])
#print('##############################################################################')
#print('')

#					2. RECURSIVE FEATURE ELIMINATION (RFE):
##############################################################################
##
## FEATURE EXTRACTION with RFE
##				- Use of RFE-class with LogisticRegression algorithm to select 3 best features
#from pandas import read_csv
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
#
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into Input/Output components
##
#X = array[:,0:8]
#Y = array[:, 8]
##
## FEATURE EXTRACTION
##
#model = LogisticRegression()
#rfe = RFE(model, 3)
#fit = rfe.fit(X, Y)
#
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
#print("Feature Ranking: %s") % fit.ranking_
##print('##############################################################################')
#print('')

#					3. PRINCIPLE COMPONENT ANALYSIS (PCA):
#############################################################################

# FEATURE EXTRACTION with PCA
#
#from pandas import read_csv
#from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
#
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into Input/Output components
##
#X = array[:,0:8]
#Y = array[:, 8]
##
## FEATURE EXTRACTION
##
#pca = PCA(n_components = 3)
#fit = pca.fit(X)
##
## SUMMARIZE COMPONENTS
##
#print("Explained Variance: %s") % fit.explained_variance_ratio_
#print(fit.components_)
##print('##############################################################################')
#print('')

#					4. FEATURE IMPORTANCE:
#############################################################
##
## FEATURE IMPORTANCE with ExtraTreesClassifier
##
#from pandas import read_csv
#from sklearn.decomposition import PCA
#from sklearn.ensemble import ExtraTreesClassifier
#
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
##
## Separate array into Input/Output components
##
#X = array[:,0:8]
#Y = array[:, 8]
##
## FEATURE EXTRACTION
##
#model = ExtraTreesClassifier()
#model.fit(X, Y)
#print(model.features_importances_)

##print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	3a.	EVALUATING ALGORITHMS: SPLIT-OUT (TRAINING/TESTING Set)
#####################################

##			i)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for LOGISTIC-REGRESSION
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = LogisticRegression()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			ii)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for LINEAR-REGRESSION
##
## LinearRegression() does not exist
##
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = LogisticRegression()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			iii)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for LINEAR DISCRIMINATE ANALYSIS (LDA)
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = LinearDiscriminantAnalysis()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			iv-CLASSIFIER)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for K-NEAREST NEIGHBOR (KNN)
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import KNeighborsClassifier
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = KNeighborsClassifier()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			iv-REGRESSOR)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for K-NEAREST NEIGHBOR (KNN)
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import KNeighborsRegressor
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = KNeighborsRegressor()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			v-CLASSIFIER)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for CLASSIFICATION & REGRESSION TREES (CART)
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import DecisionTreeClassifier
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = DecisionTreeClassifier()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			v-REGRESSOR)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for CLASSIFICATION & REGRESSION TREES (CART)
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import DecisionTreeRegressor
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = DecisionTreeRegressor()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			vi)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for GAUSSIAN NAIVE BAYES (NB)
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import GaussianNB
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = GaussianNB()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			vii)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for SUPPORT VECTOR MACHINES (SVM)
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import SVC
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = SVC()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			viii)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for LASSO
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import Lasso
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = Lasso()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

##			ix)	Split-Out TRAINING/TESTING-Set, then TEST/EVALUATE ACCURACY for ELASTICNET
# Evaluate using a train and a test set
#from pandas import read_csv
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import ElasticNet
#filename = '_____.csv'
#names = ['', '', '', '']
#dataframe = read_csv(filename, names = names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
#random_state=seed)
#model = ElasticNet()
#model.fit(X_train, Y_train)
#result = model.score(X_test, Y_test)
#print("Accuracy: %.3f%%") % (result*100.0)

##print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	3b.	EVALUATING ALGORITHMS: CROSS VALIDATION
#####################################

################################################################
###############		K-FOLD		########################
################################################################

##			i)	K-FOLD CROSS VALIDATION with LOGISTIC REGRESSION
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = LogisticRegression()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			ii)	K-FOLD CROSS VALIDATION with LINEAR REGRESSION
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = LinearRegression()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iii)	K-FOLD CROSS VALIDATION with LINEAR DISCRIMINANT ANALYSIS
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = LinearDiscriminantAnalysis()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iv-CLASSIFIER)	K-FOLD CROSS VALIDATION with KNN-CLASSIFIER
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsClassifier
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = KNeighborsClassifier()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iv-REGRESSOR)	K-FOLD CROSS VALIDATION with KNN-REGRESSOR
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsRegressor
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = KNeighborsRegressor()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			v-CLASSIFIER)	K-FOLD CROSS VALIDATION with CART-CLASSIFIER
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeClassifier
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = DecisionTreeClassifier()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			v-REGRESSOR)	K-FOLD CROSS VALIDATION with CART-REGRESSOR
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeRegressor
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = DecisionTreeRegressor()
#results = cross_val_score(model, X, Y, cv=kfold)
#3print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			vi)	K-FOLD CROSS VALIDATION with GAUSSIAN NAIVE BAYES (NB)
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import GaussianNB
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = GaussianNB()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			vii)	K-FOLD CROSS VALIDATION with SUPPORT VECTOR MACHINES (SVM)
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import SVC
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = SVC()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			viii)	K-FOLD CROSS VALIDATION with LASSO
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import Lasso
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = Lasso()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			ix)	K-FOLD CROSS VALIDATION with ELASTICNET
# Evaluate using Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import ElasticNet
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#seed = 7
#kfold = KFold(n_splits=num_folds, random_state=seed)
#model = ElasticNet()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

########################################################################
###############		LEAVE-ONE-OUT		########################
########################################################################

##			i)	LEAVE-ONE-OUT CROSS VALIDATION with LOGISTIC REGRESSION

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = LogisticRegression()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			ii)	LEAVE-ONE-OUT CROSS VALIDATION with LINEAR REGRESSION

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = LinearRegression()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iii)	LEAVE-ONE-OUT CROSS VALIDATION with LINEAR DISCRIMINANT ALGORITHM
# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = LinearDiscriminantAnalysis()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iv-CLASSIFIER)	LEAVE-ONE-OUT CROSS VALIDATION with KNN

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsClassifier
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = KNeighborsClassifier()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iv-REGRESSOR)	LEAVE-ONE-OUT CROSS VALIDATION with KNN

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsClassifier
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = KNeighborsClassifier()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			v-CLASSIFIER)	LEAVE-ONE-OUT CROSS VALIDATION with CART

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeClassifier
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = DecisionTreeClassifier()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			v-REGRESSOR)	LEAVE-ONE-OUT CROSS VALIDATION with CART

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeRegressor
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = DecisionTreeRegressor()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			vi)	LEAVE-ONE-OUT CROSS VALIDATION with GAUSSIAN NAIVE BAYES (NB)

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import GaussianNB
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = GaussianNB()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			vii)	LEAVE-ONE-OUT CROSS VALIDATION with SVM

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import SVC
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = SVC()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			viii)	LEAVE-ONE-OUT CROSS VALIDATION with LASSO

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import Lasso
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = Lasso()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			ix)	LEAVE-ONE-OUT CROSS VALIDATION with ELASTICNET

# Evaluate using Leave One Out Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import ElasticNet
#filename = '____.csv '
#names = [ ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#num_folds = 10
#loocv = LeaveOneOut()
#model = ElasticNet()
#results = cross_val_score(model, X, Y, cv=loocv)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	3c.	EVALUATING ALGORITHMS: REPEATED RANDOM TEST-TRAIN SPLITS
#####################################

##			i)	REPEATED RANDOM TEST-TRAIN SPLITS with LOGISTIC REGRESSION

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = LogisticRegression()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			ii)	REPEATED RANDOM TEST-TRAIN SPLITS with LINEAR REGRESSION

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = LinearRegression()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iii)	REPEATED RANDOM TEST-TRAIN SPLITS with LINEAR DISCRIMINANT ANALYSIS

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = LinearDiscriminantAnalysis()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iv-CLASSIFIER)	REPEATED RANDOM TEST-TRAIN SPLITS with KNN

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsClassifier
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = KNeighborsClassifier()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			iv-REGRESSOR)	REPEATED RANDOM TEST-TRAIN SPLITS with KNN

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsRegressor
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = KNeighborsRegressor()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			v-CLASSIFIER)	REPEATED RANDOM TEST-TRAIN SPLITS with CART

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeClassifier
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = KNeighborsClassifier()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			v-REGRESSOR)	REPEATED RANDOM TEST-TRAIN SPLITS with CART

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeRegressor
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = DecisionTreeRegressor()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			vi)	REPEATED RANDOM TEST-TRAIN SPLITS with GAUSSIAN NB

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import GaussianNB
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = GaussianNB()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			vii)	REPEATED RANDOM TEST-TRAIN SPLITS with SVM

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import SVC
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = SVC()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			viii)	REPEATED RANDOM TEST-TRAIN SPLITS with LASSO

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import Lasso
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = Lasso()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

##			ix)	REPEATED RANDOM TEST-TRAIN SPLITS with ELASTICNET

# Evaluate using Shuffle Split Cross Validation
#from pandas import read_csv
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import ElasticNet
#filename = ' ____.csv '
#names = [ ' ', '', '', '' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#n_splits = 10
#test_size = 0.33
#seed = 7
#kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
#model = ElasticNet()
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

##print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	3d.	EVALUATING ALGORITHMS: PERFORMANCE METRICS
#####################################
#					################################################################################
#					################################################################################
#					###############		CLASSIFICATION METRICS		########################
#					################################################################################
#					################################################################################

################################################################################
###############		CLASSIFICATION ACCURACY		########################
################################################################################

#				###	i) CLASSIFICATION ACCURACY:	LOGISTIC REGRESSION (LR)
#					   -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = LogisticRegression()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ii) CLASSIFICATION ACCURACY:	LINEAR REGRESSION (LR)
#					    -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = LinearRegression()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iii) CLASSIFICATION ACCURACY:	LINEAR DISCRIMINANT ANALYSIS (LDA)
#					     -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = LinearDiscriminantAnalysis()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) CLASSIFICATION ACCURACY:	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER
#					               -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsClassifier
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = KNeighborsClassifier()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) CLASSIFICATION ACCURACY:	K-NEAREST NEIGHBOR (KNN) - REGRESSOR
#					   	      -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsRegressor
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = KNeighborsRegressor()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) CLASSIFICATION ACCURACY:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER
#					   	      -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeClassifier
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = DecisionTreeClassifier()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) CLASSIFICATION ACCURACY:	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR
#					   	     -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeRegressor
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = DecisionTreeRegressor()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vi) CLASSIFICATION ACCURACY:	GAUSSIAN NAIVE BAYES (NB)
#					    -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import GaussianNB
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = GaussianNB()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vii) CLASSIFICATION ACCURACY:	SUPPORT VECTOR MACHINE (SVM)
#					     -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import SVC
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = SVC()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	viii) CLASSIFICATION ACCURACY:	LASSO
#					      -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import Lasso
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = Lasso()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ix) CLASSIFICATION ACCURACY:	ELASTICNET
#					    -----------------------
## Cross Validation Classification Accuracy
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import ElasticNet
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = ElasticNet()
#scoring = ' accuracy '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

################################################################################
###############		LOGARITHMIC LOSS		########################
################################################################################

#				###	i) LOGARITHMIC LOSS:	LOGISTIC REGRESSION (LR)
#					   ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LogisticRegression
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = LogisticRegression()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ii) LOGARITHMIC LOSS:	LINEAR REGRESSION (LR)
#					    ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = LinearRegression()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iii) LOGARITHMIC LOSS:	LINEAR DISCRIMINANT ANALYSIS (LDA)
#					     ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = LinearDiscriminantAnalysis()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) LOGARITHMIC LOSS:	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER
#					               ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsClassifier
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = KNeighborsClassifier()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) LOGARITHMIC LOSS:	K-NEAREST NEIGHBOR (KNN) - REGRESSOR
#					   	      ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import KNeighborsRegressor
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = KNeighborsRegressor()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) LOGARITHMIC LOSS:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER
#					   	      ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeClassifier
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = DecisionTreeClassifier()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) LOGARITHMIC LOSS:	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR
#					   	     ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import DecisionTreeRegressor
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = DecisionTreeRegressor()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vi) LOGARITHMIC LOSS:	GAUSSIAN NAIVE BAYES (NB)
#					    ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import GaussianNB
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = GaussianNB()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vii) LOGARITHMIC LOSS:	SUPPORT VECTOR MACHINE (SVM)
#					     ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import SVC
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = SVC()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	viii) LOGARITHMIC LOSS:	LASSO
#					      ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import Lasso
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = Lasso()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ix) LOGARITHMIC LOSS:	ELASTICNET
#					    ----------------
## Cross Validation Classification Logloss
#from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import ElasticNet
#filename = '_____.csv'
#names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
#dataframe = read_csv(filename, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
#kfold = KFold(n_splits=10, random_state=7)
#model = ElasticNet()
#scoring = ' neg_log_loss '
#results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

################################################################################
###############		AREA UNDER ROC CURVE		########################
################################################################################

#				###	i) AREA UNDER ROC CURVE:	Logistic Regression (LR)

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ii) AREA UNDER ROC CURVE:	Linear Regression (LR)

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iii) AREA UNDER ROC CURVE:	Linear Discriminant Analysis (LDA)

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LinearDiscriminantAnalysis()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) AREA UNDER ROC CURVE:	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) AREA UNDER ROC CURVE:	K-NEAREST NEIGHBOR (KNN) - REGRESSOR

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) AREA UNDER ROC CURVE:	Classification & Regression Trees (CART) - CLASSIFIER

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) AREA UNDER ROC CURVE:	Classification & Regression Trees (CART) - REGRESSOR

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeRegressor()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vi) AREA UNDER ROC CURVE:	Gaussian Naive Bayes (NB)

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vii) AREA UNDER ROC CURVE:	Support Vector Machine (SVM)

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	viii) AREA UNDER ROC CURVE:	Lasso

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = Lasso()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ix) AREA UNDER ROC CURVE:	ElasticNet

# Cross Validation Classification ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = ElasticNet()
scoring = ' roc_auc '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

################################################################################
###############		CONFUSION MATRIX		########################
################################################################################

#				###	i) CONFUSION MATRIX:	Logistic Regression (LR)

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	i) CONFUSION MATRIX:	Linear Regression (LR)

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LinearRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	iii) CONFUSION MATRIX:	Linear Discriminant Analysis (LDA)

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) CONFUSION MATRIX:	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) CONFUSION MATRIX:	K-NEAREST NEIGHBOR (KNN) - REGRESSOR

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = KNeighborsRegressor()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) CONFUSION MATRIX:	Classification & Regression Trees (CART) - CLASSIFIER

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) CONFUSION MATRIX:	Classification & Regression Trees (CART) - REGRESSOR

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	vi) CONFUSION MATRIX:	Gaussian Naive Bayes (NB)

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import GaussianNB
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = GaussianNB()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	vii) CONFUSION MATRIX:	Support Vector Machine (SVM)

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SVC
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = SVC()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	viii) CONFUSION MATRIX:	Lasso

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = Lasso()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

#				###	ix) CONFUSION MATRIX:	ElasticNet

# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import confusion_matrix
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = ElasticNet()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
##print('##############################################################################')
#print('')

################################################################################
###############		CLASSIFICATION REPORT		########################
################################################################################

#				###	i) CLASSIFICATION REPORT:	Logistic Regression (LR)

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	ii) CLASSIFICATION REPORT:	Linear Regression (LR)

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LinearRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	iii) CLASSIFICATION REPORT:	Linear Discriminant Analysis (LDA)

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) CLASSIFICATION REPORT:	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import KNeighborsClassifier
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) CLASSIFICATION REPORT:	K-NEAREST NEIGHBOR (KNN) - REGRESSOR

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import KNeighborsRegressor
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = KNeighborsRegressor()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) CLASSIFICATION REPORT:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import DecisionTreeClassifier
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) CLASSIFICATION REPORT:	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import DecisionTreeRegressor
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	vi) CLASSIFICATION REPORT:	GAUSSIAN NAIVE BAYES (NB)

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import GaussianNB
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = GaussianNB()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	vii) CLASSIFICATION REPORT:	SUPPORT VECTOR MACHINE (SVM)

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SVC
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = SVC()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	viii) CLASSIFICATION REPORT:	LASSO ()

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = Lasso()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#				###	ix) CLASSIFICATION REPORT:	ElasticNet

# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import classification_report
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = ElasticNet()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
##print('##############################################################################')
#print('')

#					################################################################################
#					################################################################################
#					###############		REGRESSION METRICS		########################
#					################################################################################
#					################################################################################

################################################################################
###############		MEAN ABSOLUTE ERROR	(MAE)	########################
################################################################################

#				###	i) MEAN ABSOLUTE ERROR			(MAE):	LOGISTIC REGRESSION (LR)

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ii) MEAN ABSOLUTE ERROR			(MAE):	LINEAR REGRESSION (LR)

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iii) MEAN ABSOLUTE ERROR		(MAE):	LINEAR DISCRIMINANT ANALYSIS (LDA)

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearDiscriminantAnalysis()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) MEAN ABSOLUTE ERROR		(MAE):	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) MEAN ABSOLUTE ERROR		(MAE):	K-NEAREST NEIGHBOR (KNN) - REGRESSOR

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) MEAN ABSOLUTE ERROR		(MAE):	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) MEAN ABSOLUTE ERROR		(MAE):	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeRegressor()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vi) MEAN ABSOLUTE ERROR			(MAE):	GAUSSIAN NAIVE BAYES

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vii) MEAN ABSOLUTE ERROR	(MAE):	SUPPORT VECTOR MACHINE (SVM)

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	viii) MEAN ABSOLUTE ERROR	(MAE):	Lasso

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = Lasso()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ix) MEAN ABSOLUTE ERROR		(MAE):	ElasticNet

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = ElasticNet()
scoring = ' neg_mean_absolute_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

################################################################################
###############		MEAN SQUARED ERROR		########################
################################################################################

#				###	i) MEAN ABSOLUTE ERROR			(MAE):	LOGISTIC REGRESSION (LR)

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ii) MEAN ABSOLUTE ERROR			(MAE):	LINEAR REGRESSION (LR)

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iii) MEAN ABSOLUTE ERROR		(MAE):	LINEAR DISCRIMINANT ANALYSIS (LDA)

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LinearDiscriminantAnalysis()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) MEAN ABSOLUTE ERROR	(MAE):	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) MEAN ABSOLUTE ERROR	(MAE):	K-NEAREST NEIGHBOR (KNN) - REGRESSOR

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) MEAN ABSOLUTE ERROR	(MAE):	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) MEAN ABSOLUTE ERROR	(MAE):	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeRegressor()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vi) MEAN ABSOLUTE ERROR			(MAE):	GAUSSIAN NAIVE BAYES (NB)

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vii) MEAN ABSOLUTE ERROR		(MAE):	SUPPORT VECTOR MACHINES (SVM)

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	viii) MEAN ABSOLUTE ERROR		(MAE):	LASSO

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = Lasso()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ix) MEAN ABSOLUTE ERROR			(MAE):	ELASTICNET

# Cross Validation Regression MSE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = ElasticNet()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

################################################################################
###############			R^2			########################
################################################################################

#				###	i) R^2:		LOGISTIC REGRESSION (LR)

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ii) R^2:	LINEAR REGRESSION (LR)

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iii) R^2:	LINEAR DISCRIMINANT ANALYSIS (LDA)

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearDiscriminantAnalysis()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-CLASSIFIER) R^2:	K-NEAREST NEIGHBOR (KNN) - CLASSIFIER

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	iv-REGRESSOR) R^2:	K-NEAREST NEIGHBOR (KNN) - REGRESSOR

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-CLASSIFIER) R^2:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	v-REGRESSOR) R^2:	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeRegressor()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vi) R^2:	GAUSSIAN NAIVE BAYES (NB)

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	vii) R^2:	SUPPORT VECTOR MACHINES (SVM)

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	viii) R^2:	LASSO

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = Lasso()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')

#				###	ix) R^2:	ELASTICNET

# Cross Validation Regression R^2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = ElasticNet()
scoring = ' r2 '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)") % (results.mean(), results.std())
##print('##############################################################################')
#print('')


#################################################################################################
#################################################################################################
#################################################################################################

###	3d.	EVALUATING ALGORITHMS: SPOT-CHECKING ALGORITHMS
#####################################

#					################################################################################
#					################################################################################
#					###############		SPOT-CHECK CLASSIFICATION ALGORITHMS		########
#					################################################################################
#					###############			K-FOLD CROSS-VALIDATION		################
#					################################################################################

################################################################################
###############		LINEAR ML-ALGORITHMS		########################
################################################################################

#				###	i) LINEAR ML-ALGORITHMS:			LOGISTIC REGRESSION (LR)
# Logistic Regression Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#				###	ii) LINEAR ML-ALGORITHMS:			LINEAR DISCRIMINANT ANALYSIS (LDA)
# Logistic Regression Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

################################################################################
###############		NON-LINEAR ML-ALGORITHMS		################
################################################################################

#				###	i-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR (KNN) - CLASSIFIER
# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#				###	i-REGRESSOR) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR (KNN) - REGRESSOR
# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#				###	ii) NON-LINEAR ML-ALGORITHMS:			GAUSSIAN NAIVE BAYES (NB)
# Gaussian Naive Bayes Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#				###	iii-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER
# CART Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#				###	iii-REGRESSOR) NON-LINEAR ML-ALGORITHMS:	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR
# CART Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeRegressor()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#				###	iv) NON-LINEAR ML-ALGORITHMS:			SUPPORT VECTOR MACHINES (SVM)
# SVM Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


#					################################################################################
#					################################################################################
#					###############		SPOT-CHECK CLASSIFICATION ALGORITHMS		########
#					################################################################################
#					###############		LEAVE-ONE-OUT CROSS-VALIDATION		################
#					################################################################################
#					################################################################################


################################################################################
###############		LINEAR ML-ALGORITHMS		########################
################################################################################

#				###	i) LINEAR ML-ALGORITHMS:			LOGISTIC REGRESSION (LR)
# Logistic Regression Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	ii) LINEAR ML-ALGORITHMS:			LINEAR DISCRIMINANT ANALYSIS (LDA)
# Linear Discriminant Analysis Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

################################################################################
###############		NON-LINEAR ML-ALGORITHMS		################
################################################################################

#				###	i-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR (KNN) - CLASSIFIER
# KNN Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	i-REGRESSOR) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR (KNN) - REGRESSOR
# KNN Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = KNeighborsRegressor()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	ii) NON-LINEAR ML-ALGORITHMS:			GAUSSIAN NAIVE BAYES (NB)
# Gaussian Naive Bayes Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	iii-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER
# CART Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	iii-REGRESSOR) NON-LINEAR ML-ALGORITHMS:	CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR
# CART Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = DecisionTreeRegressor()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	iv) NON-LINEAR ML-ALGORITHMS:			SUPPORT VECTOR MACHINES (SVM)
# SVM Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()
model = SVC()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


#					################################################################################
#					################################################################################
#					###############		SPOT-CHECK REGRESSION ALGORITHMS		########
#					################################################################################
#					###############			K-FOLD CROSS-VALIDATION		################
#					################################################################################
#					################################################################################

################################################################################
###############		LINEAR ML-ALGORITHMS		########################
################################################################################

#				###	i) LINEAR ML-ALGORITHMS:			LINEAR REGRESSION (LR)
# Linear Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#				###	ii) LINEAR ML-ALGORITHMS:			RIDGE REGRESSION (LR)
# Ridge Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = Ridge()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#				###	iii) LINEAR ML-ALGORITHMS:			LASSO LINEAR REGRESSION (LASSO)
# Lasso Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = Lasso()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#				###	iv) LINEAR ML-ALGORITHMS:			ELASTICNET REGRESSION (ELASTICNET)
# ElasticNet Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = ElasticNet()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())


################################################################################
###############		NON-LINEAR ML-ALGORITHMS	########################
################################################################################

#				###	i-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR REGRESSION (LR) - CLASSIFIER
# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#				###	i-REGRESSOR) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR REGRESSION (LR) - REGRESSOR
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#				###	ii-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER
# Decision Tree Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#				###	ii-REGRESSOR) NON-LINEAR ML-ALGORITHMS:		CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR
# Decision Tree Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeRegressor()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

#				###	iii) NON-LINEAR ML-ALGORITHMS:			SUPPORT VECTOR MACHINES (SVM)
# SVM Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = SVR()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())


#					################################################################################
#					################################################################################
#					###############		SPOT-CHECK REGRESSION ALGORITHMS		########
#					################################################################################
#					###############		LEAVE-ONE-OUT CROSS-VALIDATION		################
#					################################################################################

################################################################################
###############		LINEAR ML-ALGORITHMS		########################
################################################################################

#				###	i) LINEAR ML-ALGORITHMS:			LINEAR REGRESSION (LR)
# Linear Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
loocv = LeaveOneOut()
model = LinearRegression()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	ii) LINEAR ML-ALGORITHMS:			RIDGE REGRESSION (LR)
# Ridge Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
loocv = LeaveOneOut()
model = Ridge()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	iii) LINEAR ML-ALGORITHMS:			LASSO LINEAR REGRESSION (LASSO)
# Lasso Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
loocv = LeaveOneOut()
model = Lasso()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	iv) LINEAR ML-ALGORITHMS:			ELASTICNET REGRESSION (ELASTICNET)
# ElasticNet Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
Y = array[:,13]
loocv = LeaveOneOut()
model = ElasticNet()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


################################################################################
###############		NON-LINEAR ML-ALGORITHMS	########################
################################################################################

#				###	i-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR REGRESSION (LR) - CLASSIFIER
# KNN Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
loocv = LeaveOneOut()
model = KNeighborsClassifier()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	i-REGRESSOR) NON-LINEAR ML-ALGORITHMS:		K-NEAREST NEIGHBOR REGRESSION (LR) - REGRESSOR
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
loocv = LeaveOneOut()
model = KNeighborsRegressor()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	ii-CLASSIFIER) NON-LINEAR ML-ALGORITHMS:	CLASSIFICATION & REGRESSION TREES (CART) - CLASSIFIER
# Decision Tree Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
loocv = LeaveOneOut()
model = DecisionTreeClassifier()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	ii-REGRESSOR) NON-LINEAR ML-ALGORITHMS:		CLASSIFICATION & REGRESSION TREES (CART) - REGRESSOR
# Decision Tree Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
loocv = LeaveOneOut()
model = DecisionTreeRegressor()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#				###	iii) NON-LINEAR ML-ALGORITHMS:			SUPPORT VECTOR MACHINES (SVM)
# SVM Regression
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
num_folds = 10
loocv = LeaveOneOut()
model = SVR()
scoring = ' neg_mean_squared_error '
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


#################################################################################################
#################################################################################################
#################################################################################################


###	3e.	COMPARING ALGORITHMS: SPOT-CHECKING ALGORITHMS
####################################

#					################################################################################
#					################################################################################
#					###############			CLASSIFICATION ALGORITHMS		########
#					################################################################################
#					###############			K-FOLD CROSS-VALIDATION		################
#					################################################################################
#					################################################################################

#	Compares the following algorithms:
#						- LOGISTIC REGRESSION			(LR)
#						- LINEAR REGRESSION			(LR)
#						- LINEAR DISCRIMINANT ANALYSIS		(LDA)
#						- K-NEAREST NEIGHBOR			(KNN)
#												- CLASSIFIER
#												- REGRESSOR
#						- CLASSIFICATION & REGRESSION TREES	(CART)
#												- CLASSIFIER
#												- REGRESSOR
#						- GAUSSIAN NAIVE BAYES			(NB)
#						- SUPPORT VECTOR MACHINES		(SVM)
#						- LASSO					(Lasso)
#						- ELASTICNET				(ElasticNet)

# Compare Algorithms
from pandas import read_csv
from matplotlib import pyplot

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# prepare models
models = []
models.append(( ' LR 		' , LogisticRegression()))
models.append(( ' LR 		' , LinearRegression()))
models.append(( ' Lasso 	' , Lasso()))
models.append(( ' ElasticNet 	' , ElasticNet()))
models.append(( ' CART 		' , DecisionTreeClassifier()))
models.append(( ' CART 		' , DecisionTreeRegressor()))
models.append(( ' KNN 		' , KNeighborsClassifier()))
models.append(( ' KNN 		' , KNeighborsRegressor()))
models.append(( ' LDA 		' , LinearDiscriminantAnalysis()))
models.append(( ' NB 		' , GaussianNB()))
models.append(( ' SVM 		' , SVC()))

# evaluate each model in turn
results = []
names = []
scoring = ' accuracy '
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle( ' Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()



#					################################################################################
#					################################################################################
#					###############			CLASSIFICATION ALGORITHMS		########	1/2
#					################################################################################
#					###############		LEAVE-ONE-OUT CROSS-VALIDATION		################	2/2
#					################################################################################
#					################################################################################

#	Compares the following algorithms:
#						- LOGISTIC REGRESSION			(LR)
#						- LINEAR REGRESSION			(LR)
#						- LINEAR DISCRIMINANT ANALYSIS		(LDA)
#						- K-NEAREST NEIGHBOR			(KNN)
#												- CLASSIFIER
#												- REGRESSOR
#						- CLASSIFICATION & REGRESSION TREES	(CART)
#												- CLASSIFIER
#												- REGRESSOR
#						- GAUSSIAN NAIVE BAYES			(NB)
#						- SUPPORT VECTOR MACHINES		(SVM)
#						- LASSO					(Lasso)
#						- ELASTICNET				(ElasticNet)

# Compare Algorithms
from pandas import read_csv
from matplotlib import pyplot

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
loocv = LeaveOneOut()

# prepare models
models = []
models.append(( ' LR 		' , LogisticRegression()))
models.append(( ' LR 		' , LinearRegression()))
models.append(( ' Lasso 	' , Lasso()))
models.append(( ' ElasticNet 	' , ElasticNet()))
models.append(( ' CART 		' , DecisionTreeClassifier()))
models.append(( ' CART 		' , DecisionTreeRegressor()))
models.append(( ' KNN 		' , KNeighborsClassifier()))
models.append(( ' KNN 		' , KNeighborsRegressor()))
models.append(( ' LDA 		' , LinearDiscriminantAnalysis()))
models.append(( ' NB 		' , GaussianNB()))
models.append(( ' SVM 		' , SVC()))

# evaluate each model in turn
results = []
names = []
scoring = ' accuracy '
for name, model in models:
	cv_results = cross_val_score(model, X, Y, cv=loocv)
	print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, cv_results.std()*100.0)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle( ' Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

##########################################################################################################################
##########################################################################################################################


#					################################################################################
#					################################################################################
#					###############			REGRESSION ALGORITHMS			########	2/2
#					################################################################################
#					###############			K-FOLD CROSS-VALIDATION		################	1/2
#					################################################################################
#					################################################################################

#	Compares the following algorithms:
#						- LOGISTIC REGRESSION			(LR)
#						- LINEAR REGRESSION			(LR)
#						- LINEAR DISCRIMINANT ANALYSIS		(LDA)
#						- K-NEAREST NEIGHBOR			(KNN)
#												- CLASSIFIER
#												- REGRESSOR
#						- CLASSIFICATION & REGRESSION TREES	(CART)
#												- CLASSIFIER
#												- REGRESSOR
#						- GAUSSIAN NAIVE BAYES			(NB)
#						- SUPPORT VECTOR MACHINES		(SVM)
#						- LASSO					(Lasso)
#						- ELASTICNET				(ElasticNet)




#					################################################################################
#					################################################################################
#					###############			REGRESSION ALGORITHMS			########	2/2
#					################################################################################
#					###############		LEAVE-ONE-OUT CROSS-VALIDATION		################	2/2
#					################################################################################
#					################################################################################

#	Compares the following algorithms:
#						- LOGISTIC REGRESSION			(LR)
#						- LINEAR REGRESSION			(LR)
#						- LINEAR DISCRIMINANT ANALYSIS		(LDA)
#						- K-NEAREST NEIGHBOR			(KNN)
#												- CLASSIFIER
#												- REGRESSOR
#						- CLASSIFICATION & REGRESSION TREES	(CART)
#												- CLASSIFIER
#												- REGRESSOR
#						- GAUSSIAN NAIVE BAYES			(NB)
#						- SUPPORT VECTOR MACHINES		(SVM)
#						- LASSO					(Lasso)
#						- ELASTICNET				(ElasticNet)


################################################################################
################################################################################
################################################################################

###	4.	IMPROVE PERFORMANCE/ACCURACY:
############################################

#			1.)	ENSEMBLES:
#				---------

###################################################################################################################################
###							BAGGING ALGORITHMS							###
###################################################################################################################################

#					################################################################################
#					################################################################################
#					#######				BAGGED DECISION TREES			  ######	1/3
#					################################################################################
#					################################################################################
#					###############			K(10)-FOLD CROSS-VALIDATION		########		1/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	BAGGING (BOOTSTRAP AGGREGATION):	LINEAR REGRESSION		(LR)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = LinearRegression()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						b)	BAGGING (BOOTSTRAP AGGREGATION):	LOGISTIC REGRESSION		(LR)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = LogisticRegression()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						c)	BAGGING (BOOTSTRAP AGGREGATION):	LINEAR DISCRIMINANT ANALYSIS	(LDA)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = LinearDiscriminantAnalysis()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						d)-CLASSIFIER	BAGGING (BOOTSTRAP AGGREGATION):	K-NEAREST NEIGHBOR		(KNN) - CLASSIFIER
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = KNeighborsClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						d)-REGRESSOR	BAGGING (BOOTSTRAP AGGREGATION):	K-NEAREST NEIGHBOR		(KNN) - REGRESSOR
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = KNeighborsRegressor()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						e)-CLASSIFIER	BAGGING (BOOTSTRAP AGGREGATION):	CLASSIFICATION & REGRESSION TREE		(CART) - CLASSIFIER
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						e)-REGRESSOR	BAGGING (BOOTSTRAP AGGREGATION):	CLASSIFICATION & REGRESSION TREE		(CART) - REGRESSOR
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeRegressor()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						f)	BAGGING (BOOTSTRAP AGGREGATION):	GAUSSIAN NAIVE BAYES	(NB)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = GaussianNB()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						g)	BAGGING (BOOTSTRAP AGGREGATION):	SUPPORT VECTOR MACHINES	(SVM)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = SVC()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						h)	BAGGING (BOOTSTRAP AGGREGATION):	LASSO			(Lasso)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = Lasso()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						i)	BAGGING (BOOTSTRAP AGGREGATION):	ELASTICNET		(ElasticNet)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = ElasticNet()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#############################################################################################################################
#############################################################################################################################

#					################################################################################
#					################################################################################
#					#######				BAGGED DECISION TREES			  ######	1/3
#					################################################################################
#					################################################################################
#					###############			LEAVE-ONE-OUT CROSS-VALIDATION		########		2/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	BAGGING (BOOTSTRAP AGGREGATION):	LINEAR REGRESSION		(LR)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = LinearRegression()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						b)	BAGGING (BOOTSTRAP AGGREGATION):	LOGISTIC REGRESSION		(LR)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = LogisticRegression()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						c)	BAGGING (BOOTSTRAP AGGREGATION):	LINEAR DISCRIMINANT ANALYSIS	(LDA)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = LinearDiscriminantAnalysis()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						d)-CLASSIFIER	BAGGING (BOOTSTRAP AGGREGATION):	K-NEAREST NEIGHBOR		(KNN) - CLASSIFIER
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = KNeighborsClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						d)-REGRESSOR	BAGGING (BOOTSTRAP AGGREGATION):	K-NEAREST NEIGHBOR		(KNN) - REGRESSOR
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = KNeighborsRegressor()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						e)-CLASSIFIER	BAGGING (BOOTSTRAP AGGREGATION):	CLASSIFICATION & REGRESSION TREE		(CART) - CLASSIFIER
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						e)-REGRESSOR	BAGGING (BOOTSTRAP AGGREGATION):	CLASSIFICATION & REGRESSION TREE		(CART) - REGRESSOR
#								-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = DecisionTreeRegressor()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						f)	BAGGING (BOOTSTRAP AGGREGATION):	GAUSSIAN NAIVE BAYES	(NB)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = GaussianNB()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						g)	BAGGING (BOOTSTRAP AGGREGATION):	SUPPORT VECTOR MACHINES	(SVM)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = SVC()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						h)	BAGGING (BOOTSTRAP AGGREGATION):	LASSO			(Lasso)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = Lasso()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						i)	BAGGING (BOOTSTRAP AGGREGATION):	ELASTICNET		(ElasticNet)
#							-------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = ElasticNet()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#############################################################################################################################
#############################################################################################################################

#					################################################################################
#					################################################################################
#					#######				RANDOM FOREST				  ######	2/3
#					################################################################################
#					################################################################################
#					###############			K(10)-FOLD CROSS-VALIDATION		########		1/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	RANDOM FOREST:	LINEAR REGRESSION		(LR)
#							---------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=seed)
cart = LinearRegression()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						b)	RANDOM FOREST:	LOGISTIC REGRESSION		(LR)
#							-----------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = LogisticRegression()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						c)	RANDOM FOREST:	LINEAR DISCRIMINANT ANALYSIS	(LDA)
#							--------------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = LinearDiscriminantAnalysis()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
#						d)-CLASSIFIER	RANDOM FOREST: 	K-NEAREST NEIGHBOR		(KNN) - CLASSIFIER
#								----------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = KNeighborsClassifier()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						d)-REGRESSOR	RANDOM FOREST:	K-NEAREST NEIGHBOR		(KNN) - REGRESSOR
#								----------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = KNeighborsRegressor()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						e)-CLASSIFIER	RANDOM FOREST:	CLASSIFICATION & REGRESSION TREE		(CART) - CLASSIFIER
#								------------------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						e)-REGRESSOR	RANDOM FOREST:	CLASSIFICATION & REGRESSION TREE		(CART) - REGRESSOR
#								------------------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeRegressor()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						f)	RANDOM FOREST:	GAUSSIAN NAIVE BAYES	(NB)
#							------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = GaussianNB()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						g)	RANDOM FOREST:	SUPPORT VECTOR MACHINES	(SVM)
#							---------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = SVC()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						h)	RANDOM FOREST:	LASSO			(Lasso)
#							---------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = Lasso()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#						i)	RANDOM FOREST:	ELASTICNET		(ElasticNet)
#							--------------------------
# Random Forest Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = ElasticNet()
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#############################################################################################################################
#############################################################################################################################

#					################################################################################
#					################################################################################
#					#######				RANDOM FOREST				  ######	2/3
#					################################################################################
#					################################################################################
#					###############			LEAVE-ONE-OUT CROSS-VALIDATION		########		2/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	RANDOM FOREST:	LINEAR REGRESSION		(LR)
#							---------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
cart = LinearRegression()
num_trees = 100
max_features = 3
loocv = LeaveOneOut()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						b)	RANDOM FOREST:	LOGISTIC REGRESSION		(LR)
#							-----------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
cart = LogisticRegression()
num_trees = 100
max_features = 3
loocv = LeaveOneOut()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


#						c)	RANDOM FOREST:	LINEAR DISCRIMINANT ANALYSIS	(LDA)
#							--------------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearDiscriminantAnalysis
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
cart = LinearDiscriminantAnalysis()
num_trees = 100
max_features = 3
loocv = LeaveOneOut()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


#						d)-CLASSIFIER	RANDOM FOREST:	K-NEAREST NEIGHBOR		(KNN) - CLASSIFIER
#								----------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
cart = KNeighborsClassifier()
num_trees = 100
max_features = 3
loocv = LeaveOneOut()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


#						d)-REGRESSOR	RANDOM FOREST:	K-NEAREST NEIGHBOR		(KNN) - REGRESSOR
#								----------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = KNeighborsRegressor()
num_trees = 100
model = RandomForestClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						e)-CLASSIFIER	RANDOM FOREST:	CLASSIFICATION & REGRESSION TREE		(CART) - CLASSIFIER
#								------------------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = DecisionTreeClassifier()
num_trees = 100
model = RandomForestClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						e)-REGRESSOR	RANDOM FOREST:	CLASSIFICATION & REGRESSION TREE		(CART) - REGRESSOR
#								------------------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = DecisionTreeRegressor()
num_trees = 100
model = RandomForestClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						f)	RANDOM FOREST:	GAUSSIAN NAIVE BAYES	(NB)
#							------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = GaussianNB()
num_trees = 100
model = RandomForestClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						g)	RANDOM FOREST:	SUPPORT VECTOR MACHINES	(SVM)
#							---------------------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = SVC()
num_trees = 100
model = RandomForestClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						h)	RANDOM FOREST:	LASSO			(Lasso)
#							---------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = Lasso()
num_trees = 100
model = RandomForestClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#						i)	RANDOM FOREST:	ELASTICNET		(ElasticNet)
#							--------------------------
# Bagged Decision Trees for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
cart = ElasticNet()
num_trees = 100
model = RandomForestClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

#############################################################################################################################
#############################################################################################################################

#					################################################################################
#					################################################################################
#					#######				EXTRA TREES				  ######	3/3
#					################################################################################
#					################################################################################
#					###############			K(10)-FOLD CROSS-VALIDATION		########		1/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	EXTRA TREES:	(LR) | (LR) | (LDA) | (KNN-CLASSIFIER) | (CART-CLASSIFIER) | (NB) | (LASS0) | (SVM) | (ELASTICNET)
#							-----------
# Extra Trees Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 100
max_features = 7
kfold = KFold(n_splits=10, random_state=7)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#############################################################################################################################
#############################################################################################################################

#					################################################################################
#					################################################################################
#					#######				EXTRA TREES				  ######	3/3
#					################################################################################
#					################################################################################
#					###############			LEAVE-ONE-OUT CROSS-VALIDATION		########		2/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	EXTRA TREES:	(LR) | (LR) | (LDA) | (KNN-CLASSIFIER) | (CART-CLASSIFIER) | (NB) | (LASS0) | (SVM) | (ELASTICNET)
#							-----------
# Extra Trees Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut()
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
num_trees = 100
loocv = LeaveOneOut()
model = ExtraTreesClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: % 3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


########################################################################################################################################################################################################
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
########################################################################################################################################################################################################

#		[1/3, 1/2, 1/5]
#		[2/3, 1/2, 1/5]
#		[3/3, 1/2, 1/5]
#		[1/3, 2/2, 1/5]
#		[2/3, 2/2, 1/5]
#		[3/3, 2/2, 1/5]
#		[1/3, 1/2, 2/5]
#		[2/3, 1/2, 2/5]
#		[3/3, 1/2, 2/5]
#		[1/3, 2/2, 2/5]
#		[2/3, 2/2, 2/5]
#		[3/3, 2/2, 2/5]
#		[1/3, 1/2, 3/5]
#		[2/3, 1/2, 3/5]
#		[3/3, 1/2, 3/5]
#		[1/3, 2/2, 3/5]
#		[2/3, 2/2, 3/5]
#		[3/3, 2/2, 3/5]
#		[1/3, 1/2, 4/5]
#		[2/3, 1/2, 4/5]
#		[3/3, 1/2, 4/5]
#		[1/3, 2/2, 4/5]
#		[2/3, 2/2, 4/5]
#		[3/3, 2/2, 4/5]
#		[1/3, 1/2, 5/5]
#		[2/3, 1/2, 5/5]
#		[3/3, 1/2, 5/5]
#		[1/3, 2/2, 5/5]
#		[2/3, 2/2, 5/5]
#		[3/3, 2/2, 5/5]


########################################################################################################################################################################################################
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
########################################################################################################################################################################################################


###################################################################################################################################
###							BOOSTING ALGORITHMS							###
###################################################################################################################################

#					################################################################################
#					################################################################################
#					#######					ADABOOST			  ######	1/2
#					################################################################################
#					################################################################################
#					###############			K(10)-FOLD CROSS-VALIDATION		########		1/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	BAGGING (BOOTSTRAP AGGREGATION):	(LR) | (LR) | (LDA) | (KNN-CLASSIFIER) | (CART-CLASSIFIER) | (NB) | (LASS0) | (SVM) | (ELASTICNET)
#							-------------------------------

# AdaBoost Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
seed=7
kfold = KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#					################################################################################
#					################################################################################
#					#######					ADABOOST			  ######	1/2
#					################################################################################
#					################################################################################
#					###############			LEAVE-ONE-OUT CROSS-VALIDATION		########		2/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	BAGGING (BOOTSTRAP AGGREGATION):	(LR) | (LR) | (LDA) | (KNN-CLASSIFIER) | (CART-CLASSIFIER) | (NB) | (LASS0) | (SVM) | (ELASTICNET)

# AdaBoost Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
seed=7
num_folds = 10
loocv = LeaveOneOut()
#model = ____()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

###################################################################################################################################
###################################################################################################################################

#					################################################################################
#					################################################################################
#					#######			STOCHASTIC GRADIENT BOOSTING			  ######	2/2
#					################################################################################
#					################################################################################
#					###############			K(10)-FOLD CROSS-VALIDATION		########		1/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	BAGGING (BOOTSTRAP AGGREGATION):	(LR) | (LR) | (LDA) | (KNN-CLASSIFIER) | (CART-CLASSIFIER) | (NB) | (LASS0) | (SVM) | (ELASTICNET)
#							-------------------------------

# Stochastic Gradient Boosting Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
kfold = KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#					################################################################################
#					################################################################################
#					#######					ADABOOST			  ######	2/2
#					################################################################################
#					################################################################################
#					###############			LEAVE-ONE-OUT CROSS-VALIDATION		########		2/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	BAGGING (BOOTSTRAP AGGREGATION):	(LR) | (LR) | (LDA) | (KNN-CLASSIFIER) | (CART-CLASSIFIER) | (NB) | (LASS0) | (SVM) | (ELASTICNET)

# Stochastic Gradient Boosting Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut()
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
seed=7
num_folds = 10
loocv = LeaveOneOut()
#model = ____()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

########################################################################################################################################################################################################
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
########################################################################################################################################################################################################

#		[1/2, 1/2, 1/5]
#		[2/2, 1/2, 1/5]
#		[1/2, 2/2, 1/5]
#		[2/2, 2/2, 1/5]
#		[1/2, 1/2, 2/5]
#		[2/2, 1/2, 2/5]
#		[1/2, 2/2, 2/5]
#		[2/2, 2/2, 2/5]
#		[1/2, 1/2, 3/5]
#		[2/2, 1/2, 3/5]
#		[1/2, 2/2, 3/5]
#		[2/2, 2/2, 3/5]
#		[1/2, 1/2, 4/5]
#		[2/2, 1/2, 4/5]
#		[1/2, 2/2, 4/5]
#		[2/2, 2/2, 4/5]
#		[1/2, 1/2, 5/5]
#		[2/2, 1/2, 5/5]
#		[1/2, 2/2, 5/5]
#		[2/2, 2/2, 5/5]


########################################################################################################################################################################################################
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
########################################################################################################################################################################################################

###################################################################################################################################
###							PUBLIC VOTING ENSEMBLES							###
###################################################################################################################################

#					################################################################################
#					################################################################################
#					#######					VOTING ENSEMBLES		  ######	1/1
#					################################################################################
#					################################################################################
#					###############			K(10)-FOLD CROSS-VALIDATION		########		1/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	VOTING:	(LR) | (CART-CLASSIFIER) | (SVM) |
#							------

# Voting Ensemble for Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(( ' logistic ' , model1))
model2 = DecisionTreeClassifier()
estimators.append(( ' cart ' , model2))
model3 = SVC()
estimators.append(( ' svm ' , model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())



#					################################################################################
#					################################################################################
#					#######					VOTING ENSEMBLES		  ######	1/1
#					################################################################################
#					################################################################################
#					###############			LEAVE-ONE-OUT CROSS-VALIDATION		########		2/2
#					################################################################################
#					###############		CLASSIFICATION-ACCURACY (Performance Metric)	########			1/5
#					################################################################################
#					################################################################################

#						a)	VOTING:	(LR) | (CART-CLASSIFIER) | (SVM) |
#							------

# Voting Ensemble for Classification
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
filename = ' _____.csv '
names = [ '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' , '  ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
seed=7
num_folds = 10
loocv = LeaveOneOut()
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(( ' logistic ' , model1))
model2 = DecisionTreeClassifier()
estimators.append(( ' cart ' , model2))
model3 = SVC()
estimators.append(( ' svm ' , model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

########################################################################################################################################################################################################
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
########################################################################################################################################################################################################

#		[1/1, 1/2, 1/5]
#		[1/1, 2/2, 1/5]
#		[1/1, 1/2, 2/5]
#		[1/1, 2/2, 2/5]
#		[1/1, 1/2, 3/5]
#		[1/1, 2/2, 3/5]
#		[1/1, 1/2, 4/5]
#		[1/1, 2/2, 4/5]
#		[1/1, 1/2, 5/5]
#		[1/1, 2/2, 5/5]

########################################################################################################################################################################################################
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
#======================================================================================================================================================================================================#
########################################################################################################################################################################################################



























#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
###	3.	DATA VISUALIZATION:
##################################
#	3.1:	Univariate/Unimodal Plots
#			i)	Attribute-based HISTOGRAMS

#						HISTOGRAM PLOT
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

#	3.2:		ii)	Density-Plots to determine Attribute-Distributions
#					Attribute-based DENSITY-PLOT Distributions
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

#	3.3:		iii)	BOX & WHISKER PLOTS
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

#	3.5:	Multivariate/Multimodal Plots:		- Intersections between variables
#			SCATTER-PLOT MATRIX
#						- Represents relationship between 2-variables as a 2-Dimm-dot
#						- A series/sequence of scatter-plots for multiple variable-pairs = Scatter-Plot Matrix
#	from matplotlib import pyplot
#	from pandas import read_csv
#	import numpy
#	filename = '___.csv'
#	names = ['', '', '', '']
#	data = read_csv(filename, names = names)
#	scatter_matrix(dataset)
#	pyplot.show()


#scatter_matrix(dataset)
#pyplot.show()
#print('##############################################################################')
#print('')

#	3.6:	Plot correlations between Attributes
#			CORRELATION MATRIX
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
#ticks = numpy.arange(0, 14, 1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
pyplot.show()

#	from matplotlib import pyplot
#	from pandas import read_csv
#	import numpy
#	filename = '___.csv'
#	names = ['', '', '', '']
#	data = read_csv(filename, names = names)
#	correlations = data.corr()
#plot CORRELATION-MATRIX
#	fig = pyplot.figure()
#	ax = fig.add_subplot(111)
#	cax = ax.matshow(dataset.corr(), vmin = -1, vmax = 1, interpolation = 'none')
#	fig.colorbar(cax)
#	#ticks = numpy.arange(0, 14, 1)
#	#ax.set_xticks(ticks)
#	#ax.set_yticks(ticks)
#	#ax.set_xticklabels(names)
#	#ax.set_yticklabels(names)
#	pyplot.show()

print('##############################################################################')
print('')

#	3.7:	SUMMARY OF IDEAS

# Determining transformations which could be used to better expose the structure of the data which may improve model accuracy:
#			- Feature-Selection and removing the most correlated attributes
#			- Normalizing the dataset to reduce the effect of differing scales
#			- Standardizing the dataset to reduce the effects of differing distributions
#			- For DECISION-TREE ALGORITHMS:
#							- Binning/Discretization of data (improves ACCURACY)

#################################################################################################
#################################################################################################
#################################################################################################

###	4.	VALIDATION/TESTING DATASET:
##########################################
#	4.1	Separate/Create a Validation-Dataset
#			Split-out validation dataset
array = dataset.values
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.20
seed = 7
# TRAINING DATA (100 - 20% = 80%):	X_train, Y_train
# TESTING/VALIDATION DATA (20%):	X_validation, Y_validation 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)

#	4.2	EVALUATING ALGORITHMS: Baseline
#			K-Fold (K = 10) Cross-Validation (Estimate ACCURACY)
num_folds = 10
seed = 7
scoring = 'accuracy'

#	4.3	BUILDING MODELS
#			LINEAR Algorithms: Logistic Regression (LR), Linear Discriminant Analysis (LDA)
#			NON-LINEAR Algorithms:	Classification and Regression Trees (CART), Gaussian Naive Bayes (NB), Support Vector Machines (SVM), K-Nearest Neighbors (KNN)
models = []
print('MODEL EVALUATIONS:	ACCURACY')
models.append(('Logistic Regression (LR)			', LogisticRegression()))
models.append(('Linear Discriminant Analysis (LDA)		', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbors (KNN)			', KNeighborsClassifier()))
models.append(('Classification and Regression Trees (CART)	', DecisionTreeClassifier()))
models.append(('Gaussian Naive Bayes (NB)			', GaussianNB()))
models.append(('Support Vector Machine (SVM)			', SVC()))
#models.append(('Lasso (LASSO)					', Lasso()))
#models.append(('ElasticNet (EN)				', ElasticNet()))
#models.append(('K-Nearest Neighbors (KNN)			', KNeighborsRegressor()))
#models.append(('Classification and Regression Trees (CART)	', DecisionTreeRegressor()))
#models.append(('(SVR)						', SVR()))

# evaluate each model in turn
results = []
names = []
print('##############################################################################')
print('')

#	4.4:	COMPARING ALGORITHMS
results = []
names = []
#from sklearn.model_selection import KFold, cross_val_score

for name, model in models:
	kfold = KFold(n_splits = num_folds, random_state = seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
print('##############################################################################')
print('')

#			- Examine the distribution of scores across all Cross-Validation folds by algorithm
fig = pyplot.figure()
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#	4.5:	STANDARDIZATION: Data is transformed such that each attribute has a mean-value of 0 and a standard-deviation of 1, and data-leakage is minimized (via pipelines)
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LA', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))

results = []
names = []

for name, model in pipelines:
	kfold = KFold(n_splits = num_folds, random_state = seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
print('##############################################################################')
print('')

#	4.6	SELECTING BEST MODEL: Examine distribution of n across the CROSS-VALIDATION FOLDS, and 
#			Compare Algorithms:	Plotting model-evaluation results & comparing the SPREAD & MEAN-ACCURACY
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	5.	IMPROVING RESULTS WITH TUNNING:
##############################################
#	5.1	Tunning the KNN-Algorithm
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
#k_values = numpy.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
param_grid = dict(n_neighbors = neighbors)
model = KNeighborsRegressor()
kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print('')
means = grid_result.cv_results_['mean_test_score']
stds= grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
#		Display the MEAN, STD.DEVIATION SCORES, best performing value for K:	Lowest score in leftmost-column = best
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
print('##############################################################################')
print('')

#	5.2	Tunning the SVM-Algorithm
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
#neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
#k_values = numpy.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C = c_values, kernel = kernel_values)
model = SVC()
kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print('')
means = grid_result.cv_results_['mean_test_score']
stds= grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
#		Display the MEAN, STD.DEVIATION SCORES, best performing value for K:	Lowest score in leftmost-column = best
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
print('##############################################################################')
print('')
#################################################################################################
#################################################################################################
#################################################################################################

###	6.	IMPROVING RESULTS WITH ENSEMBLE-METHODS:
#######################################################
#	6.1	Evaluate 4 different ENSEMBLE ML Algorithms (2 BOOSTING Methods {AdaBoost [AB]}{Gradient Boosting [GBM]}, 2 BAGGING Methods{Random Forests [RF]}{Extra Trees [ET]})
#			- Using Test-Harness, 10-Fold Cross-Validation, standardizing pipelines for training dataset
ensenbles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
print('##############################################################################')
print('')

#results = []
#names = []
#
#for name, model in ensembles:
#	kfold = KFold(n_splits = num_folds, random_state = seed)
#	cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)
#print('##############################################################################')
#print('')

#	6.2	Plotting the distribution of ACCURACY scores across the Cross-Validation Folds
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	7.	TUNING ENSEMBLE METHODS:
#######################################
#	7.1	Examination of the number of stages for gradient boosting (default # of boosting stages to perform (n_estimators) = 100, where a larger # of boosting-stages = better performance)
#			- Define a Parameter-Grid n_estimators values from 50 ... 400 in increments of 50 (where each setting is evaluated via 10-fold Cross-Validation)
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#param_grid = dict(n_estimators = numpy.array([50, 100, 150, 200, 250, 300, 350, 400]))
#model = GradientBoostingRegressor(random_state = seed)
#kfold = KFold(n_splits = num_folds, random_state = seed)
#grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
#grid_result = grid.fit(rescaledX, Y_train)
#print('##############################################################################')
#print('')

#	7.2	Summarize best-configuration and assess changes in performance with each different configuration
#print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))
#print('##############################################################################')
#print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	8.	FINALIZING MODEL:
################################
#	8.1	Preparing model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C = 1.5)#GradientBoostingRegressor(random_state = seed, n_estimators = 400)
model.fit(rescaledX, Y_train)
# Estimate ACCURACY on the Validation/Testing Dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print('##############################################################################')
print('')

#	8.2	Scale the inputs for the Validation/Testing Dataset and generate predictions
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print(mean_squared_error(Y_validation, predictions))
#print('##############################################################################')
#print('')
