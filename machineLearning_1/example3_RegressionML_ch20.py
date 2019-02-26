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
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostegressor
from sklearn.metrics import mean_squared_error
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
filename = ('____')
names = ['', '', '', '', '']
dataset = read_csv(filename, delim_whitespace = True, names = names)
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']#	### Specifies Column-Names
#dataset = pandas.read_csv(url, names=names)
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
print('ATTRIBUTE DATA-TYPES:')
print(dataset.dtypes)
#	2.2:	Peek at the data itself
print('HEAD(20):')
print(dataset.head(20))
print('')
#	2.3:	Statistical summary of all attributes
print('STATISTICAL SUMMARY FOR EACH COLUMN/ATTRIBUTE:')
set_option('precision', 1)
print(dataset.describe())
print('')
#	2.4:	Breakdown of the data by the class variable
print(dataset.groupby('class').size())
print('##############################################################################')
print('')

#	2.5:	Taking a look at the correlation between all of the numeric attributes
#			CORRELATION
set_option('precision', 2)
print(dataset.corr(method = 'pearson'))
# Assess where 'LSTAT' has highest |%|-correlation to an output-variable
print('##############################################################################')
print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	3.	DATA VISUALIZATION:
##################################
#	3.1:	Univariate/Unimodal Plots
#			Attribute-based HISTOGRAMS
dataset.hist(sharex = False, sharey = False, xlabelsie = 1, ylabelsize = 1)
pyplot.show()
print('##############################################################################')
print('')

#	3.2:	Density-Plots to determine Attribute-Distributions
#			Attribute-based DENSITY-PLOT Distributions
dataset.plot(kind = 'density', subplots = True, layout(4,4), sharex = False, legend = False, fontsize = 1)
pyplot.show()
print('##############################################################################')
print('')

#	3.3:	Univariate Plots
#			BOX & WHISKER PLOTS
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize = 8)
pyplot.show()
print('##############################################################################')
print('')
##plt.show()
#			HISTOGRAM PLOT
#dataset.hist()
#plt.show()

#	3.4:	Multivariate/Multimodal Plots:		- Intersections between variables
#			SCATTER-PLOT MATRIX
scatter_matrix(dataset)
pyplot.show()
print('##############################################################################')
print('')

#	3.5:	Plot correlations between Attributes
#			CORRELATION MATRIX
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
ticks = numpy.arange(0, 14, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#	3.6:	SUMMARY OF IDEAS

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
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
# TRAINING DATA (100 - 20% = 80%):	X_train, Y_train
# TESTING/VALIDATION DATA (20%):	X_validation, Y_validation 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

#	4.2	EVALUATING ALGORITHMS: Baseline
#			K-Fold (K = 10) Cross-Validation (Estimate ACCURACY)
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

#	4.3	BUILDING MODELS
#			LINEAR Algorithms: Logistic Regression (LR), Linear Discriminant Analysis (LDA)
#			NON-LINEAR Algorithms:	K-Nearest Neighbors (KNN), Classification and Regression Trees (CART), Gaussian Naive Bayes (NB), Support Vector Machines (SVM)
models = []
print('MODEL EVALUATIONS:	ACCURACY')
models.append(('Linear Regression (LR)				', LinearRegression()))
models.append(('Lasso (LASSO)					', Lasso()))
models.append(('ElasticNet (EN)					', ElasticNet()))
models.append(('K-Nearest Neighbors (KNN)			', KNeighborsRegressor()))
models.append(('Classification and Regression Trees (CART)	', DecisionTreeRegressor()))
models.append(('(SVR)						', SVR()))
#models.append(('Gaussian Naive Bayes (NB)			', GaussianNB()))
#models.append(('Support Vector Machine (SVM)			', SVC()))
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
fig = pypotfigure()
fig.suptitle('Algorithms Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#	4.5:	STANDARDIZATION: Data is transformed such that each attribute has a mean-value of 0 and a standard-deviation of 1, and data-leakage is minimized (via pipelines)
pipelines = []
pipelines.append(('ScaledLA', Pipeline([('Scaler', StandardScaler()), ('LA', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('ES', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())])))

results = []
names = []

for name, model in pipelines:
	kfold = KFold(n_splits = num_folds, random_state = seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring=scoring)
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
k_values = numpy.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
param_grid = dict(n_neighbors = k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)

#		Display the MEAN, STD.DEVIATION SCORES, best performing value for K:	Lowest score in leftmost-column = best
names = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_resuts_['params']
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
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET',E Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))

results = []
names = []

for name, model in ensembles:
	kfold = KFold(n_splits = num_folds, random_state = seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
print('##############################################################################')
print('')

#	6.2	Plotting the distribution scores across the Cross-Validation Folds
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
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
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators = numpy.array([50, 100, 150, 200, 250, 300, 350, 400]))
model = GradientBoostingRegressor(random_state = seed)
kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print('##############################################################################')
print('')

#	7.2	Summarize best-configuration and assess changes in performance with each different configuration
print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
print('##############################################################################')
print('')

#################################################################################################
#################################################################################################
#################################################################################################

###	8.	FINALIZING MODEL:
################################
#	8.1	Preparing model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state = seed, n_estimators = 400)
print('##############################################################################')
print('')

#	8.2	Scale the inputs for the Validation/Testing Dataset and generate predictions
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
print('##############################################################################')
print('')
