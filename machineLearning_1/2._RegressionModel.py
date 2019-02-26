###################################################################################################################################
###					MACHINE LEARNING PROJECT TEMPLATE STRUCTURE (PYTHON)					###
###								 ------------------						###
###################################################################################################################################

#			################################################################################
#			################################################################################
#			###############			EVALUATING ALPHA-Values			########
#			################################################################################
#			################################################################################


#	1.	PREPARE PROBLEM
#				a)	LOAD Libraries

# Load libraries
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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

#				b)	LOAD Dataset

# Load dataset
filename = '____.csv'
names = ['', '', '', '']
dataset = read_csv(filename, delim_whitespace=True, names=names)

#	2.	SUMMARIZE DATA
#				a)	Descriptive Statistics

# Shape
print(dataset.shape)
print('##############################################################################')
print('')


# Type
print(dataset.dtypes)
print('##############################################################################')
print('')

# 1ST 20 Rows
print(dataset.head(20))
print('##############################################################################')
print('')

# Summary of Attribute-Distribution
set_option('precision', 1)
print(dataset.describe())
print('##############################################################################')
print('')

# Evaluate CORRELATIONS BETWEEN ALL NUMERIC-ATTRIBUTES
set_option('precision', 2)
print(dataset.corr(method = 'pearson'))
print('##############################################################################')
print('')

#				b)	Data Visualization

#						i)	UNIMODAL Data Visualizations

#								1.	HISTOGRAM
dataset.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)
pyplot.show()
print('##############################################################################')
print('')

#								2.	DENSITY
dataset.plot(kind = 'density', subplots = True, layout = (4,4), sharex = False, legend = False, fontsize = 1)
pyplot.show()
print('##############################################################################')
print('')

#								3.	BOX & WHISKER PLOT
dataset.plot(kind = 'box', subplots = True, layout = (4,4), sharex = False, sharey = False, fotsize = 8)
pyplot.show()
print('##############################################################################')
print('')

#						i)	UNIMODAL Data Visualizations

#								1.	SCATTER-PLOT
scatter_matrix(dataset)
pyplot.show()
print('##############################################################################')
print('')

#								2.	Visualize CORRELATIONS BETWEEN ATTRIBUTES
# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation= ' none ' )
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
print('##############################################################################')
print('')




#	3.	PREPARE DATA
#				a)	Data Cleansing
#				b)	Feature-Selection
#				c)	Data Transforms

#	4a.	EVALUATE ALGORITHMS
#				a)	SLPIT-OUT Validation Dataset

#						i)	Isolate VALIDATION/TESTING-Dataset
# Split-out validation dataset
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)

#				b)	Test Options & Evaluations-Metrics

num_folds = 10
seed = 7
# evaluation via: MEAN SQUARED ERROR (MSE)
scoring = 'neg_mean_squared_error'

#				c)	SPOT-CHECK Algorithms

# Spot-Check Algorithms
models = []
models.append(( ' LR ' , LinearRegression()))
models.append(( ' LASSO ' , Lasso()))
models.append(( ' EN ' , ElasticNet()))
models.append(( ' KNN ' , KNeighborsRegressor()))
models.append(( ' CART ' , DecisionTreeRegressor()))
models.append(( ' SVR ' , SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
kfold = KFold(n_splits=num_folds, random_state=seed)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
results.append(cv_results)
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
print('##############################################################################')
print('')

#################################################################################################
#	Look for ML-Model/Algorithm with LOWEST MSE (Measurement of EVALUATION-METRIC)		#
#################################################################################################


#				d)	Compare Algorithms

# Examination of Attribute-SCORING-Distribution across all CROSS-VALIDATION FOLDS by the ML-Model/Algorithm
fig = pyplot.figure()
fig.suptitle( ' Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#	4b.	EVALUATE ALGORITHMS
#				a)	STANDARDIZATION

# Standardize the dataset
pipelines = []
pipelines.append(( ' ScaledLR ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' LR ' , LinearRegression())])))
pipelines.append(( ' ScaledLASSO ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' LASSO ' , Lasso())])))
pipelines.append(( ' ScaledEN ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' EN ' , ElasticNet())])))
pipelines.append(( ' ScaledKNN ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' KNN ' , KNeighborsRegressor())])))
pipelines.append(( ' ScaledCART ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' CART ' , DecisionTreeRegressor())])))
pipelines.append(( ' ScaledSVR ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' SVR ' , SVR())])))

results = []
names = []

for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

print('##############################################################################')
print('')

#########################################################################################################
#	Provides a list of the MSE (EVALUATION-METRIC), to then examine the effect(s) of SCALING	#
#########################################################################################################

# Examine ATTRIBUTE-SCORING-DISTRIBUTIONS across the CROSS-VALIDATION-FOLDS
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle( ' Scaled Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#	5.	IMPROVING ACCURACY
#				a)	Algorithm TUNNING

# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_[ ' mean_test_score ' ]
stds = grid_result.cv_results_[ ' std_test_score ' ]
params = grid_result.cv_results_[ ' params ' ]
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

print('##############################################################################')
print('')

#				b-i)	ENSEMBLES

# ensembles
ensembles = []
ensembles.append(( ' ScaledAB ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' AB ' , AdaBoostRegressor())])))
ensembles.append(( ' ScaledGBM ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' GBM ' , GradientBoostingRegressor())])))
ensembles.append(( ' ScaledRF ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' RF ' , RandomForestRegressor())])))
ensembles.append(( ' ScaledET ' , Pipeline([( ' Scaler ' , StandardScaler()),( ' ET ' , ExtraTreesRegressor())])))

results = []
names = []

for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#################################################################################################
#	Shows the CALCULATED MEAN-SQUARED-ERROR for each ML-Method using default-parameters	#
#################################################################################################

print('##############################################################################')
print('')

# Show DISTRIBUTION OF SCORING ACROSS THE CROSS-VALIDATION FOLDS
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle( ' Scaled Ensemble Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('##############################################################################')
print('')

#				b-ii)	TUNNING ENSEMBLES

# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

# Summarize the best configuration and get an idea of how performance changed with each different configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_[ ' mean_test_score ' ]
stds = grid_result.cv_results_[ ' std_test_score ' ]
params = grid_result.cv_results_[ ' params ' ]
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

#########################################################################################################################################################################
#	Shows best configuration was n estimators=400 resulting in a MEAN-SQUARED-ERROR of -9.356471, about 0.65 units better than the untuned method (5. b-i))		#
#########################################################################################################################################################################

print('##############################################################################')
print('')

#	6.	FINALIZE MODEL
#				a)	Execute Predictions on Validation/TESTING-Set
#				b)	Create STAND-ALONE MODEL (for entire TRAINING-Set)
#				c)	SAVE model to LOAD for later use

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)

# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
print('##############################################################################')
print('')
