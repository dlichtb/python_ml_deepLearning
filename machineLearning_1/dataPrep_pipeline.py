#!/usr/bin/env python

#	DATA PREPARATION PIPELINE:
#					- Prevents data-leakage in test-harness, via ensuring data-prep methods (i.e. standardization) are constrained to each fold of CROSS-VALIDATION procedure
#						- Focuses on TRAINING-Set

#	Pipeline functionality:
#				1.	- STANDARDIZE Data
#				2.	- Learn LINEAR DISCRIMINANT ANALYSIS (LDA) Model


# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load data
filename = '_____.csv'
names = ['', '', '', '']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create pipeline
estimators = []
estimators.append(( ' standardize ' , StandardScaler()))
estimators.append(( ' lda ' , LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
