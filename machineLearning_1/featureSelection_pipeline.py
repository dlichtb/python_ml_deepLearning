#!/usr/bin/env python

#	FEATURE EXTRACTION PIPELINE:
#					- Prevents data-leakage in test-harness, via ensuring data-prep methods (i.e. feature extraction) are constrained to each fold of CROSS-VALIDATION procedure
#						- Focuses on TRAINING-Set

#	FeatureUnion()
#			- Allows results of multiple FEATURE-SELECTION & EXTRACTION-PROCEDURES to be combined into a larger dataset for a model to be trained
#			- FEATURE-EXTRACTION & FEATURE-UNION occur within each FOLD of the CROSS-VALIDATION procedure

#	Pipeline functionality:
#				1.	- FEATURE-EXTRACTION with PRINCIPAL-COMPONENT-ANALYSIS
#												- 3 features
#				2.	- FEATURE-EXTRACTION with STATISTICAL-SELECTION
#												- 6 features
#				3.	- FEATURE-UNION
#				4.	- Learn LOGISTIC-REGRESSION (LR) Model

# Create a pipeline that extracts features from the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# load data
filename = '_____.csv'
names = ['', '', '', '']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create feature union
features = []
features.append(( ' pca ' , PCA(n_components=3)))
features.append(( ' select_best ' , SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(( ' feature_union ' , feature_union))
estimators.append(( ' logistic ' , LogisticRegression()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
