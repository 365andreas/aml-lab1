# TODO: README: I have tried 2 ways:
# 1. I split the dataset into 5 folds, and at each iteration i keep 4 folds to train and 1 to validate.
#    At each iteration i do feature selection of 200 features, generate a new model, train and validate it.
#    After these iterations, I keep the features that the 5 SelectKBest have selected and I train again a model. Then I test it at the test set.

# 2. I split the dataset into 5 folds, and at each iteration i keep 4 folds to train and 1 to validate.
#    At each iteration i do feature selection of 200 features, generate a new model, train, validate it and test it at the test set.
#    After these iterations, I keep the mean value for the 5 generated prediction sets.
#    I don't know if it is correct :))))


import pandas as pd
import sys
import numpy
import copy

from functools import partial

from sklearn.utils import column_or_1d
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, Lasso, SGDRegressor, ElasticNet, LassoLarsCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, AdaBoostRegressor
from sklearn.impute import SimpleImputer
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFE, RFECV, SelectFwe
from statistics import mean
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeClassifier

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold

numpy.set_printoptions(threshold=sys.maxsize)

res = open("results", "w")

def sort(val):
    return val[1]  # sort using the 2nd element


x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

imputer = SimpleImputer(missing_values=numpy.nan, strategy='median')
x_train_filled = imputer.fit_transform(x_train)
x_train = pd.DataFrame(x_train_filled)
'''
outliers_list = []
for i in range(1, 20):
	clf = IsolationForest()
	outliers_predict = clf.fit_predict(x_train)
	#print(outliers_predict)
	outliers = 0
	outliers_id = []
	for i in range(1, len(outliers_predict)):
		o = outliers_predict[i]
		if o == -1:
		    outliers += 1
		    outliers_id.append(i)
	print(outliers_id)
	outliers_list.append(outliers_id)

result = set(outliers_list[0])
for s in outliers_list[1:]:
    result.intersection_update(s)
result = list(result)
print('result:', result)
'''

result= [137, 1018, 268, 797]  # these have come by 10 calls of IsolationForest
outliers_predict = [1] * x_train.shape[0]
for i in range(0, len(result)):
	outliers_predict[result[i]] = -1

# majority for outliers once

#TODO: take out only these outliers (result)
#TODO: split dataset into training and feature-selection set

x_train['is_outlier'] = outliers_predict
x_train = x_train[x_train.is_outlier != -1]
x_train = x_train.drop('is_outlier', axis=1)

y_train['is_outlier'] = outliers_predict
y_train = y_train[y_train.is_outlier != -1]
y_train = y_train.drop('is_outlier', axis=1)
#y_train = column_or_1d(y_train, warn=True)



# 3. Scaling

scaler = RobustScaler()
x_train_new = scaler.fit_transform(x_train)
cols = list(x_train.columns.values)
x_train = pd.DataFrame(data=x_train_new, columns=cols, index=x_train.index)

# TEST

test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
# missing values
x_test_filled = imputer.transform(x_test)
x_test = pd.DataFrame(x_test_filled)
# scaling
x_test_scaled = scaler.fit_transform(x_test)
cols = list(x_test.columns.values)
x_test = pd.DataFrame(data=x_test_scaled, columns=cols)

N = 10
# kf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
kf = KFold(n_splits=N, shuffle=False, random_state=42)
for r in [ GradientBoostingRegressor(n_estimators = 200, min_samples_split = 4)
		 , SVR(kernel="rbf", degree=3, gamma=0.01, coef0=0.0, tol=0.001, C=100, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
		#  , LassoCV(eps=1e-6, n_alphas=100, cv=3, max_iter=10000, n_jobs=10)
		#  , LassoLarsCV(cv=3, max_iter=10000, n_jobs=10)
		#  , AdaBoostRegressor(LassoCV(eps=1e-6, n_alphas=100, cv=3, max_iter=10000, n_jobs=10))
		#  , AdaBoostRegressor(LassoLarsCV(cv=3, max_iter=10000, n_jobs=10))
		 ]:

	train_scores = []
	val_scores = []
	features = []
	test_df=[]

	print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
	for train_index, test_index in kf.split(x_train, y_train):
		X_train_i, X_val_i = x_train.iloc[train_index], x_train.iloc[test_index]
		Y_train_i, Y_val_i = y_train.iloc[train_index], y_train.iloc[test_index]

		feature_selector = SelectKBest(f_regression, k=200)
		x_train_sel = feature_selector.fit_transform(X_train_i, Y_train_i.y) # .ravel())
		mask = feature_selector.get_support()  # list of booleans
		new_features = []  # The list of your best features

		for bool, feature in zip(mask, cols):
				if bool:
					new_features.append(feature)

		features.append(new_features)
		X_train_i = pd.DataFrame(data=x_train_sel, columns=new_features)

		reg = r.fit(X_train_i, Y_train_i.y)
		train_s = reg.score(X_train_i, Y_train_i.y)
		print('train score:', train_s)

		X_val_i = pd.DataFrame(data=X_val_i, columns=new_features)

		Y_pred = reg.predict(X_val_i)
		test_s = r2_score(Y_val_i, Y_pred)
		print('validation score:', test_s)

		train_scores.append(train_s)
		val_scores.append(test_s)

	# print("k in SelectKBest: ", k, file=res)
	print(val_scores, file=res)
	print("mean: ", mean(val_scores), file=res)
	print("std: ", numpy.std(val_scores), file=res)
	print("----------------------------------------------------------------", file=res)

'''
		# Way 2
		X_new_test_i = pd.DataFrame(data=x_test, columns=new_features)
		y_new_test = reg.predict(X_new_test_i)

		Id = test_set['id']
		df = pd.DataFrame(Id)
		df.insert(1, "y", y_new_test)
		test_df.append(df)

# Way 1
f = set(features[0])
for s in features[1:]:
    f.intersection_update(s)
features = list(f)
print("----------------------------------------------------------------")
print('features:', f)
print(len(f))

print("----------------------------------------------------------------")
print(train_scores)
print(mean(train_scores))
print("----------------------------------------------------------------")
'''

'''
# Way 2
print("----------------------------------------------------------------")
chunks=[]
for i in range (0, N):
	 chunks.append(test_df[i])
df_concat = pd.concat(chunks)
by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()
df_means.to_csv('solution2.csv', index=False)

def weighted_mean(w, x):
	d = {}
	d['id'] = (x.id).mean()
	d['y'] = numpy.average(x.y, weights=w)
	return pd.Series(d, index=['id', 'y'])

df_weighted_means = by_row_index.apply(partial(weighted_mean, val_scores))
df_weighted_means.to_csv('solution2_weighted.csv', index=False)
###################################################################################

# Way 1
X_train = pd.DataFrame(data=x_train, columns=features)

#reg = GradientBoostingRegressor(n_estimators = 400, min_samples_split =4).fit(X_train, y_train)
#print(reg.score(X_train, y_train))


# score = R2 score
cv_results = cross_validate(reg, X_train, y_train, cv=5)
# print(sorted(cv_results.keys()))
print("cross_validation scores:")
print(cv_results['test_score'])
print("mean of CV scores:")
print(mean(cv_results['test_score']))

###################################################################################
# TEST
# feature selection
# print(x_test)
x_test = pd.DataFrame(data=x_test, columns=features)
# print(' '.join(map(str, x_test.columns)))
# print(' '.join(map(str, x_train.columns)))
# prediction
y_test = reg.predict(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution1.csv', index=False)
'''
