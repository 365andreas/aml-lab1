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

from sklearn.utils import column_or_1d
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, Lasso, SGDRegressor, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFE, RFECV
from statistics import mean
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeClassifier

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=sys.maxsize)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor




def sort(val):
    return val[1]  # sort using the 2nd element


def detect_outliers(train_data):
	outliers_list = []
	for i in range(1, 1000):
		clf = IsolationForest(contamination='auto', behaviour='new')
		outliers_predict = clf.fit_predict(train_data)
		print("------------------- Isolation Forest ", i)
		outliers = 0
		outliers_id = []
		for i in range(1, len(outliers_predict)):
			o = outliers_predict[i]
			if o == -1:
				outliers += 1
				outliers_id.append(i)
		outliers_list.append(outliers_id)

	my_dict = {}
	results=[]
	for i in range(0, x_train.shape[0]):
		my_dict[i] = 0
	for l in outliers_list:
		for i in l:
			my_dict[i]+=1

	my_dict_s = sorted(my_dict.items(), key=lambda kv: kv[1])
	print(my_dict_s)

	for i in my_dict.keys():
		if my_dict[i] > 900:
			results.append(i)

	#print_histogram(d)
	return results


x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

imputer = SimpleImputer(missing_values=numpy.nan, strategy='median')
# est = ExtraTreesRegressor(n_estimators=10, random_state=42, max_features='sqrt', n_jobs=-1, verbose=0)
# imputer = IterativeImputer( estimator=est, max_iter=10, tol=0.001, n_nearest_features=100
#                               , initial_strategy='median', imputation_order='ascending', verbose=2
#                               , random_state=0)
x_train_filled = imputer.fit_transform(x_train)
x_train = pd.DataFrame(x_train_filled)


results = [44, 108, 137, 268, 332, 341, 461, 502, 580, 606, 664, 797, 833, 839, 882, 1007, 1018, 1148] # this is result after 1000 isolation forests

# 3. Scaling

scaler = RobustScaler()
x_train_new = scaler.fit_transform(x_train)
cols = list(x_train.columns.values)
x_train = pd.DataFrame(data=x_train_new, columns=cols, index=x_train.index)

#results = detect_outliers(x_train)
#print(results)


outliers_predict = [1] * x_train.shape[0]
for i in range(0, len(results)):
	outliers_predict[results[i]] = -1


#TODO: take out only these outliers (result)
#TODO: split dataset into training and feature-selection set

x_train['is_outlier'] = outliers_predict
x_train = x_train[x_train.is_outlier != -1]
x_train = x_train.drop('is_outlier', axis=1)

y_train['is_outlier'] = outliers_predict
y_train = y_train[y_train.is_outlier != -1]
y_train = y_train.drop('is_outlier', axis=1)
#y_train = column_or_1d(y_train, warn=True)


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

# SPLIT!
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

train_scores = []
val_scores = []
features = []
test_df=[]
'''
kf = KFold(n_splits=20, random_state=None, shuffle=False )
kf.get_n_splits(x_train)
for train_index, test_index in kf.split(x_train):
	X_train_i, X_val_i = x_train.iloc[train_index], x_train.iloc[test_index]
	Y_train_i, Y_val_i = y_train.iloc[train_index], y_train.iloc[test_index]

	feature_selector = SelectKBest(f_regression, k=200)
	x_train_sel = feature_selector.fit_transform(X_train_i, Y_train_i)
	mask = feature_selector.get_support()  # list of booleans
	new_features = []  # The list of your best features

	for bool, feature in zip(mask, cols):
    		if bool:
	        	new_features.append(feature)

	features.append(new_features)
	X_train_i = pd.DataFrame(data=x_train_sel, columns=new_features)

	#TODO: outliers after feature selection

	reg = GradientBoostingRegressor(n_estimators = 200, min_samples_split = 4).fit(X_train_i, Y_train_i)
	#reg = SVR('rbf', C=100.0, gamma=0.01).fit(X_train_i, Y_train_i)
	#reg = RandomForestRegression(n_estimators = 200)
	train_s = reg.score(X_train_i, Y_train_i)
	print('train score:', train_s)

	X_val_i = pd.DataFrame(data=X_val_i, columns=new_features)

	Y_pred = reg.predict(X_val_i)
	test_s = r2_score(Y_val_i, Y_pred)
	print('validation score:', test_s)

	train_scores.append(train_s)
	val_scores.append(test_s)


# Way 1
f = set(features[0])
for s in features[1:]:
    f.intersection_update(s)
features = list(f)
print("----------------------------------------------------------------")
print('features:', sorted(f))
print(len(f))


print("----------------------------------------------------------------")
print(train_scores)
print(mean(train_scores))
print("----------------------------------------------------------------")
print(val_scores)
print(mean(val_scores))
'''

###################################################################################

# Way 1
# these are the features from the above feature selection
features = [9, 11, 16, 20, 23, 25, 29, 33, 36, 42, 44, 45, 48, 49, 53, 55, 56, 58, 59, 60, 71, 72, 75, 78, 80, 81, 87, 90, 93, 99, 105, 109, 110, 113, 116, 117, 118, 121, 123, 124, 125, 130, 132, 134, 145, 149, 151, 154, 156, 166, 167, 174, 176, 178, 191, 209, 210, 214, 224, 234, 239, 243, 245, 246, 248, 250, 252, 254, 255, 261, 266, 267, 270, 273, 279, 284, 288, 292, 297, 302, 307, 309, 312, 315, 319, 320, 330, 344, 346, 347, 348, 362, 366, 377, 383, 387, 419, 420, 422, 428, 438, 445, 451, 453, 455, 457, 466, 472, 474, 480, 481, 484, 486, 502, 504, 506, 510, 511, 513, 523, 525, 529, 530, 531, 533, 534, 537, 541, 548, 551, 554, 556, 563, 564, 565, 569, 593, 595, 596, 607, 610, 614, 617, 618, 623, 624, 629, 632, 635, 636, 637, 645, 646, 662, 668, 669, 670, 671, 672, 682, 684, 686, 695, 696, 706, 709, 721, 724, 726, 731, 732, 737, 738, 744, 756, 757, 760, 767, 768, 769, 776, 791, 793, 798, 799, 803, 808, 809, 811, 816, 817, 828, 831]

x_train = pd.DataFrame(data=x_train, columns=features)


# Remove outliers again

#results = detect_outliers(x_train)
#print(results)
#this is the result of outliers
results = [6, 12, 98, 112, 114, 129, 155, 176, 181, 190, 233, 242, 245, 249, 280, 293, 317, 320, 351, 369, 393, 457, 459, 490, 512, 517, 543, 555, 578, 597, 621, 625, 653, 714, 794, 882, 947, 951, 956, 962, 1081, 1140, 1170, 1182]


outliers_predict = [1] * x_train.shape[0]
for i in range(0, len(results)):
	outliers_predict[results[i]] = -1

x_train['is_outlier'] = outliers_predict
x_train = x_train[x_train.is_outlier != -1]
x_train = x_train.drop('is_outlier', axis=1)

y_train['is_outlier'] = outliers_predict
y_train = y_train[y_train.is_outlier != -1]
y_train = y_train.drop('is_outlier', axis=1)


'''
# with gridsearchCV
param_grid = [
             {'n_estimators': [200, 250, 300, 350, 400, 450, 500, 600],
              'min_samples_split': [0.1, 1.0, 2, 4]}]
reg = GradientBoostingRegressor()
gs = GridSearchCV(reg,
                  param_grid=param_grid,
                  scoring=make_scorer(r2_score),
                  cv=10,
                  n_jobs=-1, refit=True, return_train_score=True)

gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
reg = gs.best_estimator_
'''

# No gridseacrhCV
gbr = GradientBoostingRegressor(n_estimators=600, min_samples_split=2)
reg = AdaBoostRegressor(gbr, n_estimators=400).fit(x_train, y_train)
print(reg.score(x_train, y_train))
# score = R2 score
cv_results = cross_validate(reg, x_train, y_train, cv=10)
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
df.to_csv('solution_ada_boost.csv', index=False)
