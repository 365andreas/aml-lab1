import pandas as pd
import sys
import numpy

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV
from statistics import mean
import copy

numpy.set_printoptions(threshold=sys.maxsize)

def sort(val): 
    return val[2]  # sort using the 3rd element

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

# 1. Missing Values
# mean
# median
# (most_frequent)? 
# TODO: Maybe compute mean after splitting to validation and training set
filler = x_train.median()
x_train = x_train.fillna(filler)

# 2. Outliers detection

clf = IsolationForest()
outliers_predict = clf.fit_predict(x_train)
outliers = 0
for o in outliers_predict:
	if o == -1:
	    outliers += 1

print('number of outliers:', outliers)
x_train['is_outlier'] = outliers_predict
x_train = x_train[x_train.is_outlier != -1]
x_train = x_train.drop('is_outlier', axis=1)

y_train['is_outlier'] = outliers_predict
y_train = y_train[y_train.is_outlier != -1]
y_train = y_train.drop('is_outlier', axis=1)


# 3. Scaling

#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
x_train_new = scaler.fit_transform(x_train)
cols = list(x_train.columns.values)
x_train = pd.DataFrame(data=x_train_new, columns=cols)

# 4. Feature selection

cv_score_list = []
x_train_in = copy.deepcopy(x_train)
for n in range(50, 260, 10): # for select KBest
	for a in numpy.arange(0.1, 2.0, 0.1): # for Lasso
		x_train = copy.deepcopy(x_train_in)
		feature_selector = SelectKBest(f_regression, k=240)
		# svc = SVC(kernel="linear", C=1)
		# feature_selector = RFE(estimator=svc,
		#                        n_features_to_select=200, step=1, verbose=3)
		# feature_selector = RFECV(estimator=svc,
		#                          min_features_to_select=20,
		#                          cv=10,
		#                          step=1,
		#                          n_jobs=3, verbose=3)

		x_train_sel = feature_selector.fit_transform(x_train, y_train)
		mask = feature_selector.get_support()  # list of booleans
		new_features = []  # The list of your best features

		for bool, feature in zip(mask, cols):
			if bool:
				new_features.append(feature)
		x_train = pd.DataFrame(data=x_train_sel, columns=new_features)
		# print(x_train)
		#print("new_features size:", len(new_features))

		# TODO: try various regressors
		# SVR
		# Lasso
		# Ridge
		# TODO: CV to tune parameters
		'''
		param_grid = [
			 {'alpha': [1e-3, 1e-2, 1e-1, 1.0, 2.0],
			  'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
		]
		reg = Ridge()
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
		reg = Lasso(alpha=a).fit(x_train, y_train)
		print(reg.score(x_train, y_train))

		# score = R2 score
		cv_results = cross_validate(reg, x_train, y_train, cv=10)
		#print('Number of features:', n)
		# print(sorted(cv_results.keys()))
		print("cross_validation scores:")
		print(cv_results['test_score'])
		print("mean of CV scores:")
		print(mean(cv_results['test_score']))
		cv_score_list.append([n, a, mean(cv_results['test_score'])])

cv_score_list.sort(key = sort, reverse = True) 
print(cv_score_list)
###################################################################################
# TEST
'''
test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
# missing values
# TODO missing values of x_test must be filled with the corresponding values of x_train !!!
x_test = x_test.fillna(filler)
# scaling
x_test_scaled = scaler.fit_transform(x_test)
cols = list(x_test.columns.values)
x_test = pd.DataFrame(data=x_test_scaled, columns=cols)
# feature selection
# print(x_test)
x_test = pd.DataFrame(data=x_test, columns=new_features)
# print(' '.join(map(str, x_test.columns)))
# print(' '.join(map(str, x_train.columns)))
# prediction
y_test = reg.predict(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution1.csv', index=False)
'''
