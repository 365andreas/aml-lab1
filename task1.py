import pandas as pd
import sys
import numpy

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from statistics import mean


numpy.set_printoptions(threshold=sys.maxsize)

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

# Missing Values

# TODO: Experiment with other replacement techniques
# TODO: Maybe compute mean after splitting to validation and training set
x_train = x_train.fillna(x_train.mean())

test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
x_test = x_test.fillna(x_test.mean())

# Outliers detection

clf = IsolationForest()
outliers_predict = clf.fit_predict(x_train)
outliers = 0
for o in outliers_predict:
    if o == -1:
        outliers += 1

x_train['is_outlier'] = outliers_predict
x_train = x_train[x_train.is_outlier != -1]
x_train = x_train.drop('is_outlier', axis=1)

y_train['is_outlier'] = outliers_predict
y_train = y_train[y_train.is_outlier != -1]
y_train = y_train.drop('is_outlier', axis=1)


# Scaling

scaler = MinMaxScaler()
x_train_new = scaler.fit_transform(x_train)
cols = list(x_train.columns.values)
x_train = pd.DataFrame(data=x_train_new, columns=cols)

# Feature selection

select_k_best_classifier = SelectKBest(chi2, k=100)
x_train_sel = select_k_best_classifier.fit_transform(x_train, y_train)
'''
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=400, step=1)
x_train_sel = rfe.fit_transform(x_train, y_train)
'''

mask = select_k_best_classifier.get_support()  # list of booleans
new_features = []  # The list of your K best features

for bool, feature in zip(mask, cols):
    if bool:
        new_features.append(feature)

x_train = pd.DataFrame(data=x_train_sel, columns=new_features)
print(x_train)


# TODO: try various regressors
# SVR
# Lasso
# Ridge
# TODO: CV to tune parameters
param_grid = [
    {'alpha': [1e-3, 1e-2, 1e-1, 1, 10]},
    # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

gs = GridSearchCV(Ridge(),
                  param_grid=param_grid,
                  scoring=make_scorer(r2_score),
                  cv=10,
                  n_jobs=-1, refit=True, return_train_score=True)
gs.fit(x_train, y_train)
results = gs.cv_results_
reg = gs.best_estimator_

# reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_train, y_train))

# score = R2 score
# cv_results = cross_validate(reg, x_train, y_train, cv=10)
# print(sorted(cv_results.keys()))
# print(cv_results['test_score'] )
# print(mean(cv_results['test_score']))

###################################################################################
# TEST

# missing values
x_test = x_test.fillna(x_test.mean())
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
