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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
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

numpy.set_printoptions(threshold=sys.maxsize)


def sort(val):
    return val[1]  # sort using the 2nd element


x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

# 1. Missing Values
# mean
# median
# most_frequent
# TODO: Maybe compute mean after splitting to validation and training set
imputer = SimpleImputer(missing_values=numpy.nan, strategy='median')
# est = ExtraTreesRegressor(n_estimators=10, random_state=0, max_features='sqrt', n_jobs=12, verbose=0)
# imputer = IterativeImputer(estimator=est, max_iter=10, tol=0.001, n_nearest_features=None,
#                            initial_strategy='median', imputation_order='ascending', verbose=2,
#                            random_state=0)
x_train_filled = imputer.fit_transform(x_train)
x_train = pd.DataFrame(x_train_filled)

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
y_train = column_or_1d(y_train, warn=True)



# 3. Scaling

# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
# scaler = PowerTransformer()
scaler = RobustScaler()
x_train_new = scaler.fit_transform(x_train)
cols = list(x_train.columns.values)
x_train = pd.DataFrame(data=x_train_new, columns=cols)

# 4. Feature selection

cv_score_list = []
x_train_in = copy.deepcopy(x_train)
for n in range(1):  # for select KBest
    #for a in numpy.arange(0.1, 1.5, 0.1):
        x_train = copy.deepcopy(x_train_in)
        # feature_selector = SelectKBest(f_regression, k=250)
        # svc = SVC(kernel="linear", C=1)
        svc = ExtraTreeClassifier()
        #feature_selector = RFE(estimator=svc,
        #                        n_features_to_select=240, step=1, verbose=3)
        clf = LassoCV(cv=10)
        feature_selector = SelectFromModel(clf, threshold=0.25, max_features=300)
        # feature_selector = RFECV(estimator=svc,
        #                          min_features_to_select=1,
        #                          cv=10,
        #                          step=1,
        #                          n_jobs=10, verbose=1)
        # print("new_features size:", feature_selector.n_features_)
        # print("grid scores:", feature_selector.grid_scores_)

        x_train_sel = feature_selector.fit_transform(x_train, y_train)
        mask = feature_selector.get_support()  # list of booleans
        new_features = []  # The list of your best features

        for bool, feature in zip(mask, cols):
            if bool:
                new_features.append(feature)
        x_train = pd.DataFrame(data=x_train_sel, columns=new_features)
        # print(x_train)
        print("new_features size:", len(new_features))


        # TODO: try various regressors
        # SVR
        # Lasso
        # Ridge
        # TODO: CV to tune parameters

        param_grid = [
            # {'n_estimators': [200, 250, 300, 350, 400, 500, 600, 700, 800, 1000],
            #  'min_samples_split': [2, 4, 5, 6, 7, 8, 9, 10, 12]}]
            {'n_estimators': [200],
             'min_samples_split': [4]}]
        reg = GradientBoostingRegressor()
        gs = GridSearchCV(reg,
                          param_grid=param_grid,
                          scoring=make_scorer(r2_score),
                          cv=10,
                        n_jobs=-1, refit=True, return_train_score=True, verbose=1)
        gs.fit(x_train, y_train)
        print(gs.cv_results_)
        print(gs.best_score_)
        print(gs.best_params_)
        reg = gs.best_estimator_

        #reg = GradientBoostingRegressor().fit(x_train, y_train)
        #print(reg.score(x_train, y_train))

        # score = R2 score
        cv_results = cross_validate(reg, x_train, y_train, cv=10)
        # print('Number of features:', n)
        # print(sorted(cv_results.keys()))
        print("cross_validation scores:")
        print(cv_results['test_score'])
        print("mean of CV scores:")
        print(mean(cv_results['test_score']))
        cv_score_list.append([n, mean(cv_results['test_score'])])

cv_score_list.sort(key=sort, reverse=True)
print(cv_score_list)
###################################################################################
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
