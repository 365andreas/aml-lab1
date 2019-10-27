import pandas as pd
import sys
import numpy
import copy

from sklearn.utils import column_or_1d
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, Lasso, SGDRegressor, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, AdaBoostRegressor
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.covariance import EllipticEnvelope
numpy.set_printoptions(threshold=sys.maxsize)


def sort(val):
    return val[1] # sort using the 2nd element

res = open("results_1", "w")

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train0 = y_train.drop('id', axis=1)
x_train0 = x_train.drop('id', axis=1)

for n_estimators, max_iter in [ (e, i) for e in [10, 100] for i in [10, 100]]:

    x_train = x_train0
    y_train = y_train0

    # 1. Missing Values
    est = ExtraTreesRegressor(n_estimators=n_estimators, random_state=42, max_features='sqrt', n_jobs=10, verbose=0)
    imputer = IterativeImputer( estimator=est, max_iter=max_iter, tol=0.001, n_nearest_features=100
                              , initial_strategy='median', imputation_order='ascending', verbose=2
                              , random_state=0)
    x_train_filled = imputer.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_filled)

    # 2. Outliers detection

    clf = IsolationForest( n_estimators=150
                        , max_samples=1000
                        , contamination=0.02
                        , max_features=1.0, bootstrap=False
                        , n_jobs=10
                        , behaviour='old'
                        , random_state=42
                        , verbose=0
                        , warm_start=False)
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

    scaler = RobustScaler()
    x_train_new = scaler.fit_transform(x_train)
    cols = list(x_train.columns.values)
    x_train = pd.DataFrame(data=x_train_new, columns=cols)

    # 4. Feature selection

    cv_score_list = []
    x_train_in = copy.deepcopy(x_train)
    x_train = copy.deepcopy(x_train_in)
    feature_selector = SelectKBest(f_regression, k=200)

    x_train_sel = feature_selector.fit_transform(x_train, y_train)
    mask = feature_selector.get_support()  # list of booleans
    new_features = []  # The list of your best features

    for bool, feature in zip(mask, cols):
        if bool:
            new_features.append(feature)
    x_train = pd.DataFrame(data=x_train_sel, columns=new_features)
    print("new_features size:", len(new_features))

    # Regression

    gbr = GradientBoostingRegressor( loss='ls'
                                , learning_rate=0.1
                                , n_estimators=600
                                , subsample=1.0
                                , criterion='friedman_mse'
                                , min_samples_split=0.05
                                , min_samples_leaf=1
                                , min_weight_fraction_leaf=0.0
                                , max_depth=5
                                , min_impurity_decrease=0.0
                                , min_impurity_split=None
                                , init=None
                                , random_state=42
                                , max_features=None
                                , alpha=0.9
                                , verbose=0
                                , max_leaf_nodes=None
                                , warm_start=False
                                , presort='auto'
                                , validation_fraction=0.1
                                , n_iter_no_change=None
                                , tol=0.0001)
    reg = AdaBoostRegressor(gbr)
    reg.fit(x_train, y_train)

    # score = R2 score
    cv_results = cross_validate(reg, x_train, y_train, cv=10)
    print("cross_validation scores:")
    print(cv_results['test_score'])
    print("mean of CV scores:")
    print(mean(cv_results['test_score']))

    print("cross_validation scores:", file=res)
    print(cv_results['test_score'], file=res)
    print("mean of CV scores:", file=res)
    print(mean(cv_results['test_score']), file=res)

    # TEST

    test_set = pd.read_csv("X_test.csv")
    x_test = test_set.drop('id', axis=1)
    # missing values
    x_test_filled = imputer.transform(x_test)
    x_test = pd.DataFrame(x_test_filled)
    # scaling
    x_test_scaled = scaler.transform(x_test)
    cols = list(x_test.columns.values)
    x_test = pd.DataFrame(data=x_test_scaled, columns=cols)
    # feature selection
    x_test = pd.DataFrame(data=x_test, columns=new_features)
    # prediction
    y_test = reg.predict(x_test)
    Id = test_set['id']
    df = pd.DataFrame(Id)
    df.insert(1, "y", y_test)
    df.to_csv(('solution1.csv_' + str(n_estimators) + 'estimators_' + str(max_iter) + 'max_iter'), index=False)
