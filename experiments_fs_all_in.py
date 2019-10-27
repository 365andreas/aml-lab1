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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold
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

from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold

numpy.set_printoptions(threshold=sys.maxsize)

res = open("results_all_in", "w")

def sort(val):
    return val[1]  # sort using the 2nd element


x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

N = 5
# kf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
kf = KFold(n_splits=N, shuffle=True, random_state=42)
r = GradientBoostingRegressor( loss='ls'
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
for imputer, label in [ (SimpleImputer(missing_values=numpy.nan, strategy='median'), "simple median imputer")
         ] + [ (IterativeImputer( estimator=ExtraTreesRegressor(n_estimators=n, random_state=0, max_features='sqrt', n_jobs=1, verbose=0)
                               , max_iter=m, tol=0.001, n_nearest_features=100
                               , initial_strategy='median'
                               , imputation_order='ascending'
                               , verbose=2
                               , random_state=42), "[iterative imputer] n_estimators: " + str(n) + " max_iter: " + str(m))
                for n in [10, 100]
                for m in [10, 100, 1000]
		 ]:


    train_scores = []
    val_scores = []
    features = []
    test_df=[]

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(label, file=res)
    print(label)
    for train_index, test_index in kf.split(x_train):
        X_train_i, X_val_i = x_train.iloc[train_index], x_train.iloc[test_index]
        Y_train_i, Y_val_i = y_train.iloc[train_index], y_train.iloc[test_index]

        imputer = imputer
        x_train_i_filled_tmp = imputer.fit_transform(X_train_i)
        x_train_i_filled = pd.DataFrame(x_train_i_filled_tmp)
        x_val_i_filled_tmp = imputer.transform(X_val_i)
        x_val_i_filled = pd.DataFrame(x_val_i_filled_tmp)

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
        # clf = EllipticEnvelope ###################
        outliers_predict = clf.fit_predict(x_train_i_filled)

        # outliers_id = []
        # id = 0
        outliers = 0
        for o in outliers_predict:
            if o == -1:
                outliers += 1
                # outliers_id.append(id)
        # print('number of outliers:', outliers, file=res)
        # print('outliers ids:', outliers_id, file=res)

        x_train_i_filled['is_outlier'] = outliers_predict
        x_train_i_filled = x_train_i_filled[x_train_i_filled.is_outlier != -1]
        x_train_i_filtered_outliers = x_train_i_filled.drop('is_outlier', axis=1)

        y_train_i = pd.DataFrame(Y_train_i)
        y_train_i['is_outlier'] = outliers_predict
        y_train_i = y_train_i[y_train_i.is_outlier != -1]
        y_train_i_filtered_outliers = y_train_i.drop('is_outlier', axis=1)

        # do not fit just predict
        outliers_predict_val = clf.predict(x_val_i_filled)

        outliers_val = 0
        for o in outliers_predict_val:
            if o == -1:
                outliers_val += 1
        print('number of outliers_val:', outliers_val)

        x_val_i_filled['is_outlier'] = outliers_predict_val
        x_val_i_filled = x_val_i_filled[x_val_i_filled.is_outlier != -1]
        x_val_i_filtered_outliers = x_val_i_filled.drop('is_outlier', axis=1)

        y_val_i = pd.DataFrame(Y_val_i)
        y_val_i['is_outlier'] = outliers_predict_val
        y_val_i = y_val_i[y_val_i.is_outlier != -1]
        y_val_i_filtered_outliers = y_val_i.drop('is_outlier', axis=1)

        # 3. Scaling

        scaler = RobustScaler()
        x_train_i_scaled_tmp = scaler.fit_transform(x_train_i_filtered_outliers)
        cols = list(x_train_i_filtered_outliers.columns.values)
        x_train_i_scaled = pd.DataFrame(data=x_train_i_scaled_tmp, columns=cols,
                                        index=x_train_i_filtered_outliers.index)

        x_val_i_scaled_tmp = scaler.transform(x_val_i_filtered_outliers)
        cols_val = list(x_val_i_filtered_outliers.columns.values)
        x_val_i_scaled = pd.DataFrame(data=x_val_i_scaled_tmp, columns=cols_val,
                                      index=x_val_i_filtered_outliers.index)

        # 4. Feature selection

        feature_selector = SelectKBest(f_regression, k=200)
        x_train_i_sel = feature_selector.fit_transform(x_train_i_scaled, y_train_i_filtered_outliers.y)
        mask = feature_selector.get_support()  # list of booleans
        new_features = []  # The list of your best features

        for bool, feature in zip(mask, cols):
                if bool:
                    new_features.append(feature)

        x_train_i_feature_selected = pd.DataFrame(data=x_train_i_sel, columns=new_features)

        # here features are selected in the train and applied to the val set
        x_val_i_sel = feature_selector.transform(x_val_i_scaled)
        x_val_i_feature_selected = pd.DataFrame(data=x_val_i_sel, columns=new_features)

        # 5. Regression

        reg = r.fit(x_train_i_feature_selected, y_train_i_filtered_outliers.y)
        # GradientBoostingRegressor(n_estimators = 200, min_samples_split = 4).fit(x_train_i_feature_selected, y_train_i_filtered_outliers.y)
        train_s = reg.score(x_train_i_feature_selected, y_train_i_filtered_outliers.y)
        print('train score:', train_s)

        val_s = reg.score(x_val_i_feature_selected, y_val_i_filtered_outliers.y)
        print('validation score:', val_s)
        Y_pred = reg.predict(x_val_i_feature_selected)
        test_s = r2_score(y_val_i_filtered_outliers, Y_pred)

        train_scores.append(train_s)
        val_scores.append(test_s)

    print(val_scores, file=res)
    print("mean: ", mean(val_scores), file=res)
    print("std: ", numpy.std(val_scores), file=res)
    print("----------------------------------------------------------------", file=res)

'''
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
# print(x_test)
x_test = pd.DataFrame(data=x_test, columns=features)
# prediction
y_test = reg.predict(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution1.csv', index=False)
'''
