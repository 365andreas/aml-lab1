import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from statistics import mean
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

# TODO: Experiment with other replacement techniques
# TODO: Maybe compute mean after splitting to validation and training set
x_train = x_train.fillna(x_train.mean())

test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
x_test = x_test.fillna(x_test.mean())

# TODO: outliers detection
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
# TODO: scaling ?

scaler = MinMaxScaler()
x_train_new = scaler.fit_transform(x_train)
print(x_train_new.shape)
# TODO: feature selection

cols = []
for col in x_train.columns:
    cols.append(col)

x_train_pd = pd.DataFrame(data=x_train_new, columns=cols)
print(x_train_pd)

'''
# TEMPORARY!!
unselected = []
#for i in range(3, 832):
#	unselected.append('x' + str(i))
x_train = x_train.drop(unselected, axis=1)

#TODO: try various regressors
#TODO: CV to tune parameters
reg = LinearRegression()
# score = R2 score
cv_results = cross_validate(reg, x_train, y_train, cv=10)
#print(sorted(cv_results.keys()))
print(cv_results['test_score'] )
print(mean(cv_results['test_score']))

###################################################################################
x_test = x_test.drop(unselected, axis=1)
y_test = reg.predict(x_test)
Id = test_set['id']
df=pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution1.csv', index=False)
'''
