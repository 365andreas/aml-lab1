import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from statistics import mean 
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.svm import SVC
from sklearn.feature_selection import RFE

x_train = pd.read_csv("X_train.csv") 
y_train = pd.read_csv("y_train.csv") 

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

#TODO: Experiment with other replacement techniques
#TODO: Maybe compute mean after splitting to validation and training set
x_train = x_train.fillna(x_train.mean())

test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
x_test = x_test.fillna(x_test.mean())

#TODO: outliers detection
clf = IsolationForest()
outliers_predict = clf.fit_predict(x_train)
outliers=0
for o in outliers_predict:
	if o==-1:
		outliers+=1

x_train['is_outlier'] = outliers_predict
x_train = x_train[x_train.is_outlier != -1]
x_train = x_train.drop('is_outlier', axis=1)

y_train['is_outlier'] = outliers_predict
y_train = y_train[y_train.is_outlier != -1]
y_train = y_train.drop('is_outlier', axis=1)
#TODO: scaling ?

scaler = MinMaxScaler()
x_train_new = scaler.fit_transform(x_train)
cols= list(x_train.columns.values)
x_train = pd.DataFrame(data=x_train_new, columns = cols)

#TODO: feature selection

'''
select_k_best_classifier = SelectKBest(chi2, k=400)
x_train_sel = select_k_best_classifier.fit_transform(x_train, y_train)
'''
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=400, step=1)
x_train_sel = rfe.fit_transform(x_train, y_train)

mask = select_k_best_classifier.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, cols):
    if bool:
        new_features.append(feature)

x_train = pd.DataFrame(data=x_train_sel, columns = new_features)
print(x_train)


#TODO: try various regressors
#TODO: CV to tune parameters
reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_train, y_train))

# score = R2 score
#cv_results = cross_validate(reg, x_train, y_train, cv=10)
#print(sorted(cv_results.keys())) 
#print(cv_results['test_score'] )
#print(mean(cv_results['test_score']))
'''
###################################################################################
x_test = x_test.drop(unselected, axis=1)
y_test = reg.predict(x_test)
Id = test_set['id']
df=pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution1.csv', index=False)
'''
