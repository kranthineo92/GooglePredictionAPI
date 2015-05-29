__author__ = 'KranthiDhanala'

import pandas as pd
from sklearn import svm, grid_search
from sklearn.metrics import classification_report

def main():
    train_data = pd.read_csv("train_data.csv",header = None)
    test_data = pd.read_csv("test_data.csv",header = None)
    actual = pd.read_csv("test_pred.csv",header = None)
    actual_pred = actual[0].values

    train_pred = train_data[0].values
    #remove target column from dataframe
    train_data = train_data.drop([train_data.columns[0]],axis = 1)

    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters,n_jobs=-1, refit=True, cv=2)
    clf.fit(train_data.values,train_pred)
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))

    best_model = clf.best_estimator_
    best_model.fit(train_data.values,train_pred)
    pred = best_model.predict(test_data.values)


    target_nbrs = range(1,27)
    target = [str(x) for x in target_nbrs]
    #pred has all the classification
    print(classification_report(actual[0].values, pred,target_names= target))


    return


if __name__ == "__main__":
    main()
