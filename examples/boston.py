import numpy as np
import pandas as pd
from pyparsing import col
import scipy.sparse
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from tensorfm.sklearn import FactorizationMachineRegressor
from sklearn.metrics import mean_squared_error


boston = load_boston()
y = boston.target

#check whther has numerical columns 
X = pd.DataFrame(boston.data, columns=boston.feature_names)
cols = X.columns
n = X._get_numeric_data().columns
not_numerical = list(set(cols) - set(n))



data_split = train_test_split(X, y, test_size=100, random_state=0)
X_train, X_test, y_train, y_test = data_split

  
# converting to dict
# X_train = X_train.to_dict('dict')


# v = DictVectorizer()
# X_train = v.fit_transform(X_train)
# X_test = v.transform(X_test)
# print(X_train)

scaler_X = StandardScaler(with_mean=True, with_std=True)
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler(with_mean=True, with_std=True)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# X_train = scipy.sparse.csr_matrix(X_train)
# X_test = scipy.sparse.csr_matrix(X_test)
# print(X_test.shape)
fm = FactorizationMachineRegressor(max_iter=100, n_factors=3, eta=0.01, C=10000, random_state=12345, penalty='l2')


fm.fit(X_train, y_train)
y_pred= fm.predict(X_test)
print(y_pred)
print("MSE test ", mean_squared_error(y_test, y_pred))