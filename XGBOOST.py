import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

#load the data
data = pd.read_csv('../../../Data/BankChurners.csv')

#describe the data
data.describe()
data.info()

#drop the naive bayes columns but preserve them into data_bayes_cols
naive_bayes_cols = data.columns[(len(data.columns) - 2):]
data_bayes_cols = data[naive_bayes_cols]

data.drop(naive_bayes_cols, axis=1, inplace=True)

#select only object columns
obj_data = data.select_dtypes(include=['object']).copy()
obj_data_columns = obj_data.columns
# no null values
obj_data[obj_data.isnull().any(axis=1)]

#create dummy variables with dropping the first column after the dummy creation
dummy_data = pd.get_dummies(obj_data, drop_first=True)

#join the dummy data and the raw data with object columns excluded
data = data.drop(obj_data_columns, axis=1).join(dummy_data)

#create pandas profiler
prof = ProfileReport(data)
prof.to_file(output_file='output.html')

with open('Data/data.pickle', 'wb') as f:
    pickle.dump(data, f)

#remove columns
data.columns
data.drop(['CLIENTNUM'], inplace=True, axis=1)

test_size = 0.2
dep_col = ''

X = data.drop('Attrition_Flag_Existing Customer', axis=1)
Y = data['Attrition_Flag_Existing Customer']

#prepare for xgboost
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=420)

#init the model
model = xgb.XGBClassifier()

#fit the initial model
model.fit(X_train, y_train)

#predict the model
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
recall_score(y_test, y_pred)
precision_score(y_test, y_pred)

