import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Put the dataset here
dataset = pd.read_csv('Data.csv')

# print(dataset)

#Get all the columns except the last one
X = dataset.iloc[:, :-1].values
#Index of thhe last column
Y = dataset.iloc[:, 3].values

# print(X)
# print(Y)

#Missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
#Put the index to fit imputer object
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
