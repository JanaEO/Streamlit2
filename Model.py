import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

#importing data
data_bytes = pd.read_csv('/Users/janaaloud/Desktop/Marketing Campaign Data1.csv')
data_bytes = data_bytes[['Education','Marital_Status', 'Income','Kidhome', 'Teenhome','Recency','Response']].copy()

data_bytes = data_bytes.copy()
#replacing missing values
data_bytes.fillna(0, inplace=True)
#encoding categorical variables
encode = ['Education','Marital_Status']
for col in encode:

    dummy = pd.get_dummies(data_bytes[col], prefix=col)

    data_bytes = pd.concat([data_bytes, dummy], axis=1)

    del data_bytes[col]

#Creating X and Y
X = data_bytes.drop('Response', axis=1)
Y = data_bytes['Response']
#defining random forest classifier
clf = RandomForestClassifier()
clf.fit(X, Y)
#Creating a pickle file
file = open("/Users/janaaloud/Desktop/response_clf.pkl", "wb")
pickle.dump(clf , file)
file.close()
model = open("/Users/janaaloud/Desktop/response_clf.pkl", "rb")
forest = pickle.load(model)
