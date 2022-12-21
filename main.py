
import pandas as pd
import numpy as np
import tensorflow as tf

print(tf.__version__)

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

print(X)
