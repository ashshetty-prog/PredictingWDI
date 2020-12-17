
# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import seaborn as sns
from copy import copy

def generate_data(m):
    Xval=[]
    Yval=[]
    for _ in range(m):
        # Creating X (feature) vectors for the data
        X=[1]
        X+=list(np.random.normal(size=10))
        X.append(X[1]+X[2]+np.random.normal(scale=0.1**0.5))
        X.append(X[3]+X[4]+np.random.normal(scale=0.1**0.5))
        X.append(X[4]+X[5]+np.random.normal(scale=0.1**0.5))
        X.append(0.1*X[7]+np.random.normal(scale=0.1**0.5))
        X.append(2*X[2]-10+np.random.normal(scale=0.1**0.5))
        X+=list(np.random.normal(size=5))
        multipliers=[0.6**(i) for i in range(1,11)]
        # Creating target column for the data
        Y=10+np.dot(multipliers, X[1:11])+np.random.normal(scale=0.1**0.5)
        Xval.append(X)
        Yval.append(Y)
    Xval=np.array(Xval)
    Xval=np.reshape(Xval,(m,21))
    Yval=np.array(Yval)
    Yval=np.reshape(Yval,(m,1))
    return Xval,Yval


# Combining data points into a dataframe
def create_dataset(m):
    X,y=generate_data(1000)
    # Training data combining x and y
    data = pd.DataFrame(np.append(X, y, axis=1), columns=["X" + str(i) for i in range(21)] + ['Y'])
    data['X0'] = 1
    return data


if __name__ == '__main__':
    m = 1000

    train_data = create_dataset(m)
    train_data.head()
    print(train_data)


