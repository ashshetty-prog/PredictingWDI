

import pandas as pd
import seaborn as sns
import sklearn.linear_model as sklm
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import utils

from sklearn.neighbors import KNeighborsClassifier

import scipy.spatial
from collections import Counter

from sklearn import datasets
from sklearn.model_selection import train_test_split
#
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)
# KNeighborsClassifier(...)
# print(neigh.predict([[1.1]]))
# [0]
# >>> print(neigh.predict_proba([[0.9]]))
# [[0.66666667 0.33333333]]


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):
        distance = scipy.spatial.distance.euclidean(X1, X2)

    def predict(self, X_test):
        result = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(X_train)):
                # calculates Eucledian distance
                dist = scipy.spatial.distance.euclidean(X_train[j], X_test[i])
                d.append([dist, j])
            # sorts and picks the nearest one
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(y_train[j])
            ans = Counter(votes).most_common(1)[0][0]
            result.append(ans)

        return result






if __name__ == '__main__':

    xl = pd.ExcelFile("/Users/siri/PycharmProjects/BIC_2/src/WDIdata3decades.xlsx")
    print(xl.sheet_names)

    df = xl.parse('WDIdata3decades')
    print(df.head())

    emp_df = df.filter(regex=("^SL.EMP*"))
    print(emp_df.shape)
    print(emp_df.count())

    print(emp_df.columns)

    # Compute correlation matrix. Only select subset of features with a correlation above threshold.
    def correlation(data):
        corr_matrix = data.corr()
        plt.figure(figsize=(50, 50))
        sns.heatmap(corr_matrix, linewidths=0.1, vmax=1.0,
                    square=True, cmap=sns.diverging_palette(20, 220, n=200), linecolor='white', annot=True)
        plt.show()
        corr_features = []
        for i in range(len(corr_matrix.columns)):
            features = []
            for j in range(len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= 0.4:
                    # print(corr_matrix.columns[j])
                    features.append(corr_matrix.columns[j])
            corr_features.append(features)
        return corr_features


    correlated_features = correlation(emp_df)
    print(correlated_features)

    '''
    To predict a feature X we find other features that are highly correlated with X. corr_feature_df is the subset of data 
    with X and its correlated features. 
    X_train: Dataset with non-null values of feature X. Drop X and fill in the null values with train_data.mean()
    X_test: Dataset with null values of feature X. Drop X and fill in the null values with train_data.mean()
    y_train: rows with non null values of X
    y_test: rows with null values of X

    For every feature we apply KNN by calculating distance (Euclidean) between a test data point and every training data point. This is t
     to see who is closer. We then sort the distance and pick the nearest neighbors to our given test data point.

    '''
    for i, col in enumerate(emp_df.columns):

        emp_df_copy = emp_df.copy()
        print('feature', col)
        corr_feature_df = emp_df_copy[correlated_features[i]]



        test_data = corr_feature_df[corr_feature_df[col].isnull()]
        print('test data', test_data)

        # corr_feature_df.dropna(inplace=True)
        # label encoding to represent categorical column in numerical column
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(corr_feature_df[col])
        print(corr_feature_df)
        y_train = pd.DataFrame(encoded)

        y_train = y_train.fillna(y_train.mean())



        # y_train = corr_feature_df[col]
        # y_train = y_train.fillna(y_train.mean())

        X_train = corr_feature_df.drop(col, axis=1)
        X_train = X_train.fillna(X_train.mean())


        X_test = test_data.drop(col, axis=1)

        # X_test.dropna(inplace=True)
        X_test = X_test.fillna(X_train.mean())

        print('X test', X_test)

        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X_train, y_train)

        # Training accuracy

        print('accuracy', neigh.score(X_train, y_train))


        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        y_train = y_train.ravel()
        X_test = X_test.to_numpy()
        # y_test = y_test.to_numpy()

        clf = KNN(1)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        for i in prediction:
            print(i, end=' ')



        # Below predicted values are similar to the ones obtained using Sklearn kmeans function
        y_pred1 = neigh.predict(X_test)




        print('col', col, 'prediction', prediction)
        print(test_data.index.values)
        for c, r in enumerate(test_data.index.values):
            emp_df.iloc[r][col] = prediction[c]









