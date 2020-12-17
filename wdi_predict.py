# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Computation.regression import LinearRegression
from knn import KNN


class StepByStepRegression:
    def __init__(self, df: pd.DataFrame, method):
        self.df = df
        self.sim_fun = method
        self.norm_constants = {}

    def normalize(self):
        for col in self.df.columns:
            self.norm_constants[col] = {
                "min": self.df[col].min(),
                "max": self.df[col].max()
            }
            self.df[col] = (self.df[col] - self.norm_constants[col]["min"]) / (
                    self.norm_constants[col]["max"] - self.norm_constants[col]["min"])

    def un_normalize(self):
        for col in self.df.columns:
            self.df[col] = self.df[col] * (self.norm_constants[col]["max"] - self.norm_constants[col]["min"]) + \
                           self.norm_constants[col]["min"]

    # Compute correlation matrix. Only select subset of features with a correlation above threshold.
    @staticmethod
    def correlation(data, show_plot=False):
        corr_matrix = data.corr()
        if show_plot:
            plt.figure(figsize=(50, 50))
            sns.heatmap(corr_matrix, linewidths=0.1, vmax=1.0,
                        square=True, cmap=sns.diverging_palette(20, 220, n=200), linecolor='white', annot=True)
            plt.show()
        corr_features = dict()
        for i in range(len(corr_matrix.columns)):
            features = []
            for j in range(len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= 0.2:
                    # print(corr_matrix.columns[j])
                    features.append(corr_matrix.columns[j])
            corr_features[corr_matrix.columns[i]] = features
        return corr_features

    @staticmethod
    def get_least_nan_columns(df):
        feature_df = df.iloc[:, :]
        null_cols = feature_df.isna().sum().sort_values()
        return null_cols

    def train_test_interpolate_split(self, df, col):
        non_null_df = df[~df[col].isnull()]
        split_point = int(0.80 * non_null_df.shape[0])
        null_df = df[df[col].isnull()]
        train_x = non_null_df.iloc[:split_point, non_null_df.columns != col]
        train_y = non_null_df.iloc[:split_point, non_null_df.columns == col]
        test_x = non_null_df.iloc[split_point:, non_null_df.columns != col]
        test_y = non_null_df.iloc[split_point:, non_null_df.columns == col]
        interpolate_x = null_df.iloc[:, null_df.columns != col]

        return train_x, train_y, test_x, test_y, interpolate_x

    def fill_missing_data_step_by_step(self, sorted_df):
        num_nans = sorted_df.isnull().sum()
        accepted_cols = []
        for i, col in enumerate(sorted_df.columns):
            if num_nans[col] == 0:
                accepted_cols.append(col)
                continue

            accepted_cols.append(col)
            sorted_df_copy = sorted_df.copy()
            print('feature', col)
            accepted_df = sorted_df_copy[accepted_cols]

            train_x, train_y, test_x, test_y, i_x = self.train_test_interpolate_split(accepted_df, col)
            model = LinearRegression()
            w_trained = model.simple_naive_regression(np.asarray(train_x), np.asarray(train_y))
            print('trained weights', w_trained)

            err, y_pred = model.error_calculation(np.asarray(test_x), np.asarray(test_y), w_trained)
            print("RMS Error: ", err)
            # print('actual', test_y)
            # print('pred', y_pred)
            # print('test data predictions', np.dot(X_test, w_trained))
            for k, inter_x in i_x.iterrows():
                sorted_df.at[k, col] = model.predict(np.asarray(inter_x), w_trained)

    @staticmethod
    def fill_missing_data(sorted_df, correlated_features):
        """
        To predict a feature X we find other features that are highly correlated with X. corr_feature_df is the subset
        of data with X and its correlated features.
        X_train: Dataset with non-null values of feature X. Drop X and fill in the null values with train_data.mean()
        X_test: Dataset with null values of feature X. Drop X and fill in the null values with train_data.mean()
        y_train: rows with non null values of X
        y_test: rows with null values of X

        For every feature apply linear regression and predict the missing values of that feature. Copy it to the
        original dataset so that the values of that feature can be used to make prediction for the remaining features

        """

        for col in sorted_df.columns:
            sorted_df_copy = sorted_df.copy()
            print('feature', col)
            corr_feature_df = sorted_df_copy[correlated_features[col]]

            test_data = corr_feature_df[corr_feature_df[col].isnull()]
            print('test data', test_data)
            if test_data.empty:
                continue
            # corr_feature_df.dropna(inplace=True)

            # print(corr_feature_df)
            y_train = corr_feature_df[col]
            y_train = y_train.fillna(y_train.mean())

            X_train = corr_feature_df.drop(col, axis=1)
            X_train = X_train.fillna(X_train.mean())
            # print('X train', X_train[:5])

            X_test = test_data.drop(col, axis=1)

            # X_test.dropna(inplace=True)
            X_test = X_test.fillna(X_train.mean())
            # print('X test', X_test)

            # model = sklm.LinearRegression()
            model = LinearRegression()
            # model.fit(X_train, y_train)
            X = np.asarray(X_train)
            # fetches last column data
            y = np.asarray(y_train)
            w_trained = model.simple_naive_regression(X, y)
            print('trained weights', w_trained)

            # Training accuracy
            # print('accuracy', model.score(X_train, y_train))

            # y_pred = model.predict(X_test)
            err, y_pred = model.error_calculation(X, y, w_trained)
            print("RMS Error: ", err)
            print('actual', y)
            print('pred', y_pred)
            # print('test data predictions', np.dot(X_test, w_trained))
            for c, r in enumerate(test_data.index.values):
                sorted_df.at[r, col] = y_pred[c]


def execute():
    df = pd.read_csv('data/reduced_dataset_v3.csv')

    df = df.drop(['Unnamed: 0', 'Time', 'Time Code', 'Country Name', 'Country Code'], axis=1)
    sbs_reg = StepByStepRegression(df, 'correlation')
    # print(emp_df.columns)
    normalized_df = sbs_reg.normalize()
    # print(normalized_df.head())

    sorted_columns = sbs_reg.get_least_nan_columns(normalized_df)
    # print('sorted columns', sorted_columns)
    sorted_df = pd.DataFrame()
    correlated_features_list = dict()
    for col, nulls in sorted_columns.iteritems():
        sorted_df[col] = normalized_df[col]
    # print('sorted dataframe', sorted_df.columns)

    if sbs_reg.sim_fun == 'KNN':
        knn = KNN(sorted_df, 3)
        correlated_features_list = knn.findKNeighbors()
    elif sbs_reg.sim_fun == 'correlation':
        correlated_features_list = sbs_reg.correlation(sorted_df)
    print('correlated features', correlated_features_list)
    sbs_reg.fill_missing_data(sorted_df, correlated_features_list)
    # %%


def execute_2():
    df = pd.read_csv('data/reduced_dataset_v3.csv')
    df = df.drop(['Unnamed: 0', 'Time', 'Time Code', 'Country Name', 'Country Code'], axis=1)
    sbs_reg = StepByStepRegression(df, 'correlation')

    sbs_reg.normalize()
    sorted_columns = sbs_reg.get_least_nan_columns(sbs_reg.df)
    sorted_df = pd.DataFrame()
    for col, nulls in sorted_columns.iteritems():
        sorted_df[col] = sbs_reg.df[col]
    sbs_reg.fill_missing_data_step_by_step(sorted_df)
    sbs_reg.un_normalize()
    print(sbs_reg.df)

if __name__ == '__main__':
    execute_2()
