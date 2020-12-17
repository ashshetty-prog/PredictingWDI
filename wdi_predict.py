# %%

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Computation.regression import LinearRegression


class StepByStepRegression:
    def __init__(self, df: pd.DataFrame, method="correlation"):
        self.df = df
        self.normalized_df = self.df.copy()
        self.sim_fun = method
        self.norm_constants = {}

    def normalize(self):
        for col in self.df.columns:
            self.norm_constants[col] = {
                "min": self.df[col].min(),
                "max": self.df[col].max()
            }
            self.normalized_df[col] = (self.df[col] - self.norm_constants[col]["min"]) / (
                    self.norm_constants[col]["max"] - self.norm_constants[col]["min"])

    def un_normalize(self, df):
        for col in df.columns:
            df[col] = df[col] * (self.norm_constants[col]["max"] - self.norm_constants[col]["min"]) + \
                      self.norm_constants[col]["min"]
        return df

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

    def fill_missing_data_mean(self, df, plot_rmse=False):
        rmse_list = []
        rmse_dict = {}
        for col in df.columns:
            train_data = df[~df[col].isnull()]
            test_data = df[df[col].isnull()]
            if test_data.empty:
                continue
            y_train = train_data[col]
            y_train = y_train.fillna(y_train.mean())

            X_train = train_data.drop(col, axis=1)
            X_train = X_train.fillna(X_train.mean())
            train_x, train_y, validation_x, validation_y = self.train_validation_split(X_train, y_train)
            v_x = np.asarray(validation_x)
            v_y = np.asarray(validation_y)
            pred_y = df[col].mean()
            df[col].fillna(value=pred_y, inplace=True)
            rmse = 0
            for y in v_y:
                rmse += (y - pred_y) ** 2
            err = (rmse / len(v_y)) ** 0.5
            rmse_list.append(err)
            rmse_dict[col] = err
        rmse_avg = sum(rmse_list) / len(rmse_list)
        if plot_rmse:
            self.plot_rmse(rmse_dict, "Basic Mean Computation")
        return df, rmse_avg

    def train_test_interpolate_split(self, df, col):
        non_null_df = df[~df[col].isnull()]
        split_point = int(0.95 * non_null_df.shape[0])
        null_df = df[df[col].isnull()]
        train_x = non_null_df.iloc[:split_point, non_null_df.columns != col]
        train_y = non_null_df.iloc[:split_point, non_null_df.columns == col]
        test_x = non_null_df.iloc[split_point:, non_null_df.columns != col]
        test_y = non_null_df.iloc[split_point:, non_null_df.columns == col]
        interpolate_x = null_df.iloc[:, null_df.columns != col]

        return train_x, train_y, test_x, test_y, interpolate_x

    def train_validation_split(self, x, y, train_size=0.90):
        split_point = int(x.shape[0] * train_size)
        train_x = x.iloc[:split_point]
        valid_x = x.iloc[split_point:]
        train_y = y.iloc[:split_point]
        valid_y = y.iloc[split_point:]
        return train_x, train_y, valid_x, valid_y

    def plot_rmse(self, rmse_dict, model_name):
        print("Legend for RMSE plots:")
        for i, key in enumerate(rmse_dict.keys()):
            print("F{}: {}".format(i, key))
        plt.title("RMSE for various interpolating features using {}".format(model_name))
        plt.bar(rmse_dict.keys(), rmse_dict.values())
        plt.xticks(list(rmse_dict.keys()), ["F" + str(i) for i in range(len(rmse_dict))], rotation=90)
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    def fill_missing_data_step_by_step(self, sorted_df, show_predictions=False, plot_rmse=False):
        num_nans = sorted_df.isnull().sum()
        accepted_cols = []
        rmse_list = []
        rmse_dict = {}
        sum_weights = [0 for _ in range(len(sorted_df.columns.values))]
        n_considerations = [0 for _ in range(len(sorted_df.columns.values))]
        for i, col in enumerate(sorted_df.columns):
            if num_nans[col] == 0:
                accepted_cols.append(col)
                continue

            accepted_cols.append(col)
            sorted_df_copy = sorted_df.copy()
            accepted_df = sorted_df_copy[accepted_cols]

            train_x, train_y, test_x, test_y, i_x = self.train_test_interpolate_split(accepted_df, col)
            model = LinearRegression()
            w_trained = model.simple_naive_regression(np.asarray(train_x), np.asarray(train_y))
            for k in range(len(w_trained)):
                sum_weights[k] = w_trained[k][0]
                n_considerations[k] += 1
            err, y_pred = model.error_calculation(np.asarray(test_x), np.asarray(test_y), w_trained)
            rmse_list.append(err[0])
            rmse_dict[col] = err[0]
            if show_predictions:
                print("RMS Error: ", err)
                for k, pred in enumerate(y_pred):
                    if k % 50 == 0:
                        print("Actual: {}, Predicted: {}".format(test_y.iloc[k].values[0], pred))
            for k, inter_x in i_x.iterrows():
                sorted_df.at[k, col] = model.predict(np.asarray(inter_x), w_trained)

        if plot_rmse:
            self.plot_rmse(rmse_dict, "Step by step regression")
        for i in range(len(sum_weights)):
            if n_considerations[i] == 0:
                sum_weights[i] = 0
                continue
            sum_weights[i] = sum_weights[i] / n_considerations[i]
        if plot_rmse:
            plt.bar(["F" + str(i) for i in range(len(sum_weights))], sum_weights)
            plt.xticks(rotation=90)
            plt.show()
        return sum(rmse_list) / len(rmse_list)

    def fill_missing_data(self, sorted_df, correlated_features, show_predictions=False, plot_rmse=False):
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
        rmse_list = []
        rmse_dict = {}
        for col in sorted_df.columns:
            sorted_df_copy = sorted_df.copy()
            corr_feature_df = sorted_df_copy[correlated_features[col]]
            train_data = corr_feature_df[~corr_feature_df[col].isnull()]
            test_data = corr_feature_df[corr_feature_df[col].isnull()]
            if test_data.empty:
                continue
            # corr_feature_df.dropna(inplace=True)

            # print(corr_feature_df)
            y_train = train_data[col]
            y_train = y_train.fillna(y_train.mean())

            X_train = train_data.drop(col, axis=1)
            X_train = X_train.fillna(X_train.mean())
            train_x, train_y, validation_x, validation_y = self.train_validation_split(X_train, y_train)
            v_x = np.asarray(validation_x)
            v_y = np.asarray(validation_y)
            X_test = test_data.drop(col, axis=1)
            X_test = X_test.fillna(X_train.mean())
            model = LinearRegression()
            X = np.asarray(train_x)
            # fetches last column data
            y = np.asarray(train_y)
            w_trained = model.simple_naive_regression(X, y)
            err, y_pred = model.error_calculation(v_x, v_y, w_trained)
            rmse_list.append(err)
            rmse_dict[col] = err
            if show_predictions:
                print("RMS Error: ", err)
                for k in range(len(v_y)):
                    if k % 50 == 0:
                        print("Actual: {}, Predicted: {}".format(v_y[k], y_pred[k]))
            # print('test data predictions', np.dot(X_test, w_trained))

            for k, inter_x in X_test.iterrows():
                sorted_df.at[k, col] = model.predict(np.asarray(inter_x), w_trained)

        if plot_rmse:
            self.plot_rmse(rmse_dict, "Correlation regression")
        return sum(rmse_list) / len(rmse_list)


def init_dataset_and_models():
    df = pd.read_csv('data/reduced_dataset_v3.csv')

    df = df.drop(['Unnamed: 0', 'Time', 'Time Code', 'Country Name', 'Country Code'], axis=1)
    sbs_reg = StepByStepRegression(df, 'correlation')
    # print(emp_df.columns)
    sbs_reg.normalize()
    return sbs_reg


def execute():
    sbs_reg = init_dataset_and_models()
    sorted_columns = sbs_reg.get_least_nan_columns(sbs_reg.normalized_df)
    sorted_df = pd.DataFrame()
    for col, nulls in sorted_columns.iteritems():
        sorted_df[col] = sbs_reg.normalized_df[col]
    correlated_features_list = sbs_reg.correlation(sorted_df, show_plot=False)
    rmse_final = sbs_reg.fill_missing_data(sorted_df, correlated_features_list, show_predictions=False, plot_rmse=False)
    un_normalized = sbs_reg.un_normalize(sorted_df)
    print("The final RMSE for regression based interpolation is: {}".format(round(rmse_final, 3)))
    return un_normalized
    # %%


def execute_2():
    sbs_reg = init_dataset_and_models()
    sorted_columns = sbs_reg.get_least_nan_columns(sbs_reg.normalized_df)
    sorted_df = pd.DataFrame()
    for col, nulls in sorted_columns.iteritems():
        sorted_df[col] = sbs_reg.normalized_df[col]
    rmse_final = sbs_reg.fill_missing_data_step_by_step(sorted_df, show_predictions=False, plot_rmse=False)
    un_normalized = sbs_reg.un_normalize(sorted_df)
    print("The final RMSE for step by step regression interpolation is: {}".format(round(rmse_final, 3)))
    return un_normalized
    # print(un_normalized)


def execute_with_generated_data_correlation():
    sbs_reg = init_dataset_and_models()
    random_row_num = random.randint(0, sbs_reg.df.shape[0])
    for col in sbs_reg.normalized_df.columns.values.tolist()[5:]:
        sbs_reg.normalized_df.at[random_row_num, col] = np.NaN
    a = sbs_reg.df.iloc[random_row_num]
    sorted_df = pd.DataFrame()
    sorted_columns = sbs_reg.get_least_nan_columns(sbs_reg.normalized_df)
    for col, nulls in sorted_columns.iteritems():
        sorted_df[col] = sbs_reg.normalized_df[col]
    correlated_features_list = sbs_reg.correlation(sorted_df, show_plot=False)
    rmse_final = sbs_reg.fill_missing_data(sorted_df, correlated_features_list, show_predictions=False, plot_rmse=False)
    un_normalized = sbs_reg.un_normalize(sorted_df)
    b = un_normalized.iloc[random_row_num]
    c = pd.concat([a, b], axis=1)
    pd.options.display.float_format = '{:.2f}'.format
    print(c)


def execute_with_generated_data_sbs():
    sbs_reg = init_dataset_and_models()
    random_row_num = random.randint(0, sbs_reg.df.shape[0])
    for col in sbs_reg.normalized_df.columns.values.tolist()[5:]:
        sbs_reg.normalized_df.at[random_row_num, col] = np.NaN
    a = sbs_reg.df.iloc[random_row_num]
    sorted_df = pd.DataFrame()
    sorted_columns = sbs_reg.get_least_nan_columns(sbs_reg.normalized_df)
    for col, nulls in sorted_columns.iteritems():
        sorted_df[col] = sbs_reg.normalized_df[col]
    rmse_final = sbs_reg.fill_missing_data_step_by_step(sorted_df, show_predictions=False, plot_rmse=False)
    un_normalized = sbs_reg.un_normalize(sorted_df)
    b = un_normalized.iloc[random_row_num]
    c = pd.concat([a, b], axis=1)
    pd.options.display.float_format = '{:.2f}'.format
    print(c)


def basic_mean_prediction():
    sbs_reg = init_dataset_and_models()
    df, rmse_final = sbs_reg.fill_missing_data_mean(sbs_reg.normalized_df, plot_rmse=False)
    print("The final RMSE for Basic mean interpolation is: {}".format(round(rmse_final, 3)))
    un_normalized = sbs_reg.un_normalize(df)
    return un_normalized


if __name__ == '__main__':
    execute()
    execute_2()
    basic_mean_prediction()
    # execute_with_generated_data_sbs()
