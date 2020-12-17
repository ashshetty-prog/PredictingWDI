import pandas as pd

import sklearn.linear_model as sklm


# Compute correlation matrix. Only select subset of features with a correlation above threshold.
def correlation(data):
    corr_matrix = data.corr()
    # plt.figure(figsize=(50, 50))
    # sns.heatmap(corr_matrix, linewidths=0.1, vmax=1.0,
    #             square=True, cmap=sns.diverging_palette(20, 220, n=200), linecolor='white', annot=True)
    # plt.show()
    corr_features = []
    for i in range(len(corr_matrix.columns)):
        features = []
        for j in range(len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= 0.4:
                # print(corr_matrix.columns[j])
                features.append(corr_matrix.columns[j])
        corr_features.append(features)
    return corr_features


def execute(df, correlated_features):
    for i, col in enumerate(df.columns):
        emp_df_copy = df.copy()
        print('feature', col)
        corr_feature_df = emp_df_copy[correlated_features[i]]

        test_data = corr_feature_df[corr_feature_df[col].isnull()]
        print('test data', test_data)

        # corr_feature_df.dropna(inplace=True)

        print(corr_feature_df)
        y_train = corr_feature_df[col]
        y_train = y_train.fillna(y_train.mean())

        X_train = corr_feature_df.drop(col, axis=1)
        X_train = X_train.fillna(X_train.mean())

        X_test = test_data.drop(col, axis=1)

        # X_test.dropna(inplace=True)
        X_test = X_test.fillna(X_train.mean())
        print('X test', X_test)

        model = sklm.LinearRegression()
        model.fit(X_train, y_train)

        # Training accuracy
        print('accuracy', model.score(X_train, y_train))

        y_pred = model.predict(X_test)
        print('col', col, 'prediction', y_pred)
        print(test_data.index.values)
        for c, r in enumerate(test_data.index.values):
            df.iloc[r][col] = y_pred[c]


class StepByStepRegression:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_least_nan_columns(self, feature_index_start):
        features_df = self.df.iloc[:, feature_index_start:]
        null_cols = features_df.isna().sum().sort_values()
        for c_name, nulls in null_cols.iteritems():
            print(c_name, nulls)


if __name__ == '__main__':
    df1 = pd.read_csv("data/reduced_dataset_v3.csv").iloc[:, 1:]
    sbs = StepByStepRegression(df1)
    sbs.get_least_nan_columns(feature_index_start=4)
