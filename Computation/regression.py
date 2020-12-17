import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Computation.data import create_dataset


class LinearRegression:
    def __init__(self):
        self.w = None

    def simple_naive_regression(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(shape=(n_features, 1))
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        return self.w

    def ridge_regression(self, X, y, lmbda):
        n_samples, n_features = X.shape
        self.w = np.zeros(shape=(n_features, 1))
        t = np.dot(X.T, X) + lmbda * np.identity(n_features)
        self.w = np.dot(np.dot(np.linalg.inv(t), X.T), y)
        return self.w

    def lasso_regression(self, X, y, lmbda, literation_count):
        n_samples, n_features = X.shape
        self.w = np.zeros(shape=(n_features, 1))

        # Since bias is basically mean of original - predictions
        self.w[0] = np.sum(y - np.dot(X[:, 1:], self.w[1:])) / n_samples

        for i in range(literation_count):
            for j in range(1, n_features):
                copy_w = self.w.copy()
                copy_w[j] = 0.0
                # difference between observed and predicted values
                residue_val = y - np.dot(X, copy_w)
                term1 = np.dot(X[:, j], residue_val)
                term2 = lmbda * n_samples
                # As lasso inv_methodolves absolute value , we are using soft thresholding
                if term1 > 0.0 and term2 < abs(term1):
                    t = (term1 - term2)
                elif term1 < 0.0 and term2 < abs(term1):
                    t = (term1 + term2)
                else:
                    t = 0.0
                self.w[j] = t / (X[:, j] ** 2).sum()

        return self.w

    @staticmethod
    def predict(X, w):
        return np.dot(X, w)

    @staticmethod
    def error_calculation(X, y, w):
        hx_val = np.dot(X, w)
        error = 0
        for i in range(len(y)):
            error += (y[i] - hx_val[i]) ** 2

        error /= len(y)
        return error**0.5, hx_val


def plot_comparison(actual, predicted, features):
    plt.figure(figsize=(8, 6))
    actual_data = pd.DataFrame(pd.Series(actual), columns=['weight'])
    actual_data['type'] = pd.Series(['actual'] * len(actual))
    actual_data['feature'] = pd.Series(list(range(features)))
    actual_data = actual_data.iloc[1:]

    predicted_data = pd.DataFrame(pd.Series(predicted), columns=['weight'])
    predicted_data['type'] = pd.Series(['pred'] * len(predicted))
    predicted_data['feature'] = pd.Series(list(range(features)))
    predicted_data = predicted_data.iloc[1:]

    data = pd.concat([actual_data, predicted_data])

    sns.barplot(x="feature", y="weight", hue="type", data=data)


def generate_data(m):
    data = create_dataset(10000)
    X = np.asarray(data.iloc[:, :-1])
    y = np.asarray(data.iloc[:, -1:])

    return X, y


def error_estimation(w):
    error_estimation = 0
    for i in (range(10)):
        X, y = generate_data(10000)
        error_estimation += LinearRegression.error_calculation(X, y, w)

    return float(error_estimation / 10)


def varying_lambda(m):
    lmbda = np.arange(0, 0.3, 0.01)
    error_estimation_list = []

    for i in lmbda:
        print(round(i, 2), end="\t")
        error = 0
        data = create_dataset(m)
        X = np.asarray(data.iloc[:, :-1])
        y = np.asarray(data.iloc[:, -1:])
        l_r = LinearRegression()
        w = l_r.lasso_regression(X, y, i, 100)
        error_estimation_list.append(error_estimation(w))

    plt.figure(figsize=(10, 8))
    plt.plot(lmbda, error_estimation_list, marker='x')
    plt.title("True Error w.r.t. lambda")
    plt.xlabel("Lambda")
    plt.ylabel("True Error")
    plt.show()


def rel_features(new_data_val):
    relevant_features = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X10", "X11", "X12", "X13", "X15", "X16", "Y"]
    rel = [a[1:] for a in relevant_features[:-1]]
    rel_data = new_data_val[relevant_features]
    print(rel_data.head())

    X_rel = np.asarray(rel_data.iloc[:, :-1])
    y_rel = np.asarray(rel_data.iloc[:, -1:])

    return X_rel, y_rel, rel


def calculate_actual_weights(rel, w_actual):
    rel_actual_wt = []

    for i in range(1, 21):
        if str(i) in rel:
            rel_actual_wt.append(w_actual[i])
        else:
            rel_actual_wt.append(0)

    return rel_actual_wt


def calculate_w():
    w_actual = [10]
    for i in range(1, 21):
        if i <= 10:
            w_actual.append((0.6) ** i)
        else:
            w_actual.append(0)

    return w_actual


def print_plot(n, lmbda, weights_lasso, train_data):
    for i in range(n):
        plt.plot(lmbda, weights_lasso[1:][i], label=train_data.columns[1:][i])

    # plt.xscale('log')
    plt.xlabel('$\\lambda$')
    plt.ylabel('Coefficients')
    plt.title('Lasso Paths')
    plt.legend()
    plt.axis('tight')
    plt.show()


def execute():
    m = 1000

    train_data = create_dataset(m)
    train_data.head()
    print(train_data)

    l_r = LinearRegression()
    # fetches all the data except for the last column
    X = np.asarray(train_data.iloc[:, :-1])
    # fetches last column data
    y = np.asarray(train_data.iloc[:, -1:])
    w_actual = calculate_w()
    w_trained = l_r.simple_naive_regression(X, y)
    features = X.shape[1]
    plot_comparison(w_actual, w_trained.flatten(), features)
    print("Error: ", float(l_r.error_calculation(X, y, w_trained)))
    print("Trained Bias: ", w_trained[0])

    estimated_err = error_estimation(w_trained)
    print("True error: ", estimated_err)

    # Ridge regression

    lmbda = 0.01
    w_lval = l_r.ridge_regression(X, y, lmbda)

    estimated_err = error_estimation(w_lval)
    print(estimated_err)

    lmbda = np.arange(0, 1, 0.1)
    error_estimation_list = []

    for i in lmbda:
        print(round(i, 2), end='\t')
        error = 0
        X, y = generate_data(10000)
        l_r = LinearRegression()
        w = l_r.ridge_regression(X, y, i)
        error_estimation_list.append(error_estimation(w))

    plt.figure(figsize=(10, 8))
    plt.plot(lmbda, error_estimation_list, marker='x')
    plt.title("True Error w.r.t. lambda")
    plt.xlabel("Lambda")
    plt.ylabel("True Error")
    plt.show()

    w_lval = l_r.ridge_regression(X, y, 0.08)

    estimated_err = error_estimation(w_lval)

    plot_comparison(w_actual, w_lval.flatten(), features)
    print("Error: ", float(l_r.error_calculation(X, y, w_lval)))

    print("Weights (Bias at zero index): \n", w_lval)

    #     Lasso Model
    lmbda = np.arange(0, 1.0, 0.01)
    weights = []
    lin_reg = LinearRegression()

    for l in lmbda:
        print(round(l, 2), end='\t')
        weight = lin_reg.lasso_regression(X, y, l, 100)
        weights.append(weight.flatten())

    weights_lasso = np.stack(weights).T
    print(weights_lasso[1:].shape)

    n, _ = weights_lasso[1:].shape
    plt.figure(figsize=(20, 10))
    print_plot(n, lmbda, weights_lasso, train_data)

    varying_lambda(1000)
    optimal_weight = lin_reg.lasso_regression(X, y, 0.001, 1000)
    plot_comparison(w_actual, optimal_weight.flatten(), features)
    print("Error: ", float(lin_reg.error_calculation(X, y, optimal_weight)))

    new_data_val = create_dataset(1000)
    linreg = LinearRegression()
    X = np.asarray(new_data_val.iloc[:, :-1])
    y = np.asarray(new_data_val.iloc[:, -1:])
    new_weight = linreg.lasso_regression(X, y, 0.001, 100)

    plot_comparison(w_actual, new_weight.flatten(), features)

    print("Error: ", float(l_r.error_calculation(X, y, new_weight)))

    f_weights = dict(zip(*(new_data_val.drop(['Y'], 1).columns, new_weight)))
    print(f_weights)
    X_rel, y_rel, rel = rel_features(new_data_val)

    rel_actual_wt = calculate_actual_weights(rel, w_actual)
    new_r = LinearRegression()
    w_lval_2 = new_r.ridge_regression(X_rel, y_rel, 0.001)

    plot_comparison(rel_actual_wt, w_lval_2.flatten(), features)


if __name__ == '__main__':
    execute()
