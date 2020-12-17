import numpy as np


def inv_method(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]


def kernel(X1, X2):
    return (1 + np.dot(X1, X2)) ** 2


def new_alpha(m, k):
    new_alpha_values = np.array(m)
    new_alpha_values = np.reshape(new_alpha_values, (3, 1))
    new_alpha_values = list(new_alpha_values - 0.01 * k)
    k = 0
    for i in range(1, len(alpha_val)):
        k = k + (Y[i] * new_alpha_values[i - 1])
    k *= -1
    new_alpha_values.insert(0, k)
    return new_alpha_values


def newton_function(alpha_val, X, Y, eta_val):
    gradients_val = np.zeros((len(alpha_val) - 1, 1))
    alphay_sum = 0
    for i in range(1, len(alpha_val)):
        alphay_sum += alpha_val[i] * Y[i]
    for i in range(1, len(alpha_val)):
        teta_val_val = -eta_val * ((1 / alpha_val[i]) + Y[i] / alphay_sum)
        t0_val = (1 - Y[i] * Y[0])
        t1_val = kernel(X[0], X[0]) * 2 * alphay_sum * Y[i]
        t2_val = 0
        for j in range(1, len(alpha_val)):
            t2_val += alpha_val[j] * Y[j] * kernel(X[j], X[0])
        t3_val = kernel(X[i], X[0]) * alphay_sum
        t4_val = 0
        for j in range(1, len(alpha_val)):
            t4_val += alpha_val[j] * Y[j] * kernel(X[j], X[i])
        t4_val = t4_val * 2 * Y[i]
        gradients_val[i - 1] = teta_val_val - t0_val + 0.5 * (t1_val - 2 * Y[i] * (t2_val + t3_val) + t4_val)
    new_alpha_values = new_alpha(alpha_val[1:], gradients_val)

    return new_alpha_values


if __name__ == "__main__":
    X = [[-1, 1], [-1, -1], [1, 1], [1, -1]]
    Y = [1, -1, -1, 1]
    alpha_val = [5, 5, 5, 5]
    eta_val = 1
    for _ in range(250):
        alpha_val = newton_function(alpha_val, X, Y, eta_val)
        eta_val = eta_val / 2
        print(eta_val / 2, alpha_val[0], alpha_val[1], alpha_val[2], alpha_val[3])
