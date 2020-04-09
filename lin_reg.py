import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def gradient_descent(X, Y, w, learning_rate=1e-6):
    x = np.asarray(np.asmatrix(X))
    y = np.asarray(Y)
    w = np.asarray(w)
    for i in range(1000):
        y_pred = x.dot(w)
        e = y_pred - y
        w = w - learning_rate * x.T.dot(e)
    return w


row_size = 5000
common_column_size = 54
feature_column_size = 53
fold_count = 5
step = int(row_size / fold_count)
rmse_train = []
r2_train = []
rmse_test = []
r2_test = []
w = []

columns_name = []
for i in range(0, common_column_size):
    columns_name.append(i)
dataset = pd.read_csv('Features_TestSet.csv', names=columns_name)
dataset = pd.DataFrame(StandardScaler().fit_transform(dataset))

for i in range(0, row_size, step):
    next_i = i + step

    target_test = dataset.iloc[i: next_i, feature_column_size].to_numpy()
    feature_test = dataset.iloc[i: next_i, 0: feature_column_size]

    target_train = dataset.iloc[np.r_[0: i, next_i: row_size], feature_column_size].to_numpy()
    feature_train = dataset.iloc[np.r_[0: i, next_i: row_size], 0: feature_column_size]

    # аналитическое решение
    #m = np.linalg.lstsq(feature_train.T.dot(feature_train), feature_train.T.dot(target_train), rcond=None)[0]

    # решение через градиентный спуск
    m = np.random.randn(feature_column_size) / np.sqrt(feature_column_size)
    m = gradient_descent(feature_train, target_train, m)

    y_pred = np.array(feature_train.dot(m))
    r2_train.append(r2_score(target_train, y_pred))
    rmse_train.append(mean_squared_error(target_train, y_pred, squared=False))

    y_pred = np.array(feature_test.dot(m))
    r2_test.append(r2_score(target_test, y_pred))
    rmse_test.append(mean_squared_error(target_test, y_pred, squared=False))

    w.append(m)

    plt.plot(target_test, label='test')
    plt.plot(y_pred, label='pred')
    plt.legend()
    plt.show()

rmse_train.append(sum(rmse_train) / fold_count)
rmse_train.append(np.std(rmse_train))
r2_train.append(sum(r2_train) / fold_count)
r2_train.append(np.std(r2_train))
rmse_test.append(sum(rmse_test) / fold_count)
rmse_test.append(np.std(rmse_test))
r2_test.append(sum(r2_test) / fold_count)
r2_test.append(np.std(r2_test))

w_mean = []
w_std = []

for i in range(0, len(w[0])):
    t = [w[0][i], w[1][i], w[2][i], w[3][i], w[4][i]]
    w_mean.append(sum(t) / fold_count)
    w_std.append(np.std(t))

w.append(w_mean)
w.append(w_std)

w_feature_index = [[], [], [], [], [], [], []]
for i in range(0, len(w)):
    for j in range(0, len(w[i])):
        if w[i][j] > 9e-2:
            w_feature_index[i].append(j)

table = pd.DataFrame(np.array([rmse_train, rmse_test, r2_train, r2_test, w_feature_index]),
                     columns=('F1', 'F2', 'F3', 'F4', 'F5', 'E', 'SD'))
