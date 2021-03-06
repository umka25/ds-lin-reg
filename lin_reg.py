import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def gradient_descent(X, Y, w, learning_rate=1e-6):
    for i in range(1000):
        y_pred = X.dot(w)
        e = y_pred - Y
        w = w - learning_rate * X.T.dot(e)
    return w

#test
solutionVar = 1

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
dataset = pd.read_csv('Features_Variant_1.csv', names=columns_name)
dataset = dataset.sample(frac=1)

row_count = dataset.shape[0]
dataset_feature = dataset.iloc[0: row_count, 0: feature_column_size].values
dataset_target = np.asarray(dataset.iloc[0: row_count, feature_column_size])

dataset_feature = np.asarray(np.asmatrix(pd.DataFrame(StandardScaler().fit_transform(dataset_feature))))
dataset_feature = np.hstack((dataset_feature, np.ones((row_count, 1))))

for i in range(0, row_size, step):
    next_i = i + step

    target_test = dataset_target[i: next_i]
    feature_test = dataset_feature[i: next_i, 0: common_column_size]

    target_train = dataset_target[np.r_[0: i, next_i: row_size]]
    feature_train = dataset_feature[np.r_[0: i, next_i: row_size], 0: common_column_size]

    # аналитическое решение
    if solutionVar == 0:
        m = np.linalg.lstsq(feature_train.T.dot(feature_train), feature_train.T.dot(target_train), rcond=None)[0]
    else:
        # решение через градиентный спуск
        m = np.random.randn(common_column_size) / np.sqrt(common_column_size)
        m = gradient_descent(feature_train, target_train, m)

    y_pred = np.asarray(feature_train.dot(m))
    r2_train.append(r2_score(target_train, y_pred))
    rmse_train.append(mean_squared_error(target_train, y_pred, squared=False))

    y_pred = np.asarray(feature_test.dot(m))
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

w_feature_index = [[], [], [], [], []]
for i in range(0, len(w)):
    for j in range(0, len(w[i])):
        if w[i][j] > 9e-2:
            w_feature_index[i].append(j)

w_f = list((set(w_feature_index[0])
            & set(w_feature_index[1]))
           & set(w_feature_index[2])
           & set(w_feature_index[3])
           & set(w_feature_index[4]))

table = pd.DataFrame(np.array([rmse_train, rmse_test, r2_train, r2_test]),
                     columns=('F1', 'F2', 'F3', 'F4', 'F5', 'E', 'SD'))

# f = open("README.md", 'a')
# f.write(table.to_string())
# f.write(str(w_f))
# f.close()
