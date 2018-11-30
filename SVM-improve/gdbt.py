import numpy as np
import matplotlib.pyplot as plt
import toolbox
from sklearn.metrics.pairwise import rbf_kernel
from config import cfg
from sklearn import ensemble
from sklearn import svm
from sklearn.metrics import accuracy_score
import rbfclf

def gdbt(data,kernel):
    # data, label = toolbox.loadData("SVM-improve\ionosphere.data")
    # kernel = rbf_kernel(data)
    # model = svm.SVC(C = 1.0, kernel = 'precomputed')
    # model.fit(kernel, label)
    # predict = model.predict(kernel)
    # score = accuracy_score(label, predict)
    # print(score)
    length = int(data.shape[0] * (data.shape[0] + 1) / 2)
    X = np.mat(np.zeros((length, data.shape[1])))
    y = np.array(np.zeros(length))
    index = 0
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0], 1):
            X[index] = np.square(data[i] - data[j])
            y[index] = kernel[i, j]
            index += 1
    rbf = rbfclf.rbf()
    params = {'n_estimators': cfg.n_estimators, 'max_depth': cfg.max_depth, 'min_samples_split': cfg.min_samples_split,
                  'learning_rate': cfg.learning_rate, 'loss': 'ls','subsample' : 0.8, 'tol':1e-6}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X, y)
    predict = clf.predict(X)
    loss = 0
    for (k, j) in zip(predict, y):
        loss += (k - j) ** 2
    loss = loss / len(predict)
    print(loss)
    return clf
    # newkernel = np.mat(np.zeros(kernel.shape))
    # index = 0
    # for i in range(data.shape[0]):
    #     for j in range(i, data.shape[0], 1):
    #         newkernel[i, j] = predict[index]
    #         newkernel[j, i] = predict[index]
    #         index += 1
    # print(score)
    # plt.plot(X, predict, 'b-o',label = str(i),linewidth=0.5)
    # plt.plot(X,y,'r-o',label="$sin(x)$",linewidth=1)
    # plt.show()

def kernel_function(clf, train_data, test_data):
    test_kernel = np.mat(np.zeros((test_data.shape[0],train_data.shape[0])))
    for i in range(test_data.shape[0]):
        for j in range(train_data.shape[0]):
            x = np.square(test_data[i] - train_data[j])
            test_kernel[i, j] = clf.predict(x)
    return test_kernel

# rbf = rbfclf.rbf()
# X = np.linspace(0,100,100)
# X = np.mat(X).reshape((100,1))
# y = rbf.predict(X)
# # y = y + np.square(X)
# params = {'init':rbf,'n_estimators': 50, 'max_depth': cfg.max_depth, 'min_samples_split': cfg.min_samples_split,
#                   'learning_rate': cfg.learning_rate, 'loss': 'ls'}
# clf = ensemble.GradientBoostingRegressor(**params)
# ensemble.GradientBoostingRegressor()
# clf.fit(X, y)
# predict = clf.predict(X)
# print(clf.predict(np.mat([200])))
# print(rbf.predict(np.mat([200])))
# loss = 0
# for (k, j) in zip(predict, y):
#     loss += (k - j) ** 2
# loss = loss / len(predict)
# print(loss)
# plt.plot(X, predict, 'b-o',linewidth=0.5)
# plt.plot(X,y,'r-o',linewidth=1)
# plt.show()