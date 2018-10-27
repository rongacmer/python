#encoding=utf-8
import numpy as np
import time
from sklearn import ensemble
from config import cfg
# from sklearn import cross_validation
# from sklearn.metrics import roc_auc_score
# from sklearn.utils import shuffle
# from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import os


def adaboost(X_train, y_train, test_data, test_label, supportVectors, alphasy, b):
    print("进入adaboost阶段")
    params = {'n_estimators': cfg.n_estimators, 'max_depth': cfg.max_depth, 'min_samples_split': cfg.min_samples_split,
              'learning_rate': cfg.learning_rate, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    print("没有出现警告!")
    if os.path.exists('train_model.pkl'):
        os.remove('train_model.pkl')
        joblib.dump(clf, 'train_model.pkl', compress=3)
    else:
        joblib.dump(clf, 'train_model.pkl', compress=3)
    test_acc = []
    matchCount = 0
    for i in range(test_data.shape[0]):
        test_data_bp = np.zeros([1, 68])
        BP_predictkernel = np.mat(np.zeros((len(supportVectors), 1)))
        # 预测每一个样本
        for j in range(len(supportVectors)):
            test_data_bp[0, 0] = float(test_data[i, 0])
            test_data_bp[0, 1] = float(test_data[i, 1])
            test_data_bp[0, 2] = float(test_data[i, 2])
            test_data_bp[0, 3] = float(test_data[i, 3])
            test_data_bp[0, 4] = float(test_data[i, 4])
            test_data_bp[0, 5] = float(test_data[i, 5])
            test_data_bp[0, 6] = float(test_data[i, 6])
            test_data_bp[0, 7] = float(test_data[i, 7])
            test_data_bp[0, 8] = float(test_data[i, 8])
            test_data_bp[0, 9] = float(test_data[i, 9])
            test_data_bp[0, 10] = float(test_data[i, 10])
            test_data_bp[0, 11] = float(test_data[i, 11])
            test_data_bp[0, 12] = float(test_data[i, 12])
            test_data_bp[0, 13] = float(test_data[i, 13])
            test_data_bp[0, 14] = float(test_data[i, 14])
            test_data_bp[0, 15] = float(test_data[i, 15])
            test_data_bp[0, 16] = float(test_data[i, 16])
            test_data_bp[0, 17] = float(test_data[i, 17])
            test_data_bp[0, 18] = float(test_data[i, 18])
            test_data_bp[0, 19] = float(test_data[i, 19])
            test_data_bp[0, 20] = float(test_data[i, 20])
            test_data_bp[0, 21] = float(test_data[i, 21])
            test_data_bp[0, 22] = float(test_data[i, 22])
            test_data_bp[0, 23] = float(test_data[i, 23])
            test_data_bp[0, 24] = float(test_data[i, 24])
            test_data_bp[0, 25] = float(test_data[i, 25])
            test_data_bp[0, 26] = float(test_data[i, 26])
            test_data_bp[0, 27] = float(test_data[i, 27])
            test_data_bp[0, 28] = float(test_data[i, 28])
            test_data_bp[0, 29] = float(test_data[i, 29])
            test_data_bp[0, 30] = float(test_data[i, 30])
            test_data_bp[0, 31] = float(test_data[i, 31])
            test_data_bp[0, 32] = float(test_data[i, 32])
            test_data_bp[0, 33] = float(test_data[i, 33])
            test_data_bp[0, 34] = float(supportVectors[j, 0])
            test_data_bp[0, 35] = float(supportVectors[j, 1])
            test_data_bp[0, 36] = float(supportVectors[j, 2])
            test_data_bp[0, 37] = float(supportVectors[j, 3])
            test_data_bp[0, 38] = float(supportVectors[j, 4])
            test_data_bp[0, 39] = float(supportVectors[j, 5])
            test_data_bp[0, 40] = float(supportVectors[j, 6])
            test_data_bp[0, 41] = float(supportVectors[j, 7])
            test_data_bp[0, 42] = float(supportVectors[j, 8])
            test_data_bp[0, 43] = float(supportVectors[j, 9])
            test_data_bp[0, 44] = float(supportVectors[j, 10])
            test_data_bp[0, 45] = float(supportVectors[j, 11])
            test_data_bp[0, 46] = float(supportVectors[j, 12])
            test_data_bp[0, 47] = float(supportVectors[j, 13])
            test_data_bp[0, 48] = float(supportVectors[j, 14])
            test_data_bp[0, 49] = float(supportVectors[j, 15])
            test_data_bp[0, 50] = float(supportVectors[j, 16])
            test_data_bp[0, 51] = float(supportVectors[j, 17])
            test_data_bp[0, 52] = float(supportVectors[j, 18])
            test_data_bp[0, 53] = float(supportVectors[j, 19])
            test_data_bp[0, 54] = float(supportVectors[j, 20])
            test_data_bp[0, 55] = float(supportVectors[j, 21])
            test_data_bp[0, 56] = float(supportVectors[j, 22])
            test_data_bp[0, 57] = float(supportVectors[j, 23])
            test_data_bp[0, 58] = float(supportVectors[j, 24])
            test_data_bp[0, 59] = float(supportVectors[j, 25])
            test_data_bp[0, 60] = float(supportVectors[j, 26])
            test_data_bp[0, 61] = float(supportVectors[j, 27])
            test_data_bp[0, 62] = float(supportVectors[j, 28])
            test_data_bp[0, 63] = float(supportVectors[j, 29])
            test_data_bp[0, 64] = float(supportVectors[j, 30])
            test_data_bp[0, 65] = float(supportVectors[j, 31])
            test_data_bp[0, 66] = float(supportVectors[j, 32])
            test_data_bp[0, 67] = float(supportVectors[j, 33])
            kernelValue = clf.predict(test_data_bp)
            BP_predictkernel[j] = kernelValue
        predict = alphasy * BP_predictkernel + b
        print("样本预测值为：", float(predict), "标签值为：", test_label[i], "预测标签值为：", np.sign(predict))
        if np.sign(predict) == np.sign(test_label[i]):
            matchCount += 1
            test_acc.append([float(test_label[i])])
        else:
            test_acc.append([-float(test_label[i])])
    t3 = time.time()
    test_yy = []
    for index, jj in enumerate(test_label):
        if jj == -1:
            test_yy.append([float(-1.0)])
        else:
            test_yy.append([float(1.0)])
    accuracy = float(matchCount) / test_data.shape[0]
    # roc_len = len(test_yy)
    # print("test_yy:", np.reshape(test_yy, (roc_len,)))
    # print("test_acc:",test_acc)
    # print("test_acc:%f" % float(accuracy))

    # print("test_auc:", roc_auc_score(np.reshape(test_yy, (roc_len,)), np.reshape(test_acc,(roc_len,))))
    return float(accuracy)

