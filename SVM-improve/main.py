import tensorflow as tf
import toolbox
import time
import numpy as np
import copy
import mulsvmpso
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, linear_kernel
from config import cfg
from Adaboost import adaboost
from sklearn.model_selection import StratifiedKFold,train_test_split
import select_support_vector
from sklearn.metrics import accuracy_score

def main(_):
    start = time.time()
    data, label = toolbox.loadData('PokerHand.data')
    data = np.mat(data)
    label = np.array(label)
    # inter_support = select_support_vector.solve(data, label)
    # train_length = len(inter_support)
    # i = 0
    # j = len(inter_support)
    # newdata = np.mat(np.zeros(data.shape))
    # newlabel = label.copy()
    # for index in range(data.shape[0]):
    #     if index in inter_support:
    #         newdata[i] = data[index]
    #         newlabel[i] = label[index]
    #         i += 1
    #     else:
    #         newdata[j] = data[index]
    #         newlabel[j] = label[index]
    #         j += 1
    # kernel = rbf_kernel(newdata)
    # clf = svm.SVC(C=cfg.C, kernel='precomputed')
    # clf.fit(kernel[:train_length, :train_length], newlabel[:train_length])
    # predict = clf.predict(kernel[:,:train_length])
    # accuracy = accuracy_score(newlabel, predict)
    # print(accuracy,clf.support_.shape)

    crossdata = data.copy()
    crosslabel = label.copy()
    print(type(data))
    f_handle = open('mulsvmlog-PokerHand4.txt', mode='w')
    test_data, test_label = toolbox.loadData('test.data')
    # svmpso.cross_test(data, label, f_handle)
    TEST = 0
    sfolder = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = False)
    for train, test in sfolder.split(crossdata, crosslabel):
        # print(crossdata[train])
        data[0:len(train)], data[len(train):] = crossdata[train],crossdata[test]
        label[0:len(train)], label[len(train):] =  crosslabel[train], crosslabel[test]
        X_train, X_test, y_train, y_test = train_test_split(data[0:len(train)], label[0:len(train)], test_size=1 - cfg.test_percent ,stratify=label[0:len(train)])
        data[0: X_train.shape[0]], data[X_train.shape[0]:len(train)], label[0:len(y_train)], label[len(y_train):len(train)] = X_train, X_test, y_train, y_test
        f_handle.write("*******************" + str(TEST) + "th test *****************\n")
        print("*******************" + str(TEST) + "th test *****************\n")
        mulsvmpso.cross_test(data, label, f_handle)
        TEST += 1
        print("\n\n")
    bacc, aacc, bauc, aauc = 0, 0, 0, 0
    times = len(mulsvmpso.aftacc)
    for i in range(len(mulsvmpso.aftacc)):
        bacc += mulsvmpso.beforeacc[i]
        aacc += mulsvmpso.aftacc[i]
        bauc += mulsvmpso.beauc[i]
        aauc += mulsvmpso.aftauc[i]
    bacc = bacc / times
    aacc = aacc / times
    bauc = bauc / times
    aauc = aauc / times
    fbacc, faacc, fbauc, faauc = 0, 0, 0, 0
    for i in range(len(mulsvmpso.aftacc)):
        fbacc += (mulsvmpso.beforeacc[i] - bacc) ** 2
        faacc += (mulsvmpso.aftacc[i] - aacc) ** 2
        fbauc += (mulsvmpso.beauc[i] - bauc) ** 2
        faauc += (mulsvmpso.aftauc[i] - aauc) ** 2
    fbacc = fbacc / times
    faacc = faacc / times
    fbauc = fbauc / times
    faauc = faauc / times
    f_handle.write(str(bacc)+" " + str(bauc)+" "+str(fbacc)+" "+str(fbauc) + "\n")
    print("优化前： %f %f %f %f\n" % (bacc, bauc, fbacc, fbauc))
    f_handle.write(str(aacc) + " "+str(aauc) + " "+ str(faacc) + " "+ str(faauc)+"\n")
    print("优化后： %f %f %f %f\n" % (aacc, aauc, faacc, faauc))
    end = time.time()
    print("本次运行总共花费%f分钟。" % ((end - start)/60))
    f_handle.close()


if __name__ == '__main__':
    tf.app.run()