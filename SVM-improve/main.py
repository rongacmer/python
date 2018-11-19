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
    data, label = toolbox.loadData('wisconsin.data')
    f_handle = open('mulsvmlog-wisconsin14.txt', mode='w')
    data = np.mat(data)
    label = np.array(label)
    # m = int(data.shape[0])
    # m = int(int(m * cfg.test_percent) * cfg.train_percent)
    # kernel = rbf_kernel(data)
    # clf = svm.SVC(C=cfg.C, kernel='precomputed')
    # clf.fit(kernel[:m,:m], label[:m])
    # predict = clf.predict(kernel[:, :m])
    # test_decision = toolbox.decision_function( kernel[:,clf.support_], np.mat(clf.dual_coef_), clf.intercept_)
    # oldaccuracy = accuracy_score(label, predict)
    # oldsupport = clf.support_.copy()
    # cnt = 0
    # error = 0
    # new_support = []
    # new_coef = []
    # for i in range(len(oldsupport)):
    #     index = oldsupport[i]
    #     if label[index] == predict[index]:
    #         new_support.append(oldsupport[i])
    #         new_coef.append(clf.dual_coef_[0,i])
    # new_decision = toolbox.decision_function(kernel[:, new_support], np.mat(new_coef),clf.intercept_)
    # print(new_decision)
    # # print(clf.dual_coef_)
    # newdecision = clf.decision_function(kernel[:, :m]).copy()
    # # print(decision)
    # print("错误：")
    # print(error, cnt)
    # inter_support = select_support_vector.solve(data[:m], label[:m])
    # train_length = len(inter_support)
    # maxstep = 10
    # step = 0
    # while step < maxstep:
    #     i = 0
    #     j = len(inter_support)
    #     newdata = np.mat(np.zeros(data.shape))
    #     newlabel = label.copy()
    #     for index in range(data.shape[0]):
    #         if index in inter_support:
    #             newdata[i] = data[index]
    #             newlabel[i] = label[index]
    #             i += 1
    #         else:
    #             newdata[j] = data[index]
    #             newlabel[j] = label[index]
    #             j += 1
    #     kernel = mulsvmpso.cross_test(newdata, newlabel, f_handle, train_percent = train_length)
    #     clf.fit(kernel[:train_length, :train_length], newlabel[:train_length])
    #     predict = clf.predict(kernel[:, :train_length])
    #     newaccuracy = accuracy_score(newlabel, predict)
    #     if clf.support_.shape[0] >= train_length:
    #         break
    #     else:
    #         data = newdata.copy()
    #         label = newlabel.copy()
    #         inter_support = clf.support_.copy()
    #         train_length = clf.support_.shape[0]
    #     step += 1
    # f_handle.write("优化前\n")
    # f_handle.write(str(oldaccuracy) + ","+str(oldsupport.shape)+'\n')
    # f_handle.write(str(oldsupport))
    # print("优化前")
    # print(oldaccuracy,oldsupport.shape)
    # print(oldsupport)
    # f_handle.write("优化后\n")
    # f_handle.write(str(newaccuracy) + ","+str(clf.support_.shape) + '\n')
    # f_handle.write(str(clf.support_))
    # print("优化后")
    # print(newaccuracy,clf.support_.shape)
    # print(clf.support_)
    crossdata = data.copy()
    crosslabel = label.copy()
    # f_handle = open('mulsvmlog-PokerHand4.txt', mode='w')
    # test_data, test_label = toolbox.loadData('test.data')
    # svmpso.cross_test(data, label, f_handle)
    TEST = 0
    sfolder = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = False)
    for train, test in sfolder.split(crossdata, crosslabel):
        # print(crossdata[train])
        data[0:len(train)], data[len(train):] = crossdata[train],crossdata[test]
        label[0:len(train)], label[len(train):] =  crosslabel[train], crosslabel[test]
        # m = data.shape[0]
        # m = int(m * cfg.test_percent)
        # kernel = rbf_kernel(data)
        # clf = svm.SVC(C=cfg.C, kernel='precomputed')
        # clf.fit(kernel[:m,:m], label[:m])
        # i = 0
        # j = len(clf.support_)
        # newdata = data.copy()
        # newlabel = label.copy()
        # for index in range(data.shape[0]):
        #     if index in clf.support_:
        #         newdata[i] = data[index]
        #         newlabel[i] = label[index]
        #         i += 1
        #     else:
        #         newdata[j] = data[index]
        #         newlabel[j] = label[index]
        #         j += 1
        X_train, X_test, y_train, y_test = train_test_split(data[0:len(train)], label[0:len(train)], test_size=1 - cfg.test_percent ,stratify=label[0:len(train)])
        data[0: X_train.shape[0]], data[X_train.shape[0]:len(train)], label[0:len(y_train)], label[len(y_train):len(train)] = X_train, X_test, y_train, y_test
        f_handle.write("*******************" + str(TEST) + "th test *****************\n")
        print("*******************" + str(TEST) + "th test *****************\n")
        mulsvmpso.cross_test(data, label, f_handle)
        TEST += 1
        print("\n\n")
    bacc, aacc, bauc, aauc, sacc= 0, 0, 0, 0, 0
    times = len(mulsvmpso.aftacc)
    for i in range(len(mulsvmpso.aftacc)):
        bacc += mulsvmpso.beforeacc[i]
        aacc += mulsvmpso.aftacc[i]
        bauc += mulsvmpso.beauc[i]
        aauc += mulsvmpso.aftauc[i]
        sacc += mulsvmpso.studentacc[i]
    bacc = bacc / times
    aacc = aacc / times
    bauc = bauc / times
    aauc = aauc / times
    sacc = sacc / times
    fbacc, faacc, fbauc, faauc, fsacc= 0, 0, 0, 0, 0
    for i in range(len(mulsvmpso.aftacc)):
        fbacc += (mulsvmpso.beforeacc[i] - bacc) ** 2
        faacc += (mulsvmpso.aftacc[i] - aacc) ** 2
        fbauc += (mulsvmpso.beauc[i] - bauc) ** 2
        faauc += (mulsvmpso.aftauc[i] - aauc) ** 2
        fsacc += (mulsvmpso.studentacc[i] - sacc) ** 2
    fbacc = fbacc / times
    faacc = faacc / times
    fbauc = fbauc / times
    faauc = faauc / times
    fsacc =  fsacc / times
    f_handle.write(str(bacc)+" " + str(bauc)+" "+str(fbacc)+" "+str(fbauc) + "\n")
    print("优化前： %f %f %f %f\n" % (bacc, bauc, fbacc, fbauc))
    f_handle.write(str(aacc) + " "+str(aauc) + " "+ str(faacc) + " "+ str(faauc)+"\n")
    f_handle.write(str(sacc) + "" + str(fsacc) + "\n")
    print("优化后： %f %f %f %f\n" % (aacc, aauc, faacc, faauc))
    print(sacc, fsacc)
    end = time.time()
    print("本次运行总共花费%f分钟。" % ((end - start)/60))
    f_handle.close()


if __name__ == '__main__':
    tf.app.run()