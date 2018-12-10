import tensorflow as tf
import toolbox
import time
import numpy as np
import copy
import parallelsvm
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, linear_kernel
from config import cfg
from Adaboost import adaboost
from sklearn.model_selection import StratifiedKFold,train_test_split
import select_support_vector
from sklearn.metrics import accuracy_score

def main(_):
    start = time.time()
    # data, label = toolbox.loadData('ionosphere.data')
    #
    # label = np.array(label)
    f_handle = open('new_MCIccvMCInclog3.txt', mode='w')
    Train = 0
    TEST = 0
    Pre_Acc, Op_Acc, St_Acc = [], [], []
    # per = 5
    # sfolder = StratifiedKFold(n_splits=per, shuffle=True)
    # for test in range(10):
    # # for train, test in sfolder.split(data, label):
    #     f_handle.write("*******************" + str(Train) + "th Train *****************\n")
    #     print("*******************" + str(Train) + "th Train *****************\n")
    #     cfg.train_percent = 150 / data.shape[0]
    #     X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=150,stratify=label)
    #     Pre_acc,Op_acc,St_acc = parallelsvm.bagging(X_test,y_test,X_train,y_train,n_estimator=15,subsample=0.8,f_handle = f_handle)
    #     Pre_Acc.append(Pre_acc)
    #     Op_Acc.append(Op_acc)
    #     St_Acc.append(St_acc)
    #     Train += 1
    for j in range(5):
        for i in range(5):
            data, label, train_percent = toolbox.loadfold(i + 1)
            # Pre_Acc, Op_Acc, St_Acc = [], [], []
            # for j in range(5):
            f_handle.write("*******************" + str(Train) + "th Train *****************\n")
            print("*******************" + str(Train) + "th Train *****************\n")
            # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=100,stratify=label)
            cfg.train_percent = train_percent / data.shape[0]
            # Pre_acc,Op_acc,St_acc = parallelsvm.bagging(X_test,y_test,X_train,y_train,n_estimator=15,subsample=0.8,f_handle = f_handle)
            Pre_acc, Op_acc, St_acc = parallelsvm.bagging(data[:train_percent], label[:train_percent], data[train_percent:], label[train_percent:], n_estimator=15,subsample=0.8, f_handle=f_handle)
            Pre_Acc.append(Pre_acc)
            Op_Acc.append(Op_acc)
            St_Acc.append(St_acc)
            Train += 1
                # per = 5
                # sfolder = StratifiedKFold(n_splits = per, random_state = 0, shuffle = True)
                # crossdata = data[:train_percent]
                # crosslabel = label[:train_percent]
                # per_predict = np.zeros(len(label[train_percent:]))
                # Op_predict = per_predict.copy()
                # St_predict = per_predict.copy()
                # for j in range(15):
                #     X_train, X_test, y_train, y_test = train_test_split(data[0:train_percent], label[0:train_percent], test_size=1 - cfg.test_percent ,stratify=label[0:train_percent])
                #     data[0: X_train.shape[0]], data[X_train.shape[0]:train_percent], label[0:len(y_train)], label[len(y_train):train_percent] = X_train, X_test, y_train, y_test
                #     f_handle.write("*******************" + str(TEST) + "th test *****************\n")
                #     print("*******************" + str(TEST) + "th test *****************\n")
                #     # data[0:len(train)], data[len(train):train_percent] = crossdata[train], crossdata[test]
                #     # label[0:len(train)], label[len(train):train_percent] = crosslabel[train], crosslabel[test]
                #     op_predict,st_predict = mulsvmpso.cross_test(np.mat(data), label, f_handle)
                #     Op_predict += op_predict
                #     St_predict += st_predict
                #     TEST += 1
                # Op_predict = np.sign(Op_predict)
                # St_predict = np.sign(St_predict)
                # Op_acc = accuracy_score(label[train_percent:], Op_predict)
                # St_acc = accuracy_score(label[train_percent:], St_predict)
            # print(np.mean(Pre_Acc), np.std(Pre_Acc))
            # print(np.mean(Op_Acc), np.std(Op_Acc))
            # print(np.mean(St_Acc), np.std(St_Acc))
            # f_handle.write(str(np.mean(Pre_Acc)) + " " + str(np.std(Pre_Acc)) + "\n")
            # f_handle.write(str(np.mean(Op_Acc)) + " " + str(np.std(Op_Acc)) + "\n")
            # f_handle.write(str(np.mean(St_Acc)) + " " + str(np.std(St_Acc)) + "\n")
    print(np.mean(Pre_Acc), np.std(Pre_Acc))
    print(np.mean(Op_Acc), np.std(Op_Acc))
    print(np.mean(St_Acc), np.std(St_Acc))
    f_handle.write(str(np.mean(Pre_Acc)) + " " + str(np.std(Pre_Acc)) + "\n")
    f_handle.write(str(np.mean(Op_Acc)) + " " + str(np.std(Op_Acc)) + "\n")
    f_handle.write(str(np.mean(St_Acc)) + " " + str(np.std(St_Acc)) + "\n")

    end = time.time()
    print("本次运行总共花费%f分钟。" % ((end - start)/60))
    # f_handle.write("本次运行总共花费%f分钟。"+str((end-str)/60)+'\n')
    f_handle.close()


if __name__ == '__main__':
    tf.app.run()