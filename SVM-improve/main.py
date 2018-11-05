import tensorflow as tf
import toolbox
import time
import numpy as np
import copy
import mulsvmpso
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel
from config import cfg
from Adaboost import adaboost
from sklearn.model_selection import StratifiedKFold



def main(_):
    start = time.time()
    data, label = toolbox.loadData('vehicle.data')
    data = np.mat(data)
    label = np.array(label)
    crossdata = data.copy()
    crosslabel = label.copy()
    # print(type(data))
    f_handle = open('mulsvmlog-vehicle16.txt', mode='w')
    # test_data, test_label = toolbox.loadData('test.data')
    # svmpso.cross_test(data, label, f_handle)
    TEST = 0
    sfolder = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = False)
    for train, test in sfolder.split(crossdata, crosslabel):
        # print(crossdata[train])
        data[0:len(train)], data[len(train):] = crossdata[train],crossdata[test]
        label[0:len(train)], label[len(train):] =  crosslabel[train], crosslabel[test]
        f_handle.write("*******************" + str(TEST) + "th test *****************\n")
        print("*******************" + str(TEST) + "th test *****************\n")
        mulsvmpso.cross_test(data, label, f_handle)
        TEST += 1
        print("\n\n")

    # m = data.shape[0]
    # f_handle.write("*******************" + str(0) + "th test *****************\n")
    # print("*******************" + str(0) + "th test *****************\n")
    # mulsvmpso.cross_test(data, label, f_handle)
    # print("\n\n")
    # for i in range(4):
    #     f_handle.write("*******************" + str(i+1) + "th test *****************\n")
    #     print("*******************" + str(i+1) + "th test *****************\n")
    #     for j in range(int(m * 0.2)):
    #         k1 = int(i * 0.2 * m ) + j
    #         k2 = m - int(m * 0.2) + j
    #         data[[k1,k2],:] = data[[k2,k1], :]
    #         label[k1], label[k2] = label[k2], label[k1]
    #     mulsvmpso.cross_test(data, label, f_handle)
    #     print("\n\n")
    bacc, aacc, bauc, aauc = 0, 0, 0, 0
    for i in range(len(mulsvmpso.aftacc)):
        bacc += mulsvmpso.beforeacc[i]
        aacc += mulsvmpso.aftacc[i]
        bauc += mulsvmpso.beauc[i]
        aauc += mulsvmpso.aftauc[i]
    bacc = bacc / 5
    aacc = aacc / 5
    bauc = bauc / 5
    aauc = aauc / 5
    fbacc, faacc, fbauc, faauc = 0, 0, 0, 0
    for i in range(len(mulsvmpso.aftacc)):
        fbacc += (mulsvmpso.beforeacc[i] - bacc) ** 2
        faacc += (mulsvmpso.aftacc[i] - aacc) ** 2
        fbauc += (mulsvmpso.beauc[i] - bauc) ** 2
        faauc += (mulsvmpso.aftauc[i] - aauc) ** 2

    fbacc = fbacc / 5
    faacc = faacc / 5
    fbauc = fbauc / 5
    faauc = faauc / 5
    f_handle.write(str(bacc)+" " + str(bauc)+" "+str(fbacc)+" "+str(fbauc) + "\n")
    print("优化前： %f %f %f %f\n" % (bacc, bauc, fbacc, fbauc))
    f_handle.write(str(aacc) + " "+str(aauc) + " "+ str(faacc) + " "+ str(faauc)+"\n")
    print("优化后： %f %f %f %f\n" % (aacc, aauc, faacc, faauc))
    end = time.time()

    print("本次运行总共花费%f分钟。" % ((end - start)/60))
    f_handle.close()


if __name__ == '__main__':
    tf.app.run()