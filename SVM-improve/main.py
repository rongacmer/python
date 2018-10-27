import tensorflow as tf
import toolbox
import time
import numpy as np
import copy
from svmpso import Solve
from config import cfg
from Adaboost import adaboost


def main(_):
    start = time.time()

    data, label = toolbox.loadData('vehicle.data')
    # test_data, test_label = toolbox.loadData('test.data')

    f_handle = open('runlog1.txt', mode='w')
    kernel, fitness = Solve(np.mat(data), np.mat(label), f_handle)
    # kernel = toolbox.nearestPD(kernel)
    f_handle.write("best_kernel:"+str(kernel)+"\n")
    f_handle.write("==========================================\n")
    print("best_kernel:", kernel)
    print("==========================================")
    # print("训练集准确率：%f" % toolbox.computer_tr(kernel, label))
    f_handle.write("最优核校准值："+str(fitness)+"\n")
    f_handle.write("测试集准确率："+str(toolbox.computer_acc_test(kernel, label))+"\n")
    print("最优核校准值：%f" % fitness)
    print("测试集准确率：%f" % toolbox.computer_acc_test(kernel, label))
    # m, n = np.shape(data)
    # data_b = copy.copy(data)
    # label_b = copy.copy(label)
    # data_b = np.mat(data_b)
    # label_b = np.mat(label_b)
    # b, alphasy, support_vectors = toolbox.computer_b(np.mat(data), np.mat(label), kernel)
    # new_data, new_label = toolbox.generate_data(data, kernel)
    # acc = adaboost(new_data,new_label, np.mat(test_data), np.mat(test_label), support_vectors,alphasy, b)
    # print("acc:%f" % acc)
    # print("auc:%f" % auc)

    end = time.time()

    print("本次运行总共花费%f分钟。" % ((end - start)/60))
    f_handle.close()


if __name__ == '__main__':
    tf.app.run()