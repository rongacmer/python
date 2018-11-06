import numpy as np
import random
import copy
import sys
import toolbox
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, linear_kernel
from config import cfg
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing

beforeacc = list()
beauc = list()
aftacc = list()
aftauc = list()

def rand_pick(seq, probabilities):
    x = random.uniform(0, 1)
    cumprob = 0.0
    for item, item_pro in zip(seq, probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item



class Particle:
    def __init__(self, seed, X, label, position_list, f_test = None):
        m, n = np.shape(X)
        seed = seed * random.Random().random()
        self.position = X.copy()
        self.rnd = random.Random(seed)
        self.velocity = np.mat(np.zeros((m, m)))
        label = label.T
        lowbound, upperbound = -1, 1
        for i in position_list:
            self.velocity[i[0], i[1]] = (cfg.max_velocity - cfg.min_velocity) * self.rnd.random() + cfg.min_velocity
            self.velocity[i[1], i[0]] = self.velocity[i[0], i[1]]
            if seed > 0:
                self.position[i[0], i[1]] = (upperbound - lowbound) * self.rnd.random() + lowbound
                self.position[i[1], i[0]] = self.position[i[0], i[1]]
        self.position = toolbox.nearestPD(self.position)
        self.best_part_position = self.position.copy()
        self.best_accuracy = f_test(self.position, np.ravel(label.T))[0]
        # toolbox.auc_measure(self.position, np.ravel(label.T))[0]]

def Solve(kernel, label, f_handle, position_list, f_test = None):
    rnd = random.Random()
    symbol  = Particle(-1, kernel, label, position_list, f_test)
    print(symbol.best_accuracy, toolbox.computer_acc_test(symbol.best_part_position, np.ravel(label))[0])
    swarm = [Particle(i, kernel, label, position_list, f_test) for i in range(cfg.n_particles)]
    # print(swarm[1].position[x,y], symbol.position[x,y])
    m = kernel.shape[0]
    # best_swarm_fitness = sys.float_info.min
    best_swarm_position = np.mat(np.zeros((m, m)))
    best_swarm_accuracy = 0.0
    for i in range(cfg.n_particles):
        # print(swarm[i].position)
        # f_handle.write(str(swarm[i].position) + "\n")
        if swarm[i].best_accuracy > best_swarm_accuracy:
            # (swarm[i].best_accuracy == best_swarm_accuracy and swarm[i].best_fitness > best_swarm_fitness):
            best_swarm_position = copy.copy(swarm[i].position)
            # best_swarm_fitness = copy.copy(swarm[i].best_fitness)
            best_swarm_accuracy = copy.copy(swarm[i].best_accuracy)

    epoch = 0
    w_max = 0.729  # 惯性因子
    w_min = 0.521
    c1 = 2  # 自我认识
    c2 = 2  # 社会认识
    while epoch < cfg.max_epoch:
        f_handle.write("*******************" + str(epoch) + "th epoch*****************\n")
        # print("第%d轮" % epoch)
        w = w_max - (w_max - w_min) * (epoch / cfg.max_epoch)
        for i in range(cfg.n_particles):
            f_handle.write("The " + str(i) + "th particle\n")
            for index in position_list:
                r1 = rnd.random()
                r2 = rnd.random()
                swarm[i].velocity[index[0], index[1]] = w * swarm[i].velocity[index[0], index[1]] + \
                                    c1 * r1 * (swarm[i].best_part_position[index[0], index[1]] - swarm[i].position[index[0], index[1]]) + \
                                    c2 * r2 * (best_swarm_position[index[0], index[1]] - swarm[i].position[index[0], index[1]])
                swarm[i].velocity[index[1], index[0]] = swarm[i].velocity[index[0], index[1]]
            swarm[i].position = swarm[i].position + swarm[i].velocity
            nearet_spd_kernel = toolbox.nearestPD(swarm[i].position)
            accuracy, allacuracy = f_test(nearet_spd_kernel, np.ravel(label))
            # accuracy = [accuracy, toolbox.auc_measure(nearet_spd_kernel, np.ravel(label))[0]]
            f_handle.write("---------------------------------------------\n")
            if accuracy > swarm[i].best_accuracy :
                swarm[i].position = copy.copy(nearet_spd_kernel)
                swarm[i].best_part_position = copy.copy(swarm[i].position)
                swarm[i].best_accuracy = accuracy
            if accuracy > best_swarm_accuracy:
                best_swarm_position = copy.copy(nearet_spd_kernel)
                best_swarm_accuracy = accuracy
                rate, support_index = toolbox.computer_acc_test(best_swarm_position, np.ravel(label))
                f_handle.write("测试集准确率：" + str(rate) + "\n")
                f_handle.write("支持向量:" + str(support_index.shape[0]) + "\n")
                f_handle.write("f_measure:" + str(accuracy) + " " + str(allacuracy) + "\n")
                print("f_measure:%f" % accuracy + "\n")
                print("测试集准确率：%f %f" % (rate,allacuracy) + "\n")
            # print("支持向量：", support_index)
        epoch += 1
    print(best_swarm_position)
    print(best_swarm_accuracy)
    return best_swarm_position, best_swarm_accuracy

def select_position(kernel, lable, m, rate):
    value_list = [1, 0]
    probabilities = [rate, 1 - rate]
    position_list = []
    train_kernel = kernel[0:m, 0:m]
    train_label = lable.T * lable
    clf = svm.SVC(C=cfg.C, kernel='precomputed')
    clf.fit(train_kernel, np.ravel(lable[0,0:m]))
    support_vector = clf.support_
    print(len(support_vector))
    for i in range(m):
        for j in range(i - 1):
            if rand_pick(value_list, probabilities): #and i in support_vector and j in support_vector: #and train_label[i, j] < 0:
                position_list.append([i, j])
    return position_list

def cross_test(data, label, f_handle):
    kernel = linear_kernel(data)
    # kernel = preprocessing.scale(kernel)
    # kernel = kernel / max(abs(np.max(kernel)), abs(np.min(kernel)))
    # kernel = toolbox.nearestPD(kernel)
    acc = toolbox.computer_acc_test(kernel, label)[0]
    auc = toolbox.PR_measure(kernel, label)[1]
    print("初始测试集准确率:%f %f" % (acc, auc))
    f_handle.write("初始测试集准确率:%f %f" + str(acc) + str(auc))
    # kernel = preprocessing.normalize(kernel,norm='l2')
    # kernel = preprocessing.scale(kernel)
    # kernel = toolbox.nearestPD(kernel)
    # print(kernel)
    beforeacc.append(acc)
    beauc.append(auc)
    rate = cfg.origin_rate
    for i in range(cfg.max_step):
        f_handle.write("*******************" + str(i) + "th step*****************\n")
        position_list = select_position(kernel, np.mat(label), int(kernel.shape[0] * cfg.test_percent), rate)
        print(rate)
        print(len(position_list))
        kernel, accuracy = Solve(kernel, np.mat(label), f_handle, position_list, toolbox.PR_measure)
        # rate *= cfg.T
    # kernel = toolbox.nearestPD(kernel)
    f_handle.write("best_kernel:" + str(kernel) + "\n")
    f_handle.write("==========================================\n")
    print("best_kernel:", kernel)
    print("==========================================")
    # print("训练集准确率：%f" % toolbox.computer_tr(kernel, label))
    # f_handle.write("最优核校准值："+str(fitness)+"\n")
    # print("最优核校准值：%f" % fitness)
    rate, support_index = toolbox.computer_acc_test(kernel, label)
    allacuracy = toolbox.PR_measure(kernel, label)[1]
    f_handle.write("测试集准确率：" + str(rate) +" "+str(allacuracy) + "\n")
    f_handle.write("支持向量:" + str(support_index) + "\n")
    print("测试集准确率：%f %f" % (rate, allacuracy) + "\n")
    print("支持向量：", support_index)
    aftacc.append(rate)
    aftauc.append(allacuracy)