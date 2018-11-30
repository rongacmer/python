import numpy as np
import random
import copy
import sys
import toolbox
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, linear_kernel
from sklearn.model_selection import StratifiedKFold,train_test_split
from config import cfg
from sklearn import metrics
from sklearn import svm
import gdbt
import multiprocessing
from sklearn.metrics import accuracy_score

def rand_pick(seq, probabilities):
    x = random.uniform(0, 1)
    cumprob = 0.0
    for item, item_pro in zip(seq, probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item



class Particle:
    def __init__(self, seed, X, label, position_list, f_test = None, train_percent = None):
        m, n = np.shape(X)
        seed = seed * random.Random().random()
        self.position = X.copy()
        self.rnd = random.Random(seed)
        self.velocity = np.mat(np.zeros((m, m)))
        label = label.T
        lowbound, upperbound = 0, 1
        for i in position_list:
            self.velocity[i[0], i[1]] = (cfg.max_velocity - cfg.min_velocity) * self.rnd.random() + cfg.min_velocity
            self.velocity[i[1], i[0]] = self.velocity[i[0], i[1]]
            if seed > 0:
                self.position[i[0], i[1]] = (upperbound - lowbound) * self.rnd.random() + lowbound
                self.position[i[1], i[0]] = self.position[i[0], i[1]]
        self.position = toolbox.nearestPD(self.position)
        self.best_part_position = self.position.copy()
        self.best_accuracy = f_test(self.position, np.ravel(label.T), train_percent = train_percent)[0]
        # toolbox.auc_measure(self.position, np.ravel(label.T))[0]]

def Solve(kernel, label, position_list, f_test = None, train_percent = None):
    rnd = random.Random()
    # symbol  = Particle(-1, kernel, label, position_list, f_test, train_percent = train_percent)
    # print(symbol.best_accuracy, toolbox.computer_acc_test(symbol.best_part_position, np.ravel(label), train_percent = train_percent)[0])
    swarm = [Particle(i, kernel, label, position_list, f_test, train_percent = train_percent) for i in range(cfg.n_particles)]
    m = kernel.shape[0]
    # best_swarm_fitness = sys.float_info.min
    best_swarm_position = np.mat(np.zeros((m, m)))
    best_swarm_accuracy = 0
    for i in range(cfg.n_particles):
        if swarm[i].best_accuracy > best_swarm_accuracy:
            best_swarm_position = copy.copy(swarm[i].position)
            best_swarm_accuracy = copy.copy(swarm[i].best_accuracy)

    epoch = 0
    w_max = 0.729  # 惯性因子
    w_min = 0.521
    c1 = 2  # 自我认识
    c2 = 2  # 社会认识
    while epoch < cfg.max_epoch:
        w = w_max - (w_max - w_min) * (epoch / cfg.max_epoch)
        for i in range(cfg.n_particles):
            for index in position_list:
                r1 = rnd.random()
                r2 = rnd.random()
                swarm[i].velocity[index[0], index[1]] = w * swarm[i].velocity[index[0], index[1]] + \
                                    c1 * r1 * (swarm[i].best_part_position[index[0], index[1]] - swarm[i].position[index[0], index[1]]) + \
                                    c2 * r2 * (best_swarm_position[index[0], index[1]] - swarm[i].position[index[0], index[1]])
                swarm[i].velocity[index[1], index[0]] = swarm[i].velocity[index[0], index[1]]
            swarm[i].position = swarm[i].position + swarm[i].velocity
            swarm[i].position[np.where(swarm[i].position < 0)] = 0
            swarm[i].position[np.where(swarm[i].position > 1)] = 1
            nearet_spd_kernel = toolbox.nearestPD(swarm[i].position)
            accuracy, traccuracy,Vaccuracy, tPR = f_test(nearet_spd_kernel, np.ravel(label), train_percent = train_percent)
            if accuracy > swarm[i].best_accuracy:
                swarm[i].position = copy.copy(nearet_spd_kernel)
                swarm[i].best_part_position = copy.copy(swarm[i].position)
                swarm[i].best_accuracy = accuracy
            # if best_swarm_accuracy >= 0.99:
            #     return best_swarm_position, best_swarm_accuracy
            if accuracy > best_swarm_accuracy :
                best_swarm_position = copy.copy(nearet_spd_kernel)
                best_swarm_accuracy = accuracy
        epoch += 1
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
            if rand_pick(value_list, probabilities) and i in support_vector and j in support_vector:
                position_list.append([i, j])
    return position_list

def cross_test(pid,data, label, subsample, train_percent = None):
    cfg.train_percent = train_percent / data.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(data[0:train_percent], label[0:train_percent],
                                                        test_size=1 - subsample,
                                                        stratify=label[0:train_percent])
    data[0: X_train.shape[0]], data[X_train.shape[0]:train_percent], label[0:len(y_train)], label[len(
        y_train):train_percent] = X_train, X_test, y_train, y_test
    kernel = rbf_kernel(data)
    oldaccuracy = 0
    mm = int(train_percent * cfg.test_percent)
    vacc = toolbox.computer_acc(kernel, label)[3]
    rate = 0.5 * (1 - vacc)
    for i in range(cfg.max_step):
        position_list = select_position(kernel, np.mat(label), mm, rate)
        kernel, accuracy = Solve(kernel, np.mat(label), position_list, toolbox.computer_acc,train_percent = train_percent)
        if accuracy - oldaccuracy < 1e-6:
            break
        oldaccuracy = accuracy
        # rate *= 0.9
    verify = train_percent
    test_assemble = int(verify * cfg.test_percent)
    tkernel = kernel[:test_assemble,:test_assemble]
    tlabel = label[:test_assemble]
    clf = svm.SVC(C = 1.0, kernel='precomputed')
    clf.fit(tkernel, tlabel)
    Op_predict = clf.predict(kernel[verify:,:test_assemble])
    acc = accuracy_score(label[verify:], Op_predict)
    print("Op",pid,acc)
    clf1 = gdbt.gdbt(data[:test_assemble], tkernel)
    test_kernel = gdbt.kernel_function(clf1,data[:test_assemble],data[verify:])
    St_predict = clf.predict(test_kernel)
    acc = accuracy_score(label[verify:], St_predict)
    print("St",pid,acc)
    return Op_predict,St_predict


def bagging(train_x,train_y,test_x,test_y,n_estimator = 15,subsample = cfg.test_percent, f_handle = None):
    train_percent = len(train_y)
    data = np.vstack((train_x, test_x))
    label = np.hstack((train_y, test_y))
    kernel = rbf_kernel(data)
    clf = svm.SVC(C=1.0, kernel='precomputed')
    clf.fit(kernel[:train_percent,:train_percent],label[:train_percent])
    predict = clf.predict(kernel[train_percent:,:train_percent])
    pre_acc = accuracy_score(test_y,predict)
    print(pre_acc)
    toolbox.checkerror(label[train_percent:], predict, f_handle)
    f_handle.write(str(pre_acc))
    per_predict = np.zeros(len(label[train_percent:]))
    Op_predict = per_predict.copy()
    St_predict = per_predict.copy()
    pool = multiprocessing.Pool()
    mul_res = [pool.apply_async(cross_test,(i,np.mat(data.copy()),label.copy(),subsample,train_percent,)) for i in range(n_estimator)]
    pool.close()
    pool.join()
    for res in mul_res:
        tmp = res.get()
        Op_predict += tmp[0]
        St_predict += tmp[1]
    Op_predict = np.sign(Op_predict)
    St_predict = np.sign(St_predict)
    Op_acc = accuracy_score(label[train_percent:], Op_predict)
    St_acc = accuracy_score(label[train_percent:], St_predict)
    print("Op_acc,St_acc:%f,%f" % (Op_acc, St_acc))
    f_handle.write(str(Op_acc) + " " + str(St_acc) + "\n")
    toolbox.checkerror(label[train_percent:], Op_predict, f_handle)
    toolbox.checkerror(label[train_percent:], St_predict, f_handle)
    return pre_acc,Op_acc,St_acc