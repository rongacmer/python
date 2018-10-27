import numpy as np
import random
import copy
import sys
import toolbox
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel
from config import cfg
from sklearn.svm import SVC


class Particle:
    def __init__(self, seed, X, label):
        m, n = np.shape(X)
        seed = seed * random.Random().random()
        self.rnd = random.Random(seed)
        # self.position = rbf_kernel(X, gamma=random.randint(0, 2.0))
        self.position = rbf_kernel(X)
        # self.position = polynomial_kernel(X)
        # print(self.position)
        # self.acc = toolbox.computer_acc_test(self.position, label)
        # print("acc", self.acc)
        label = label.T
        kernel, train_yy = toolbox.get_kernel_yy(self.position, m, label)
        self.mini, self.minj = 0, 0
        self.velocity = 0.0
        mindis = 0
        if seed >= 0:
            for i in range(m):
                for j in range(m):
                    if train_yy[i, j] < 0 and kernel[i, j] > mindis:
                        self.mini, self.minj = i, j
                        mindis = kernel[i, j]
            self.velocity = ((cfg.max_velocity - cfg.min_velocity) * self.rnd.random() + cfg.min_velocity)
            lowbound ,upperbound = 0, 1
            self.position[self.mini, self.minj] = (upperbound - lowbound) * self.rnd.random() + lowbound
            self.position[self.minj, self.mini] = self.position[self.mini, self.minj]
        self.best_part_position = self.position
        self.best_fitness = toolbox.fitness_function(kernel, train_yy, label) #核校准值
        # self.best_fitness = toolbox.fitness_function(self.position, train_yy)
        # print("position:", kernel)
        # print("shape:", np.shape(kernel))
        # print("fitness:", self.best_fitness)


def Solve(data, label, f_handle):
    rnd = random.Random()
    symbol  = Particle(-1, data, label)
    f_handle.write("初始化粒子中最优和校准值：" + str(symbol.best_fitness) + "\n")
    f_handle.write(str(symbol.position)+"\n")
    print("初始化粒子中最优核校准值：", symbol.best_fitness)
    print("测试集准确率：%f" % toolbox.computer_acc_test(symbol.best_part_position, label.T))
    swarm = [Particle(i, data, label) for i in range(cfg.n_particles)]
    x, y = swarm[0].mini, swarm[0].minj
    print(swarm[0].position[x,y],symbol.position[x,y])
    m, n = np.shape(data)
    best_swarm_fitness = sys.float_info.min
    best_swarm_position = np.mat(np.zeros((m, m)))
    for i in range(cfg.n_particles):
        # print(swarm[i].position)
        # f_handle.write(str(swarm[i].position) + "\n")
        if swarm[i].best_fitness > best_swarm_fitness:
            best_swarm_position = copy.copy(swarm[i].position)
            best_swarm_fitness = copy.copy(swarm[i].best_fitness)
    f_handle.write("初始化粒子中最优和校准值：" + str(best_swarm_fitness)+"\n")
    print("初始化粒子中最优核校准值：", best_swarm_fitness)

    epoch = 0
    w_max = 0.729  # 惯性因子
    w_min = 0.521
    c1 = 2  # 自我认识
    c2 = 2  # 社会认识
    while epoch < cfg.max_epoch:
        f_handle.write("*******************" + str(epoch) + "th epoch*****************\n")
        print("第%d轮" % epoch)
        w = w_max - (w_max - w_min) * (epoch / cfg.max_epoch)
        for i in range(cfg.n_particles):
            f_handle.write("The " + str(i) + "th particle\n")
            print("第%d个粒子" % i)
            # for k1 in range(m):
            #     for k2 in range(m):
            #         r1 = rnd.random()
            #         r2 = rnd.random()
            #         swarm[i].velocity[k1, k2] = w * swarm[i].velocity[k1, k2] + c1 * r1 * \
            #                                    (swarm[i].best_part_position[k1, k2]-swarm[i].position[k1, k2]) + \
            #                                    c2 * r2 * (best_swarm_position[k1, k2] - swarm[i].position[k1, k2])
            # print("更新速度")
            k1 = swarm[i].mini
            k2 = swarm[i].minj
            r1 = rnd.random()
            r2 = rnd.random()
            swarm[i].velocity = w * swarm[i].velocity + \
                                c1 * r1 * (swarm[i].best_part_position[k1, k2] - swarm[i].position[k1,k2]) + \
                                c2 * r2 * (best_swarm_position[k1, k2] - swarm[i].position[k1,k2])
            # for k1 in range(m):
            #     for k2 in range(m):
            #         swarm[i].position[k1, k2] += swarm[i].velocity[k1, k2]
            swarm[i].position[k1, k2] += swarm[i].velocity
            swarm[i].position[k2, k1] = swarm[i].position[k1, k2]
            # print("gggg")
            kernel_alignment, yy_alignment = toolbox.get_kernel_yy(swarm[i].position, m, label.T)
            # print("dfgee")
            fitness_alignment_ago = toolbox.fitness_function(kernel_alignment, yy_alignment, label)
            # print("adfaf")
            nearet_spd_kernel = toolbox.nearestPD(swarm[i].position)
            train_kernel, train_yy = toolbox.get_kernel_yy(nearet_spd_kernel, m, label.T)
            fitness = toolbox.fitness_function(train_kernel, train_yy, label)
            f_handle.write("---------------------------------------------\n")
            f_handle.write("nearest前的核校准值："+str(fitness_alignment_ago)+"\n")
            f_handle.write("nearest后的核校准值："+str(fitness)+"\n")
            f_handle.write("核校准前后的差值："+str(fitness - fitness_alignment_ago)+"\n")
            f_handle.write("变化比重："+str(((fitness - fitness_alignment_ago) / fitness_alignment_ago) * 100)+"\n")
            print("---------------------------------------------")
            print("nearest前的核校准值：", fitness_alignment_ago)
            print("nearest后的核校准值：", fitness)
            print("核校准前后的差值：", fitness - fitness_alignment_ago)
            print("变化比重：", ((fitness - fitness_alignment_ago) / fitness_alignment_ago) * 100)
            # print(np.shape(label))
            print("测试集准确率：%f" % toolbox.computer_acc_test(best_swarm_position, label.T))
            # fitness = toolbox.fitness_function(nearet_spd_kernel, train_yy)

            if fitness > swarm[i].best_fitness:
                swarm[i].best_fitness = copy.copy(fitness)
                swarm[i].position = copy.copy(nearet_spd_kernel)
                swarm[i].best_part_position = copy.copy(swarm[i].position)

            if fitness > best_swarm_fitness:
                # print("---------------------------------------------")
                # print("nearest前的核校准值：", fitness_alignment_ago)
                # print("nearest后的核校准值：", fitness)
                # print("核校准前后的插值：", fitness - fitness_alignment_ago)
                # print("变化比重：", ((fitness-fitness_alignment_ago)/fitness_alignment_ago)*100)
                f_handle.write("最优核校准值：%f "+str(fitness)+"\n ")
                print("最优核校准值：%f " % fitness)
                # print("核矩阵中的正负两个元素值[2,2],[0,3]：", nearet_spd_kernel[2][2], nearet_spd_kernel[0][3])
                best_swarm_fitness = copy.copy(fitness)
                best_swarm_position = copy.copy(nearet_spd_kernel)
                # data_b = copy.copy(data)
                # label_b = copy.copy(label)
                # data_b = np.mat(data_b)
                # label_b = np.mat(label_b)
                # b = toolbox.computer_b(data_b[0:int(cfg.train_percent*m)], label_b[0:int(cfg.train_percent*m)], clf)
                # best_b = copy.copy(b)
        epoch += 1
    return best_swarm_position, best_swarm_fitness