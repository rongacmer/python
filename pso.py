import numpy as np
import matplotlib.pyplot as plt

class PSO(object):
    def __init__(self, popularsize, maxstep):
        self.w = 0.6
        self.c1 = self.c2 = 2
        self.popularsize = popularsize
        self.dim = 2
        self.low = -10
        self.upper = 10
        self.x = np.random.uniform(self.low,self.upper,(self.popularsize,self.dim))
        self.v = np.random.rand(self.popularsize,self.dim)
        self.p = self.x
        fitness = self.calfitness(self.x)
        self.pg = self.x[np.argmin(fitness)]
        self.global_individual_fitness = fitness
        self.global_best_fitness = np.min(fitness)

    def calfitness(self,x):
        return np.sum(np.square(x), axis=1)

    def evolve(self):

        for i in range(self.popularsize):
            r1 = np.random.rand(self.popularsize, self.dim)
            r2 = np.random.rand(self.popularsize, self.dim)
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x) #速度变化公式
            self.x = self.x + self.v
            plt.clf()
            plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')
            plt.xlim(self.low, self.upper)
            plt.ylim(self.low, self.upper)
            plt.pause(0.1)
            fitness = self.calfitness(self.x)
            update_id = np.greater(self.global_individual_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.global_individual_fitness[update_id] = fitness[update_id]
            if np.min(fitness) < self.global_best_fitness:
                self.global_best_fitness = np.min(fitness)
                self.pg = self.x[np.argmin(fitness)]
        print(self.x)

def main():
    pso = PSO(1000, 1000)
    pso.evolve()
    plt.show()

if __name__ == '__main__':
    main()