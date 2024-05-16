from sko.PSO import PSO
import math
from matplotlib import pyplot as plt
import numpy as np

from Algorithm.population import Population
from Algorithm.population import myGA



class myPSO(PSO):
    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.99, c2=0.01, c1end=0.6, 
                 eigx=10, constraint_eq=tuple(), constraint_ueq=tuple(), verbose=False, dim=None, add_startpoint=[]):
        super().__init__(func, n_dim, pop, max_iter, lb, ub, w, c1, c2, 
                         constraint_eq, constraint_ueq, verbose, dim, add_startpoint)
        # 个体      种群
        # self.cp, self.cg = c1, c2
        self.c1, self.c2, self.c1end = c1, c2, c1end
        self.k_pso_ga1, self.k_pso_ga2 = 0.2, 0.95
        self.eigx = eigx

        self.gapop = Population([], [], lb=lb, ub=ub)
        self.gav2pop = myGA(
                        func=[], 
                        n_dim=n_dim, 
                        size_pop=pop, 
                        max_iter=max_iter, 
                        prob_mut=0.01,  # 变异概率，控制进化过程中发生变异的概率
                        lb=lb,          # 每个变量的下界限制
                        ub=ub,          # 每个变量的上界限制
                        precision=1e-7      # 精确度
                        )

    def update_C(self):
        self.cp = (self.c1 - self.c1end) * math.exp(-self.iter_num/self.eigx) + self.c1end
        self.cg = 1 - self.cp
        # self.k_pso_ga =  self.k_pso_ga2 + (self.k_pso_ga1 - self.k_pso_ga2) * math.exp(-self.iter_num/5)
        self.k_pso_ga =  self.k_pso_ga1 if self.iter_num < 40 else self.k_pso_ga2
    
    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)
        '''
        personal worst
        '''
        self.worst_update = self.pworst_y <= self.Y
        for idx, x in enumerate(self.X):
            if self.worst_update[idx]:
                self.worst_update[idx] = self.check_constraint(x)

        self.pworst_x = np.where(self.worst_update, self.X, self.pworst_x)
        self.pworst_y = np.where(self.worst_update, self.Y, self.pworst_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]
        '''
        global worst
        :return:
        '''
        idx_max = self.pworst_y.argmax()
        if self.gworst_y <= self.pworst_y[idx_max]:
            self.gworst_x = self.X[idx_max, :].copy()
            self.gworst_y = self.pworst_y[idx_max]

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V  \
               + self.cp * r2 * (self.pbest_x - self.X) \
               + self.cg * r1 * (self.gbest_x - self.X)
        # if(self.iter_num<self.max_iter/2):
        #     r3 = np.random.rand(self.pop, self.n_dim)
        #     self.V = self.V \
        #             - r3/self.max_iter * (self.pworst_x - self.X) \
                    # - self.cp * r2/10 * (self.gworst_x - self.X)
        self.update_C()

    def update_X(self):
        if(self.k_pso_ga <= 1):
            self.GA_update_X()
        self.X = self.X + self.k_pso_ga * self.V
        self.X = np.clip(self.X, self.lb, self.ub)
    
    def GA_update_X(self):
        if(False):
            self.gapop.setXy(self.X.tolist(), self.Y.flatten().tolist())
            self.X = self.gapop.run()
        else:
            # print(self.X, self.Y)
            self.gav2pop.setXy(self.X, self.Y.flatten())
            self.X = self.gav2pop.run()



if __name__ == "__main__":
    def objfunc(x):
        if(-2<x[0]<-1 or 2<x[0]<3 or 4<x[0]<5):
            return 10000
        y = 100+ \
            -2*np.cos(x[0]+3)*np.exp(-abs(x[0]+3)) + \
            -10*np.cos(x[0]-10)*np.exp(-abs(x[0]-10)) + \
            +2*np.sin(100*x[0]+3) + np.sin(10*x[0]+3)
        return y
        

    exenum = 50
    psotobest = 0
    for i in range(exenum):
        pso = PSO(
            func=objfunc,     # 待优化的目标函数
            n_dim=1,            # 问题的维度，即目标函数的自变量个数
            pop=24,                     # 种群规模，即粒子群中包含的粒子数量
            max_iter=50,               # 最大迭代次数，即PSO算法运行的最大迭代次数
            lb=[-5],                  # 每个变量的下界限制
            ub=[15],                  # 每个变量的上界限制
            w=0.8,                      # 惯性权重，用于平衡粒子的飞行速度和方向
            c1=0.5,                     # 个体认知因子，用于控制粒子根据个体最优位置调整自身速度和位置
            c2=0.5,                     # 社会认知因子，用于控制粒子根据群体最优位置调整自身速度和位置
            )
        # 启动优化
        pso.run()
        print(f'{i}th:', 'best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        if(i == 0):
            psohis = pso.gbest_y_hist.copy()
        else:
            psohis = np.add(psohis, pso.gbest_y_hist)
        if(-3.5<pso.best_x<-1):
            psotobest += 1
    mypsotobest = 0
    for i in range(exenum):
        mypso = myPSO(
            func=objfunc,     # 待优化的目标函数
            n_dim=1,            # 问题的维度，即目标函数的自变量个数
            pop=24,                     # 种群规模，即粒子群中包含的粒子数量
            max_iter=50,               # 最大迭代次数，即PSO算法运行的最大迭代次数
            lb=[-5],                  # 每个变量的下界限制
            ub=[15],                  # 每个变量的上界限制
            w=0.8,                      # 惯性权重，用于平衡粒子的飞行速度和方向
            c1=0.99,                     # 个体认知因子，用于控制粒子根据个体最优位置调整自身速度和位置
            c2=0.01,                     # 社会认知因子，用于控制粒子根据群体最优位置调整自身速度和位置
            c1end=0.5, 
            eigx=15
            )
        # 启动优化
        mypso.run()
        print(f'{i}th:', 'best_x is ', mypso.gbest_x, 'best_y is', mypso.gbest_y)
        if(i == 0):
            mypsohis = mypso.gbest_y_hist.copy()
        else:
            mypsohis = np.add(mypsohis, mypso.gbest_y_hist)
        if(-3.5<mypso.best_x<-1):
            mypsotobest += 1
    print("pso:", psotobest/exenum*100, "%, mypso:", mypsotobest/exenum*100, "%")
    plt.plot([value/exenum for value in psohis], color="blue")
    plt.plot([value/exenum for value in mypsohis], color="red")
    plt.show()

