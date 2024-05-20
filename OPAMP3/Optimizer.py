import sys
import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import user
import Simulator


class optimizer():
    def __init__(self, SimFilePath, SimFileName, LTspiceExec, Model, 
                 sim_sel, lb, ub, circuitsimulation, startpoint, 
                 Avomin, GBWmin, PMmin):
        if(circuitsimulation):
            self.simulator = Simulator.Simulator(SimFilePath=SimFilePath, 
                                    SimFileName=SimFileName, 
                                    LTspiceExec=LTspiceExec, 
                                    Model=Model)
        
        if(len(lb) != len(ub)):
            raise ValueError("len(lb) != len(ub)")
        self.lb, self.ub = lb, ub
        self.dimension = len(lb)
        self.lb_nor, self.ub_nor = 0, 40
        self.nor_to = self.ub_nor - self.lb_nor

        self.num_sim = 0

        self.sim_sel = sim_sel
        self.circuitsimulation = circuitsimulation
        self.startpoint = startpoint

        self.Avomin, self.GBWmin, self.PMmin = Avomin, GBWmin, PMmin

        self.history_iter = []
        self.history_min = []
        self.best_x = []
        self.best_y = np.inf
    # 归一化变量转为实际变量（根据上下界）
    def Nor2Real(self, x_normalize):
        x_real = []
        for i in range(0, len(x_normalize)):
            x_real.append( self.lb[i] + (self.ub[i]-self.lb[i])*x_normalize[i]/self.nor_to )
        return x_real # x_real0, x_real1

    # 实际变量转为归一化变量（根据上下界）
    def Real2Nor(self, x_real):
        x_normalize = []
        print(x_real)
        print(len(x_real))
        for i in range(0, len(x_real)):
            print(i)
            x_normalize.append( (x_real[i]-self.lb[i])*self.nor_to/(self.ub[i]-self.lb[i]) )
            
        return x_normalize # x_normalize0, x_normalize1

    def Violation(self, curr, target, minmax):
        if(minmax == 0): # min
            if(curr>=target):
                return 0
            return (target-curr)/target # math.exp( (target-curr)/(target/10) ) - 1
        else: # max
            if(curr<=target):
                return 0
            return (curr-target)/target # math.exp( (curr-target)/(target/10) ) - 1  

    def ObjectionFunction(self, x_normalize):
        self.num_sim += 1
        
        if(not self.circuitsimulation):
            y = 0
            for i in range(0, len(x_normalize)):
                y += (x_normalize[i]) ** 2
            return y
        
        x = self.Nor2Real(x_normalize)
        a_Width_nm, R, Cc = user.Variable_to_CircuitParameter(x)
        sim_result = self.simulator.Simulator(a_Width_nm, user.my_Length_nm, R, Cc)

        Po = sim_result["P0"]
        Avo = sim_result["AvodB"]
        GBW = sim_result["GBW"]
        PM = sim_result["PMdeg"]
        Ro = sim_result["ro"]
        if(Avo==0 or GBW==0 or PM==0):
            viol = 1E9
        else:
            viol =  Po*1 + Ro/1E7 + \
                    10*self.Violation(Avo, self.Avomin, 0) + \
                    10*self.Violation(GBW, self.GBWmin, 0) + \
                    10*self.Violation(PM, self.PMmin, 0)
        print("\033[1;36m", "violation:", viol, "\033[0m")
        print("---------------------------------------------------------------------------------------")
        return viol

        
    def BO_objFunc(self, trial):
        x = []
        for i in range(self.dimension):
            stri = "x" + str(i)
            x.append(trial.suggest_uniform(stri, 0, self.nor_to))
        return self.ObjectionFunction(x)

    def NGOpt_objfunc(self, x):
        print(x)
        return self.ObjectionFunction(list(x))

    def StartPoint(self):
        if(not self.startpoint):
            return np.random.uniform(0, self.nor_to, size=self.dimension)
        return self.Real2Nor(user.CircuitParameter_to_Variable(a_Width_nm=user.my_Width_nm, R=user.my_R0, C=user.my_Cc))
        # return user.CircuitParameter_to_Variable()
        # pass


    def optimize(self):
        # ---------------------------------- 模拟退火 ---------------------------------------#
        if(self.sim_sel == "SA"):
            print(" start Simulation Annealing...")
            # Simulation Annealing
            from sko.SA import SA
            # init
            sa = SA(
                func=self.ObjectionFunction,     # 优化问题中要优化的函数
                x0=self.StartPoint(),            # 迭代初始点
                T_max=1,                    # 最大温度，模拟退火算法中控制接受较差解的概率的参数，初始时的温度
                T_min=1E-4,                 # 最小温度，模拟退火算法中的终止条件，当温度降至最小温度时停止搜索
                L=100,                      # 链长，表示在每个温度下搜索的步数或迭代次数
                max_stay_counter=100,       # 冷却耗时，表示在温度不变的情况下最多连续多少次没有接受更优解的情况
                lb=self.lb_nor,                  # 每个参数的下界限制
                ub=self.ub_nor                   # 每个参数的上界限制
                )
            start = time.time()
            x_star, y_star = sa.run()
            x_star = self.Nor2Real(x_star)
            print("(x*,y*) = (", x_star, ",", y_star, ")")
            print("exe_num:", self.num_sim)
            print("cost:", time.time()-start, "s")
            np.savetxt(".\\algorithm_result\\sa_history.txt", sa.best_y_history)
            np.savetxt(".\\algorithm_result\\sa_best_x.txt", x_star)
            # plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
            # plt.show()
            self.best_x, self.best_y = x_star, y_star
            self.history_min = sa.best_y_history.copy()
            self.history_iter = np.linspace(start=1, stop=sa.all_iter, num=len(self.history_min))

            # print(self.history_min)
            
        # ---------------------------------- 差分进化 ---------------------------------------#    
        elif(self.sim_sel == "DE"):
            # Differential Evolution
            from sko.DE import DE
            # init
            de = DE(
                func=self.ObjectionFunction, # Objection Function
                n_dim=self.dimension,        # Dimension of variable
                size_pop=50,            # Size of Population, which refers to the number of individuals in each generation.
                max_iter=20,           # Maximum number of iterations
                lb=self.lb_nor,       # lower bound
                ub=self.ub_nor,      # upper bound
                )
            # 修改初始点，使得其中一个个体是手动设计的点
            de.X[0] = self.StartPoint()
            #print(de.X)
            # 启动优化
            start = time.time()
            x_star, y_star = de.run()
            # 将归一化变量转为真实变量
            x_star = self.Nor2Real(x_star)
            # 打印结果
            print("(x*,y*) = (", x_star, ",", y_star, ")")
            print("exe_num:", self.num_sim)
            print("cost:", time.time()-start, "s")
            np.savetxt(".\\algorithm_result\\de_history.txt", de.all_history_Y)
            np.savetxt(".\\algorithm_result\\de_best_x.txt", x_star)
            # plt.plot(pd.DataFrame(de.all_history_Y).cummin(axis=0))
            # plt.show()
            self.best_x, self.best_y = x_star, y_star
            self.history_min = pd.DataFrame(de.all_history_Y).cummin(axis=0).copy()
            self.history_iter = np.linspace(start=de.size_pop, stop=de.size_pop*de.max_iter*2, num=len(self.history_min))
            
        # ---------------------------------- 遗传算法 ---------------------------------------#
        elif(self.sim_sel == "GA"):
            # Genetic Algorithms
            from sko.GA import GA
            # init
            ga = GA(
                func=self.ObjectionFunction, 
                n_dim=self.dimension, 
                size_pop=50, 
                max_iter=40, 
                prob_mut=0.01,     # 变异概率，控制进化过程中发生变异的概率
                lb=self.lb_nor,          # 每个变量的下界限制
                ub=self.ub_nor,          # 每个变量的上界限制
                precision=1e-7      # 精确度
                )
            # 修改初始点，使得其中一个个体是手动设计的点
            _, n = np.array(ga.Chrom).shape
            sp = self.StartPoint()
            # ga.Chrom[0] = tobin(sp[0]/nor_to, int(n/2)) + tobin(sp[1]/nor_to, int(n/2))
            tmp = []
            for i in range(0, self.dimension):
                # print(sp[i]/self.nor_to, len(self.togray(sp[i]/self.nor_to, int(n/self.dimension))))
                tmp = tmp + self.togray(sp[i]/self.nor_to, int(n/self.dimension))
            ga.Chrom[0] = tmp
            # print(ga.Chrom[0])
            # 启动优化
            start = time.time()
            x_star, y_star = ga.run()
            # 将归一化变量转为真实变量
            x_star = self.Nor2Real(x_star)
            # 打印结果
            print('best_x:', x_star, '\n', 'best_y:', y_star)
            print("exe_num:", self.num_sim)
            print("cost:", time.time()-start, "s")
            np.savetxt(".\\algorithm_result\\ga_history.txt", ga.all_history_Y)
            np.savetxt(".\\algorithm_result\\ga_best_x.txt", x_star)
            # Plot the result
            # Y_history = pd.DataFrame(ga.all_history_Y)
            # print(Y_history)
            # fig, ax = plt.subplots(2, 1)
            # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
            # Y_history.min(axis=1).cummin().plot(kind='line', color='blue')
            # plt.show()
            self.best_x, self.best_y = x_star, y_star
            self.history_min = pd.DataFrame(ga.all_history_Y).min(axis=1).cummin().copy()
            self.history_iter = np.linspace(start=ga.size_pop, stop=ga.size_pop*ga.max_iter, num=len(self.history_min))

        # --------------------------------- 粒子群算法 --------------------------------------#
        elif(self.sim_sel == "PSO"):
            
            from sko.PSO import PSO

            pso = PSO(
                func=self.ObjectionFunction,     # 待优化的目标函数
                n_dim=self.dimension,            # 问题的维度，即目标函数的自变量个数
                pop=50,                     # 种群规模，即粒子群中包含的粒子数量
                max_iter=40,               # 最大迭代次数，即PSO算法运行的最大迭代次数
                lb=self.lb_nor,                  # 每个变量的下界限制
                ub=self.ub_nor,                  # 每个变量的上界限制
                w=0.8,                      # 惯性权重，用于平衡粒子的飞行速度和方向
                c1=0.5,                     # 个体认知因子，用于控制粒子根据个体最优位置调整自身速度和位置
                c2=0.5,                     # 社会认知因子，用于控制粒子根据群体最优位置调整自身速度和位置
                add_startpoint = np.array(self.StartPoint())    # 修改初始点，使得其中一个个体是手动设计的点
                )
            # 启动优化
            start = time.time()
            pso.run()
            print(pso.my_all_X)
            print(pso.my_all_Y)
            print('best_x is ', self.Nor2Real(pso.gbest_x), 'best_y is', pso.gbest_y)
            print("exe_num:", self.num_sim)
            print("cost:", time.time()-start, "s")
            np.savetxt(".\\algorithm_result\\pso_history.txt", pso.gbest_y_hist)
            np.savetxt(".\\algorithm_result\\pso_best_x.txt", self.Nor2Real(pso.gbest_x))
            # Plot the result
            # plt.plot(pso.gbest_y_hist, color='red')
            # plt.show()
            self.best_x, self.best_y = self.Nor2Real(pso.gbest_x), pso.gbest_y
            self.history_min = pso.gbest_y_hist
            self.history_iter = np.linspace(start=pso.pop, stop=pso.pop*pso.max_iter, num=len(self.history_min))

        # --------------------------------- 贝叶斯优化 --------------------------------------#
        elif(self.sim_sel == "BO"):
            # https://blog.csdn.net/weixin_43819931/article/details/125172266
            import optuna
            study = optuna.create_study(
                    sampler=optuna.samplers.TPESampler(
                            n_startup_trials=100,
                            n_ei_candidates=50
                            )
                    )
            start = time.time()
            study.optimize(
                self.BO_objFunc, 
                n_trials=2000
                )
            print("best_trial", study.best_trial)
            print("best_params", study.best_params)
            x_nor = []
            for i in range(self.dimension):
                stri = "x" + str(i)
                x_nor.append(study.best_params[stri])
            x_real = self.Nor2Real(x_nor)

            bo_history = []
            bo_tmp = 1E9
            for tr in study.trials:
                if(tr.values[0] < bo_tmp):
                    bo_tmp = tr.values[0]
                bo_history.append(bo_tmp)
            
            print("bo_history:", bo_history)

            print("best_x:", x_real)
            print("best_value", study.best_value)
            print("exe_num:", self.num_sim)
            print("cost:", time.time()-start, "s")
            np.savetxt(".\\algorithm_result\\bo_history.txt", bo_history)
            np.savetxt(".\\algorithm_result\\bo_best_x.txt", x_real)
            # plt.plot(bo_history)
            # plt.show()
            self.best_x, self.best_y = x_real, study.best_value
            self.history_min = bo_history # [i for d2 in bo_history for d1 in d2 for i in d1]
            self.history_iter = np.linspace(start=1, stop=len(self.history_min), num=len(self.history_min))
        
        elif(self.sim_sel == "PSOGA"):
            from Algorithm import PSO_GA
            pso_ga = PSO_GA.myPSO(
                        func=self.ObjectionFunction,     # 待优化的目标函数
                        n_dim=self.dimension,            # 问题的维度，即目标函数的自变量个数
                        pop=50,                     # 种群规模，即粒子群中包含的粒子数量
                        max_iter=40,               # 最大迭代次数，即PSO算法运行的最大迭代次数
                        lb=self.lb_nor,                    # 每个变量的下界限制
                        ub=self.ub_nor,                    # 每个变量的上界限制
                        w=0.8,                      # 惯性权重，用于平衡粒子的飞行速度和方向
                        c1=0.99,                     # 个体认知因子，用于控制粒子根据个体最优位置调整自身速度和位置
                        c2=0.01,                     # 社会认知因子，用于控制粒子根据群体最优位置调整自身速度和位置
                        c1end=0.90, 
                        eigx=40,
                        add_startpoint = np.array(self.StartPoint())
                        )
            start = time.time()
            pso_ga.run()
            print('best_x is ', self.Nor2Real(pso_ga.gbest_x), 'best_y is', pso_ga.gbest_y)
            print("exe_num:", self.num_sim)
            print("cost:", time.time()-start, "s")
            np.savetxt(".\\algorithm_result\\psoga_history.txt", pso_ga.gbest_y_hist)
            np.savetxt(".\\algorithm_result\\psoga_best_x.txt", self.Nor2Real(pso_ga.gbest_x))
            # plt.plot(pso_ga.gbest_y_hist, color='green')
            # plt.show()
            self.best_x, self.best_y = self.Nor2Real(pso_ga.gbest_x), pso_ga.gbest_y
            self.history_min = pso_ga.gbest_y_hist
            self.history_iter = np.linspace(start=pso_ga.pop, stop=pso_ga.pop*pso_ga.max_iter, num=len(self.history_min))
            
        elif(self.sim_sel == "GANG"):
            import nevergrad as ng
            from Algorithm import NGOpt
            from sko.GA import GA
            ga = GA(
                func=self.ObjectionFunction, 
                n_dim=self.dimension, 
                size_pop=50, 
                max_iter=8, 
                prob_mut=0.01,     # 变异概率，控制进化过程中发生变异的概率
                lb=self.lb_nor,          # 每个变量的下界限制
                ub=self.ub_nor,          # 每个变量的上界限制
                precision=1e-7      # 精确度
                )
            _, n = np.array(ga.Chrom).shape
            sp = self.StartPoint()
            # ga.Chrom[0] = tobin(sp[0]/nor_to, int(n/2)) + tobin(sp[1]/nor_to, int(n/2))
            tmp = []
            for i in range(0, self.dimension):
                print(sp[i]/self.nor_to, len(self.togray(sp[i]/self.nor_to, int(n/self.dimension))))
                tmp = tmp + self.togray(sp[i]/self.nor_to, int(n/self.dimension))
            ga.Chrom[0] = tmp
            # print(ga.Chrom[0])
            # 启动优化
            start = time.time()
            x_star, y_star = ga.run()
            his_x, his_y = ga.my_all_X, ga.my_all_Y
            print("g_best_his_x:", his_x, "\n", "g_best_his_y:", his_y)
            print("GA result:", x_star, y_star)
            ga_history = ga.generation_best_Y
            ngopt = NGOpt.NeverGradOptimizer(
                            param = ng.p.Array(shape=(self.dimension,)).set_bounds(lower=self.lb_nor, upper=self.ub_nor), 
                            objfunc=self.NGOpt_objfunc,
                            budget=200
                            )
            for i in range(len(his_y)):
                candidate = ngopt.optimizer.parametrization.spawn_child(new_value=np.array(list(his_x[i])))
                ngopt.optimizer.tell(candidate, his_y[i])
            ngopt.run()
            print(self.Nor2Real(ngopt.x_best), ngopt.y_best)
            # print(ngopt.x_best, ngopt.y_best)
            np.savetxt(".\\algorithm_result\\gang_ga.txt", ga.generation_best_Y)
            np.savetxt(".\\algorithm_result\\gang_history.txt", ngopt.y_history)
            np.savetxt(".\\algorithm_result\\gang_best_x.txt", self.Nor2Real(ngopt.x_best))
            # Y_history = pd.DataFrame(ga.all_history_Y)
            # Y_history.min(axis=1).cummin().plot(kind='line', color='orange')
            # plt.plot(pd.DataFrame(ngopt.y_history).cummin(), color="orange")
            # plt.show()
            self.best_x, self.best_y = self.Nor2Real(ngopt.x_best), ngopt.y_best
            ga_his = pd.DataFrame(ga.all_history_Y).min(axis=1).cummin().values.tolist()
            ng_his = sum(pd.DataFrame(ngopt.y_history).cummin().values.tolist(), [])
            print("------------------------------------------------")
            print(ga_his)
            print()
            print(type(ng_his))
            print("------------------------------------------------")
            self.history_min = ga_his + ng_his
            # print(self.history_min.s)
            self.history_iter = list(np.linspace(start=ga.size_pop, stop=ga.size_pop*ga.max_iter, num=len(ga_his))) + \
                                list(np.linspace(start=ga.size_pop*ga.max_iter+1, stop=ga.size_pop*ga.max_iter+len(ng_his), num=len(ng_his)))
            
        # ------------------------------------ 测试 -----------------------------------------#
        else:
            print(self.sim_sel)


    def togray(self, x, N):
        if x == 1:
            x = x - pow(2, -N)
        if x > 1:
            return []
        bin = []
        gray = []
        for i in range(N):
            digit = 1 if x >= 0.5 else 0
            bin.append(digit)
            graydig = digit if i==0 else (bin[i] + bin[i-1])
            # print(i, digit, graydig)
            gray.append(graydig % 2)
            x = (x % 0.5) * 2
        return gray