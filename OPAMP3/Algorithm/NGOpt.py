import nevergrad as ng
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class NeverGradOptimizer():
    def __init__(self, param, objfunc, budget):
        self.objfunc = objfunc
        self.params = param
        self.optimizer = ng.optimizers.NGOpt39(parametrization=self.params, budget=budget)
        self.y_history = []
        self.x_history = []
        self.y_best = 1E9
        self.x_best = []

    def run(self):
        # recommendation = self.optimizer.minimize(objective_function=self.objfunc)
        # print("recommendation value:", recommendation.value)
        for _ in range(self.optimizer.budget):
            suggestion = self.optimizer.ask()
            loss = self.objfunc(suggestion.value)
            # print(suggestion.value)
            self.optimizer.tell(suggestion, loss)

            self.x_history.append(suggestion.value)
            self.y_history.append(loss)
            if(loss < self.y_best):
                self.x_best, self.y_best = suggestion.value, loss


if __name__ == "__main__":

    def func(x):
        y = 0
        y = (x[0]-3) ** 2 + (x[1]-1) ** 2
        return  y

    def objfunc(x):
        print(x)
        return func(list(x))

    ngopt = NeverGradOptimizer(
                    param = ng.p.Array(shape=(2,)).set_bounds(lower=0, upper=4), 
                    objfunc=objfunc,
                    budget=100)
    ngopt.run()
    print(ngopt.x_best, ngopt.y_best)
    plt.plot(pd.DataFrame(ngopt.y_history).cummin())
    plt.show()

    # from sko.PSO import PSO
    # pso = PSO(
    #     func=func,     # 待优化的目标函数
    #     n_dim=2,            # 问题的维度，即目标函数的自变量个数
    #     pop=50,                     # 种群规模，即粒子群中包含的粒子数量
    #     max_iter=200,               # 最大迭代次数，即PSO算法运行的最大迭代次数
    #     lb=[0, 0],                  # 每个变量的下界限制
    #     ub=[4, 4],                  # 每个变量的上界限制
    #     w=0.8,                      # 惯性权重，用于平衡粒子的飞行速度和方向
    #     c1=0.5,                     # 个体认知因子，用于控制粒子根据个体最优位置调整自身速度和位置
    #     c2=0.5,                     # 社会认知因子，用于控制粒子根据群体最优位置调整自身速度和位置
    #     )
    # pso.run()
    # print(pso.best_x, pso.best_y)
    # plt.plot(pso.gbest_y_hist, color='red')
    # plt.show()
# optimization on x as an array of shape (2,)
# param = ng.p.Instrumentation(ng.p.Array(shape=(2,)))
# optimizer = ng.optimizers.NGOpt(parametrization=param, budget=100)
# optimizer.suggest()
# candicate = optimizer.ask()
# print(candicate)
# print(candicate.args[0])
# print(candicate.kwargs)
# optimizer.suggest(np.array([0.5, 0.5]))
# candidate = optimizer.ask()
# print(candidate)
# optimizer.tell(candidate, 1)
# recommendation = optimizer.minimize(square)  # best value
# print(recommendation.value[0][0], recommendation.loss)
# print(sorted(ng.optimizers.registry.keys()))

# param = ng.p.Dict(
#     # logarithmically distributed float
#     log=ng.p.Log(lower=0.01, upper=1.0),
#     # one-dimensional array of length 2
#     array=ng.p.Array(shape=(2,)),
#     # character, either "a" or "b or "c".
#     ch=ng.p.Choice(["a", "b", "c"]),
# )
# print(param.value)

# # create a new instance
# child = param.spawn_child()
# # update its value
# child.value = {"log": 0.2, "array": np.array([12.0, 13.0]), "ch": "c"}

# data = child.get_standardized_data(reference=param)
# print(data)

# arg1 = ng.p.Choice(["Helium", "Nitrogen", "Oxygen"]) # choice 通过softmax，几个元素就占用几维
# arg2 = ng.p.TransitionChoice(["Solid", "Liquid", "Gas"])
# values = ng.p.Tuple(ng.p.Scalar().set_integer_casting(), ng.p.Scalar())
# instru = ng.p.Instrumentation(arg1, arg2, "blublu", amount=values)
# print(instru.dimension)

# def myfunction(arg1, arg2, arg3, amount=(2, 2)):
#     print(arg1, arg2, arg3, amount)
#     return amount[0] ** 2 + amount[1] ** 2

# optimizer = ng.optimizers.NGOpt(parametrization=instru, budget=10)
# recommendation = optimizer.minimize(myfunction)
# print(recommendation.value)


# params = ng.p.Tuple(ng.p.Scalar(), ng.p.Scalar())
# print(params)
# optimizer = ng.optimizers.CMA(parametrization=params, budget=100)
# for iter in range(optimizer.budget):
#     suggestion = optimizer.ask()
#     loss = objfunc(suggestion.value)
#     optimizer.tell(suggestion, loss)
#     x_history.append(suggestion.value)
#     y_history.append(loss)
#     if(loss < y_best):
#         x_best, y_best = suggestion.value, loss
# print(x_best, y_best)
# plt.plot(pd.DataFrame(y_history).cummin())
# plt.show()


