import numpy as np

class Population():
    def __init__(self, X, y, ub, lb) -> None:
        self.ub = ub
        self.lb = lb
        self.N = 20
        try:
            self.dimension = len(lb)
        except:
            self.dimension = 1
        self.mutationrate = 0.01
        self.crossrate = 0.8
        self.param_check()
        self.setXy(X, y)


    def param_check(self):
        # # lb, ub
        # if(len(self.ub) != len(self.lb)):
        #     raise ValueError("length of lb and ub is not equal")
        # for i in range(len(self.ub)):
        #     if(self.ub[i] <= self.lb[i]):
        #         print(self.ub[i], self.lb[i])
        #         raise ValueError(f"lb{i} >= ub{i}")
        pass

    def setXy(self, X, y):
        self.X = X
        self.y = y
        self.pop = len(y)
        self.Xgray = []
        self.new_Xgray = []
        self.new_X = []
        self.new_y = []
        self.addoneitem = self.pop % 2 == 1
        self.best_x, self.best_y = [], np.inf


    def x2gray(self):
        # print("------------- x -> gray -------------")
        # print(self.X)
        Xnp = np.array(self.X)
        Xnor = np.zeros_like(Xnp, dtype=float)
        m, n = Xnp.shape
        for i in range(n):
            Xnor[:, i] = (Xnp[:, i] - self.lb[i]) / (self.ub[i] - self.lb[i])
        # print(Xnor)
        Xgray = []
        for i in range(m):
            Xigray = []
            for j in range(n):
                Xijgray = self.togray(Xnor[i, j])
                Xigray = Xigray + Xijgray
            Xgray.append(Xigray)
        self.Xgray = Xgray.copy()
        # print(np.array(self.Xgray))


    def gray2x(self):
        # print("------------- gray -> x -------------")
        if(len(self.new_Xgray) == 0):
            raise ValueError("Xbin is empty")
        X = []
        # print(np.array(self.Xgray))
        for Xigray in self.new_Xgray:
            Xi = []
            for i in range(int(len(Xigray)/self.N)):
                Xij = self.tox(Xigray[i*self.N:(i+1)*self.N])
                Xij = self.lb[i] + Xij * (self.ub[i]-self.lb[i])
                Xi.append(Xij)
            X.append(Xi)
        self.new_X = X
        # print(self.X)
    
    def selection(self):
        # print("------------- selection -------------")
        fitness = np.array(self.y)
        # fitness[fitness > 1000] = np.inf
        fitness = 1/(np.log10(fitness) + 5)
        idx = np.random.choice(np.arange(self.pop), 
                               size=self.pop, 
                               replace=True, 
                               p=(fitness)/(fitness.sum()) )
        # print(idx)
        X, y = [], []
        for i in idx:
            X.append(self.X[i])
            y.append(self.y[i])
        self.new_X, self.new_y = X.copy(), y.copy()
        # print("selection:", len(self.new_X), len(self.new_y))

        return
    
    def crossover(self):
        # print("------------- crossover -------------")
        X = []
        for i in range(int(len(self.Xgray) / 2)):
            pidx1, pidx2 = np.random.choice(np.arange(self.pop), size=2, replace=False)
            if(np.random.rand() < self.crossrate):
                co_point1 = np.random.randint(1, self.N * self.dimension)
                co_point2 = np.random.randint(1, self.N * self.dimension)
                if(co_point1 > co_point2):
                    co_point1, co_point2 = co_point2, co_point1
                child1 = self.Xgray[pidx1][:co_point1] + self.Xgray[pidx2][co_point1:co_point2] + self.Xgray[pidx1][co_point2:]
                child2 = self.Xgray[pidx2][:co_point1] + self.Xgray[pidx1][co_point1:co_point2] + self.Xgray[pidx2][co_point2:]
                # print("parent1:", 
                #       "\033[1;35m", self.Xgray[pidx1][:co_point1], 
                #       "\033[1;35m", self.Xgray[pidx1][co_point1:co_point2], 
                #       "\033[1;35m", self.Xgray[pidx1][co_point2:], 
                #       "\033[0m")
                # print("parent2:", 
                #       "\033[1;36m", self.Xgray[pidx2][:co_point1], 
                #       "\033[1;36m", self.Xgray[pidx2][co_point1:co_point2], 
                #       "\033[1;36m", self.Xgray[pidx2][co_point2:], 
                #       "\033[0m")
                # print("child1 :", 
                #       "\033[1;35m", child1[:co_point1], 
                #       "\033[1;36m", child1[co_point1:co_point2], 
                #       "\033[1;35m", child1[co_point2:], 
                #       "\033[0m")
                # print("child2 :", 
                #       "\033[1;36m", child2[:co_point1], 
                #       "\033[1;35m", child2[co_point1:co_point2], 
                #       "\033[1;36m", child2[co_point2:], 
                #       "\033[0m")
            else:
                child1 = self.Xgray[pidx1]
                child2 = self.Xgray[pidx2]
            X.append(child1)
            X.append(child2)
        if(self.addoneitem == True):
            co_point = np.random.randint(1, len(np.arange(self.pop)))
            childadd = self.Xgray[pidx1][:co_point] + self.Xgray[pidx2][co_point:]
            X.append(childadd)
        self.new_Xgray = X.copy()
        return
    
    """
    def crossover_2point_bit(self):
        '''
        3 times faster than `crossover_2point`, but only use for 0/1 type of Chrom
        :param self:
        :return:
        '''
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        half_size_pop = int(size_pop / 2)
        Chrom1, Chrom2 = Chrom[:half_size_pop], Chrom[half_size_pop:]
        mask = np.zeros(shape=(half_size_pop, len_chrom), dtype=int)
        for i in range(half_size_pop):
            n1, n2 = np.random.randint(0, self.len_chrom, 2)
            if n1 > n2:
                n1, n2 = n2, n1
            mask[i, n1:n2] = 1
        mask2 = (Chrom1 ^ Chrom2) & mask
        Chrom1 ^= mask2
        Chrom2 ^= mask2
        return self.Chrom
    """
    
    def mutation(self):
        # print("------------- mutation -------------")
        for i in range(len(self.new_Xgray)):
            if(np.random.rand() < self.mutationrate): #以MUTATION_RATE的概率进行变异
                # print(self.new_Xgray[i])
                mutate_point = np.random.randint(0, self.dimension*self.N)	#随机产生一个实数，代表要变异基因的位置
                # print(i, len(self.new_Xgray), mutate_point, self.dimension*self.N)
                # print(self.new_Xgray[i])
                self.new_Xgray[i][mutate_point] = (self.new_Xgray[i][mutate_point] + 1) % 2 	#将变异点的二进制为反转
                # print(self.new_Xgray[i][:mutate_point], 
                #       "\033[1;35m", self.new_Xgray[i][mutate_point], 
                #       "\033[0m", self.new_Xgray[i][mutate_point+1:])

    def nutation_v2(self):
        new_Xgray_np = np.array(self.new_Xgray)
        # print(new_Xgray_np)
        mask = (np.random.rand(self.pop, self.N*self.dimension) < self.mutationrate)
        # print(mask)
        new_Xgray_np ^= mask
        # print(new_Xgray_np)
        self.new_Xgray = new_Xgray_np.tolist()
        

        return
    
    def togray(self, x):
        N = self.N
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

    def tox(self, xgray):
        xbin = []
        x = 0
        w = 0.5
        if(self.N != len(xgray)):
            raise ValueError("self.N != len(xgray)")
        for i in range(self.N):
            digit = xgray[i] if i==0 else (xbin[i-1]+xgray[i])
            xbin.append(digit%2)
            x += w * (digit%2)
            w /= 2
        return x

    def run(self):
        self.selection()
        # print(self.X, self.y)
        self.x2gray()
        # print("gray len:", len(self.Xgray))
        self.crossover()
        self.nutation_v2()
        self.gray2x()
        # print(self.new_X)
        return self.new_X


from sko.GA import GA
class myGA(GA):
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.001, lb=-1, ub=1, constraint_eq=tuple(), constraint_ueq=tuple(), precision=1e-7, early_stop=None):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, lb, ub, constraint_eq, constraint_ueq, precision, early_stop)

        self.N = self.Lind[0]

    def setXy(self, X, y):
        self.X = X
        self.Y = y
        self.size_pop = len(y)
        self.x2gray()
        # print(self.size_pop, self.Chrom)
        

    def togray(self, x):
        N = self.N
        if x == 1:
            x = x - 1/pow(2, N)
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
    
    def x2gray(self):
        # print("------------- x -> gray -------------")
        # print(self.X)
        Xnp = self.X
        Xnor = np.zeros_like(Xnp, dtype=float)
        m, n = Xnp.shape
        for i in range(n):
            Xnor[:, i] = (Xnp[:, i] - self.lb[i]) / (self.ub[i] - self.lb[i])
        # print(Xnor)
        Xgray = []
        for i in range(m):
            Xigray = []
            for j in range(n):
                Xijgray = self.togray(Xnor[i, j])
                Xigray = Xigray + Xijgray
            Xgray.append(Xigray)
        self.Chrom = np.array(Xgray.copy())
        # print(np.array(self.Xgray))

    def tox(self, xgray):
        xbin = []
        x = 0
        w = 0.5
        if(self.N != len(xgray)):
            raise ValueError("self.N != len(xgray)")
        for i in range(self.N):
            digit = xgray[i] if i==0 else (xbin[i-1]+xgray[i])
            xbin.append(digit%2)
            x += w * (digit%2)
            w /= 2
        return x

    def gray2x(self):
        # print("------------- gray -> x -------------")
        if(len(self.Chrom) == 0):
            raise ValueError("Xbin is empty")
        X = []
        # print(np.array(self.Xgray))
        for Xigray in self.Chrom:
            Xi = []
            for i in range(int(len(Xigray)/self.N)):
                Xij = self.tox(Xigray[i*self.N:(i+1)*self.N])
                Xij = self.lb[i] + Xij * (self.ub[i]-self.lb[i])
                Xi.append(Xij)
            X.append(Xi)
        self.new_X = X
        # print(self.X)
    
    def run(self):
        # print("A:", self.chrom2x(self.Chrom))
        self.ranking()
        self.selection()
        self.crossover()
        self.mutation()
        return self.chrom2x(self.Chrom)
        # print("B:", self.chrom2x(self.Chrom))



if __name__ == "__main__":
    lb = [0, 0]
    ub = [1, 1]
    X = [[0.5, 0.4], [0.9, 0.1], [0.22, 0.45], [0.77, 0.29]]
    y = [9, 2, 5, 2.3]
    # pop = Population(X, y, ub, lb)
    # # xg = pop.togray(0.75)
    # # print(xg)
    # # x = pop.tox(xg)
    # # print(x)
    # pop.run()

    ga = myGA(
        func=[], 
        n_dim=2, 
        size_pop=50, 
        max_iter=200, 
        prob_mut=0.001,     # 变异概率，控制进化过程中发生变异的概率
        lb=lb,          # 每个变量的下界限制
        ub=ub,          # 每个变量的上界限制
        precision=1e-7      # 精确度
        )
    
    ga.setXy(np.array(X), np.array(y))
    ga.run()
