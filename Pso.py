# 粒子群算法
class PSO():
    def __init__(self, pN, dim, ctr_labels, cvr_labels, ctr_scores, cvr_scores, max_iter):
        # 定义所需变量
        self.w_ini = 0.8
        self.w_end = 0.1
        self.w = self.w_ini
        self.c1 = 2  # 学习因子
        self.c2 = 2

        self.r1 = 0.6  # 超参数
        self.r2 = 0.4

        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数

        # 定义各个矩阵大小
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度矩阵
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置矩阵
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值

        self.X_min = 0.2
        self.X_max = 0.8

        # 读取数据
        self.ctr_labels = ctr_labels
        self.cvr_labels = cvr_labels
        self.ctr_scores = ctr_scores
        self.cvr_scores = cvr_scores
        self.init_Population()

    # 目标函数，通过ctr_auc和cvr_auc来衡量参数效果，暂时采用直接相加
    def function(self, x):
        sum = 0
        scores = [(float(ctr_s) ** x[0]) * (float(cvr_s) ** x[1]) for ctr_s, cvr_s in
                  zip(self.ctr_scores, self.cvr_scores)]

        ctr_auc = metrics.roc_auc_score(list(map(lambda x: float(x), self.ctr_labels)), scores)
        cvr_auc = metrics.roc_auc_score(list(map(lambda x: float(x), self.cvr_labels)), scores)
        # sum += math.log(ctr_auc)+math.log(cvr_auc)
        sum += -1 * (math.log(x[0]) * ctr_auc + math.log(x[1]) * cvr_auc)
        return sum

    # 初始化粒子群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 1)
                self.V[i][j] = random.uniform(0, 0.3)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                if (temp < self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                # 粒子群算法公式
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                self.X[i] = [min(x, self.X_max) for x in self.X[i]]
                self.X[i] = [max(x, self.X_min) for x in self.X[i]]

            self.w = (self.w_ini - self.w_end) * (self.max_iter - t) / self.max_iter + self.w_end
            fitness.append(self.fit)
            print(self.fit)  # 输出最优值
            print(self.gbest)
        return fitness, self.gbest