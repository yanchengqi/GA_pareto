# encoding: utf-8
import numpy as np
from Mopso import *


def main(ctr_scores, cvr_scores, ctr_label, cvr_label):
    w = 0.8  # 惯性因子
    c1 = 0.1  # 局部速度因子
    c2 = 0.1  # 全局速度因子
    particals = 100  # 粒子群的数量
    cycle_ = 30  # 迭代次数
    mesh_div = 10  # 网格等分数量
    thresh = 300  # 外部存档阀值
    min_ = np.array([0.3, 0.3])  # 粒子坐标的最小值
    max_ = np.array([1.9, 1.9])  # 粒子坐标的最大值
    mopso_ = Mopso(ctr_scores, cvr_scores, ctr_label, cvr_label,particals, w, c1, c2, max_, min_, thresh, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_)  # 经过cycle_轮迭代后，pareto边界粒子
    print("finish the mopso process")
    for i in range(pareto_in.shape[0]):
        print("the pareto_in is:",str(pareto_in[i][0]),str(pareto_in[i][1]),pareto_fitness[i][0],pareto_fitness[i][1])

if __name__ == "__main__":
    main()
