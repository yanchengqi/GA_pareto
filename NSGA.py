# encoding: utf-8
import random

import numpy as np
from sklearn import metrics
import math


class NSGA2:
    def __init__(self, max_gen, part_size, minx, maxx, scores1, scores2, labels1, labels2):
        self.max_gen = max_gen
        self.part_size = part_size
        self.min_x = minx
        self.max_x = maxx
        self.ctr_s = scores1
        self.cvr_s = scores2
        self.label_ctr = labels1
        self.label_cvr = labels2

    def cal_ctr_auc(self, scores):
        # 定义两个目标收益
        ctr_auc = metrics.roc_auc_score(self.label_ctr, scores)
        return ctr_auc

    def cal_cvr_auc(self, scores):
        cvr_auc = metrics.roc_auc_score(self.label_cvr, scores)
        return cvr_auc

    def cal_scores(self, k, m):
        # 计算融合分数
        scores = [self.cvr_s[i] ** m * self.ctr_s[i] ** k for i in range(len(self.cvr_s))]
        return scores

    def index_of(self, a,list1):
        for i in range(0, len(list1)):
            if list1[i] == a:
                return i
        return -1

    def sort_by_values(self, list1, values):
        sorted_list = []
        while (len(sorted_list) != len(list1)):
            if self.index_of(min(values), values) in list1:
                sorted_list.append(self.index_of(min(values), values))
            values[self.index_of(min(values), values)] = 4444444444444445
        return sorted_list

    def non_dominated_sort(self, values1, values2):
        S = [[] for i in range(0, len(values1))]
        front = [[]]
        n = [0 for i in range(0, len(values1))]
        rank = [0 for i in range(0, len(values1))]

        for p in range(0, len(values1)):
            S[p] = []
            n[p] = 0
            for q in range(0, len(values1)):
                if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                        values1[p] >= values1[q] and values2[p] > values2[q]) or (
                        values1[p] > values1[q] and values2[p] >= values2[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                        values1[q] >= values1[p] and values2[q] > values2[p]) or (
                        values1[q] > values1[p] and values2[q] >= values2[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while (front[i] != []):
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            front.append(Q)

        del front[len(front) - 1]
        return front

    def crowding_distance(self, values1, values2, front):
        distance = [0 for i in range(0, len(front))]
        sorted1 = self.sort_by_values(front, values1)
        sorted2 = self.sort_by_values(front, values2)
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (
                    max(values1) - min(values1)+1)
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                    max(values2) - min(values2)+1)
        return distance

    def crossover(self, parent1, parent2):
        r = random.random()
        if r > 0.5:
            return self.mutation((parent1 + parent2) / 2)
        else:
            return self.mutation((parent1 - parent2) / 2)

    def mutation(self, parent):
        mutation_prob = random.random()
        if mutation_prob < 1:
            solution = self.min_x + (self.max_x - self.min_x) * random.random()
        return solution

    def cal_fitness(self, solution_ctr_2, solution_cvr_2):
        scores = [self.cal_scores(solution_ctr_2[i], solution_cvr_2[i]) for i in range(len(solution_cvr_2))]
        ctr_values = [self.cal_ctr_auc(scores[i]) for i in range(len(scores))]
        cvr_values = [self.cal_cvr_auc(scores[i]) for i in range(len(scores))]
        return ctr_values, cvr_values

    def run(self):
        # 1.初始化种群
        solution_ctr = [i for i in np.arange(self.min_x, self.max_x, (self.max_x - self.min_x) / self.part_size)]
        solution_cvr = [i for i in np.arange(self.min_x, self.max_x, (self.max_x - self.min_x) / self.part_size)]

        for i in range(self.max_gen):
            # 2计算收益
            ctr_values, cvr_values = self.cal_fitness(solution_ctr, solution_cvr)
            # 3计算 parote front
            non_dominated_sorted_solution = self.non_dominated_sort(ctr_values, cvr_values)
            # 4根据拥挤度
            crowd_distance_values = []
            for j in range(len(non_dominated_sorted_solution)):
                crowd_distance_values.append(
                    self.crowding_distance(ctr_values, cvr_values, non_dominated_sorted_solution[j]))
            solution_ctr_2 = solution_ctr
            solution_cvr_2 = solution_cvr
            # 5 generate child
            while (len(solution_ctr_2) != 2 * self.part_size):
                pos1 = random.randint(0, self.part_size - 1)
                pos2 = random.randint(0, self.part_size - 1)
                solution_ctr_2.append(self.crossover(solution_ctr[pos1], solution_ctr[pos2]))
                solution_cvr_2.append(self.crossover(solution_cvr[pos1], solution_cvr[pos2]))

            ctr_values_2, cvr_values_2 = self.cal_fitness(solution_ctr_2, solution_cvr_2)
            non_dominated_sorted_solution_2 = self.non_dominated_sort(ctr_values_2, cvr_values_2)

            crowding_distance_values2 = []
            for j in range(0, len(non_dominated_sorted_solution_2)):
                crowding_distance_values2.append(
                    self.crowding_distance(ctr_values_2, cvr_values_2, non_dominated_sorted_solution_2[j]))
            new_solution = []
            for k in range(0, len(non_dominated_sorted_solution_2)):
                non_dominated_sorted_solution2_1 = [
                    self.index_of(non_dominated_sorted_solution_2[k][j], non_dominated_sorted_solution_2[k]) for j in
                    range(0, len(non_dominated_sorted_solution_2[k]))]
                front22 = self.sort_by_values(non_dominated_sorted_solution2_1, crowding_distance_values2[k])
                front = [non_dominated_sorted_solution_2[k][front22[j]] for j in
                         range(0, len(non_dominated_sorted_solution_2[k]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if (len(new_solution) == self.part_size):
                        break
                if (len(new_solution) == self.part_size):
                    break
            solution_ctr = [solution_ctr_2[k] for k in new_solution]
            solution_cvr = [solution_cvr_2[k] for k in new_solution]
        print("the best auc is")
        for i in range(len(cvr_values_2)):
            print(ctr_values_2[i],cvr_values_2[i])
        print("the params is:")
        for i in range(len(solution_cvr)):
            print(solution_ctr[i],solution_cvr[i])
        return [(solution_ctr[i],solution_cvr[i]) for i in range(len(solution_ctr))]
if __name__ == "__main__":
    sss = NSGA2(max_gen=10, part_size=20, minx=0.3, maxx=1.9, scores1=0, scores2=0, labels1=0, labels2=0).run()
