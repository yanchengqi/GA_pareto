# encoding: utf-8
import numpy as np
from sklearn import metrics


# 适应值函数：主要考虑auc
def fitness_(in_, ctr_s, cvr_s, ctr_l, cvr_l):
    scores = []
    for i in range(len(ctr_l)):
        scores.append(ctr_s ** in_[0] * cvr_s ** in_[1])
    ctr_auc = metrics.roc_auc_score(ctr_l, scores)
    cvr_auc = metrics.roc_auc_score(cvr_l, scores)
    return [ctr_auc, cvr_auc]
