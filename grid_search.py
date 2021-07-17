
from sklearn import metrics
# grid search搜索参数
def grid_search(p, score1, score2, label1, label2):
    """
        传入的是多个目标的上下范围及step
    """
    from numpy import arange
    best_auc = 0.5
    best_cvauc = 0.5
    best_p = []
    for i in arange(p[0][0], p[0][1], p[0][2]):
        for j in arange(p[1][0], p[1][1], p[1][2]):
            new_score = []
            for k in range(len(score1)):
                new_score.append(score1[k] ** i * score2[k] ** j)
            auc = metrics.roc_auc_score(label1, new_score)
            cvrauc = metrics.roc_auc_score(label2, new_score)
            if 0.6 * auc + 0.4 * cvrauc > 0.6 * best_auc + 0.4 * best_cvauc:
                best_cvauc = cvrauc
                best_auc = auc
                best_p = [i, j]
    print(
        "best auc is" + str(best_cvauc) + "best cvauc is" + str(best_auc) + "best param" + str(best_p[0]) + '\t' + str(
            best_p[1]))
    return best_auc, best_cvauc