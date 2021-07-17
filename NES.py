#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

from sklearn import metrics


# 适应值函数：主要考虑auc
def fitness_(in_, ctr_s, cvr_s, ctr_l, cvr_l):
    scores = []
    for i in range(len(ctr_l)):
        scores.append(ctr_s[i] ** in_[0] * cvr_s[i] ** in_[1])
    ctr_auc = metrics.roc_auc_score(ctr_l, scores)
    cvr_auc = metrics.roc_auc_score(cvr_l, scores)
    return ctr_auc*cvr_auc

class NES:
    def __init__(self, ctr_scores, cvr_scores, ctr_label, cvr_label, p_size=2, N_Point=50, N_generate=30, lr=0.02):
        self.p_size = p_size
        self.N_point = N_Point
        self.N_g = N_generate
        self.lr = lr
        self.ctr_s = ctr_scores
        self.cvr_s = cvr_scores
        self.ctr_l = ctr_label
        self.cvr_l = cvr_label

    def initialize(self):
        # build multivariate distribution
        mean = tf.Variable(tf.random_normal([2, ], 13., 1.), dtype=tf.float32)
        cov = tf.Variable(5. * tf.eye(self.p_size), dtype=tf.float32)
        mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
        make_params = mvn.sample(self.N_point)  # sampling operation

        # compute gradient and update mean and covariance matrix from sample and fitness
        tfkids_fit = tf.placeholder(tf.float32, [self.N_point, self.p_size])
        tfkids = tf.placeholder(tf.float32, [self.N_point, self.p_size])
        loss = -tf.reduce_mean(mvn.log_prob(tfkids) * tfkids_fit)  # log prob * fitness
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)  # compute and apply gradients for mean and cov

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for g in range(self.N_g):
            params = sess.run(make_params)
            kids_fit = self.get_evalute(params)
            res = sess.run([mean,cov,train_op], {tfkids_fit: kids_fit, tfkids: params})
            print(res[0])
        return res[0]

    def get_evalute(self, params):
        fitness_curr = []
        for i in range(params.shape[0]):
            fitness_curr.append(fitness_(params[i], self.ctr_s, self.cvr_s, self.ctr_l, self.cvr_l))
        return fitness_curr
