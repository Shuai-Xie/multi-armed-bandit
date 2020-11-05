from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n, probas=None):
        """
        :param n: 老虎机数量
        :param probas: 每台机器 获胜概率
        """
        assert probas is None or len(probas) == n
        self.n = n
        self.seed = int(time.time())
        # self.seed = 100
        if probas is None:
            np.random.seed(self.seed)  # 随机生成每台机器获胜概率
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)  # 理论上最优选择

    def generate_reward(self, i):
        """
        The player selected the i-th machine.
        选择第 i 台机器，可能获得的奖励
        """
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0
