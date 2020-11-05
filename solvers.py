import time
import numpy as np
from scipy.stats import beta

from bandits import BernoulliBandit
from tqdm import tqdm


class Solver(object):
    def __init__(self, bandit):
        """
        :param bandit: the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n  # n: 老虎机总个数；记录每台机器的选择数量
        self.actions = []  # 选择的 action 序列, 0,...n-1
        self.regret = 0.  # Cumulative regret. 累积 regret
        self.regrets = [0.]  # History of cumulative regret. 累积 regret 序列

    def update_regret(self, i):
        """
        选择机器 i 带来的 regret，每次选择后，更新整体的 regret，并加入 regrets 序列
        """
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None

        for _ in tqdm(range(num_steps)):
            i = self.run_one_step()  # 执行1次，返回1个 action

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)  # 更新 regret


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_proba=1.0):
        """
        :param bandit:
        :param eps: the probability to explore at each time step. 探索概率
        :param init_proba: default to be 1.0; optimistic initialization 乐观估计 n 台 machine 的获胜概率
        """
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        self.eps = eps

        # 基于当前 t 次选择，对各类 machine reward 的观察
        self.estimates = [init_proba] * self.bandit.n  # Optimistic initialization

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        # 选择 action i
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.n)  # 随机选择1个machine
        else:
            # Pick the best one.
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        # 生成 reward r
        r = self.bandit.generate_reward(i)

        # 更新 Q^_i
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        # 原式推理: e_i = ((e_i * cnt_i) + r) / (cnt_i +1 ) = e_i + 1/(cnt_i+1) * (r-e_i)

        return i


class UCB1(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # 选择 action i，估计上界最大的 Q^_i + U^_i, U^_i = sqrt( 2*log(t) / cnt_i )
        # 对于上界项，当前已选 i 的次数 cnt_i 越大，上界越小
        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[x])))

        # 生成 reward
        r = self.bandit.generate_reward(i)

        # 更新 Q^_i
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class BayesianUCB(Solver):
    """Assuming Beta prior. 假设每个老虎机胜率概率分布遵从 beta 分布"""

    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float): how many standard dev to consider as upper confidence bound. 3倍标准差
        init_a (int): initial value of a in Beta(a, b). beta 分布的 a
        init_b (int): initial value of b in Beta(a, b). beta 分布的 b
        """
        super(BayesianUCB, self).__init__(bandit)
        self.c = c

        # 初始化，情况未知，每台老虎机都是 beta(1,1) 均匀分布；即每台胜率可取 [0,1] 任意概率
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        # 对每台老虎机 计算 beta 分布均值，表示对每台机器胜率的估计
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit.n),
            # a/a+b 使用 beta 分布的均值，来表示当前估计的胜率 + 3倍 beta 分布标准差 = Bayesian UCB
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(self._as[x], self._bs[x]) * self.c
        )

        # reward
        r = self.bandit.generate_reward(i)

        # Update Gaussian posterior, 伯努利分布得到的 r 对应 beta 两个边界 a/b 的更新
        # α and β correspond to the counts when we succeeded or failed
        self._as[i] += r  # update a when r = 1
        self._bs[i] += (1 - r)  # update b when r = 0

        return i


class ThompsonSampling(Solver):
    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit)

        # 初始化 beta 分布
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        # beta 分布 mean 估算概率
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        # np.random.beta, 从 beta 分布中 随机采样
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]

        # 根据采样值大小，选择 action
        i = max(range(self.bandit.n), key=lambda x: samples[x])

        # reward
        r = self.bandit.generate_reward(i)

        # update beta dist
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
