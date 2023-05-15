from sys import stdout
from typing import Union, Dict, Tuple

from bdilab_detect.base import BaseDetector, concept_drift_dict, DriftConfigMixin
from bdilab_detect.utils.warnings import deprecated_alias
import numpy as np
from scipy.stats import t

from sklearn.metrics import pairwise_distances, pairwise_kernels


def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()


def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """
        Compute the bootstrap null-distribution of MMD2u.

        这是一个Python函数，用于计算最大平均差异（MMD）统计的引导零分布。MMD是两个概率分布之间距离的度量，通常用于机器学习和统计学中比较不同的数据集或模型。
        该函数将大小为（m+n）x（m+n）的核矩阵K作为输入，其中m和n是两个不同数据集的大小，并且迭代指定执行引导的迭代次数。verbose参数控制是否在计算期间打印进度信息，random_state是随机数生成器的可选种子。
        该函数通过对具有替换的组合数据集（m+n）进行重新采样，重新计算核矩阵，然后计算重新采样的数据集之间的MMD统计，来计算MMD统计的零分布。该过程重复迭代次数，以获得MMD统计的一组引导样本。
        函数的输出是包含MMD统计的引导零分布的大小迭代的阵列。
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def compute_null_distribution_given_permutations(K, m, n, permutation,
                                                 iterations=None):
    """Compute the bootstrap null-distribution of MMD2u given
    predefined permutations.

    Note:: verbosity is removed to improve speed.
    """
    if iterations is None:
        iterations = len(permutation)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = permutation[i]
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    return mmd2u_null


def kernel_two_sample_test(X, Y, kernel_function='rbf', iterations=500,
                           verbose=False, random_state=None, **kwargs):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.

    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value


def test_independence_k2st(X, Y, alpha=0.005):
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean')) ** 2
    # 用于计算一组样本之间的核矩阵，也就是将每对样本之间的相似度通过某个核函数进行计算，得到一个对称矩阵，每个元素代表对应两个样本之间的相似度。
    _, _, p_value = kernel_two_sample_test(X, Y, kernel_function='rbf', gamma=1.0 / sigma2, verbose=False)

    return True if p_value <= alpha else False, p_value


def compute_mmd2u(X, Y):
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean')) ** 2
    K = pairwise_kernels(XY, metric='rbf', gamma=1. / sigma2)

    return MMD2u(K, m, n)


def compute_histogram(X, n_bins):
    return np.array([np.histogram(X[:, i], bins=n_bins, density=False)[0] for i in range(X.shape[1])])


def compute_hellinger_dist(P, Q):
    return np.mean(
        [np.sqrt(np.sum(np.square(np.sqrt(P[i, :] / np.sum(P[i, :])) - np.sqrt(Q[i, :] / np.sum(Q[i, :]))))) for i in
         range(P.shape[0])])


class HDDDMDrift(BaseDetector, DriftConfigMixin):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self, x_ref, gamma=1., alpha=None, use_mmd2=False, use_k2s_test=True
    ) -> None:
        super().__init__()
        if gamma is None and alpha is None:
            raise ValueError("Gamma and alpha can not be None at the same time! Please specify either gamma or alpha")

        self.drift_detected = False
        self.use_mmd2 = use_mmd2
        self.use_k2s_test = use_k2s_test

        self.gamma = gamma
        self.alpha = alpha
        self.n_bins = int(np.floor(np.sqrt(x_ref.shape[0])))

        # Initialization
        self.X_baseline = x_ref
        self.hist_baseline = compute_histogram(x_ref, self.n_bins)
        self.n_samples = x_ref.shape[0]
        self.dist_old = 0.
        self.epsilons = []
        self.t_denom = 0

        # set metadata
        self.meta['online'] = False  # offline refers to fitting the CDF for K-S
        self.meta['data_type'] = 'tabular'
        self.meta['detector_type'] = 'drift'

    def score(self, x: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def predict(self, X: np.ndarray) -> Dict[Dict[str, str], Dict[str, np.ndarray]]:

        self.t_denom += 1
        self.drift_detected = False

        # Compute histogram and the Hellinger distance to the baseline histogram
        hist = compute_histogram(X, self.n_bins)
        dist = compute_hellinger_dist(self.hist_baseline, hist)
        if self.use_mmd2:
            dist = compute_mmd2u(self.X_baseline, X)
        n_samples = X.shape[0]

        # Compute test statistic
        eps = dist - self.dist_old
        self.epsilons.append(eps)

        epsilon_hat = (1. / (self.t_denom)) * np.sum(np.abs(self.epsilons))
        sigma_hat = np.sqrt(np.sum(np.square(np.abs(self.epsilons) - epsilon_hat)) / (self.t_denom))

        beta = 0.
        if self.gamma is not None:
            beta = epsilon_hat + self.gamma * sigma_hat
        else:
            beta = epsilon_hat + t.ppf(1.0 - self.alpha / 2, self.n_samples + n_samples - 2) * sigma_hat / np.sqrt(
                self.t_denom)

        # Test for drift
        drift = np.abs(eps) > beta
        p_value = np.abs(eps)
        if self.use_k2s_test:
            drift, p_value = test_independence_k2st(self.X_baseline, X,
                                                    alpha=self.alpha)  # Testing for independence: Use the kernel two sample test!

        if drift == True:
            self.drift_detected = True

            self.t_denom = 0
            self.epsilons = []
            self.n_bins = int(np.floor(np.sqrt(n_samples)))
            self.hist_baseline = compute_histogram(X, self.n_bins)
            self.n_samples = n_samples
            self.X_baseline = X
        else:
            self.hist_baseline += hist
            self.n_samples += n_samples
            self.X_baseline = np.vstack((self.X_baseline, X))

        cd = concept_drift_dict()
        cd["data"]["is_drift"] = self.drift_detected
        cd["data"]["distance"] = np.abs(eps)
        cd["data"]["threshold"] = self.alpha
        cd["data"]["p_val"] = p_value
        cd['meta'] = self.meta
        return cd
