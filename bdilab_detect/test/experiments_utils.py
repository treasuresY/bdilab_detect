# -*- coding: utf-8 -*-
import random
import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from skmultiflow.data import STAGGERGenerator, RandomRBFGenerator, SEAGenerator
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# 有些人工数据生成器不剧本生成漂移的能力，或者生成的漂移不可能被无监督检测器检测到
def make_drift(dataset_gen, batch_size, true_drift, dataset_num):
    if true_drift:
        pass
    else:
        X, y = dataset_gen.next_sample(batch_size)
        # 随机选取一个特征
        feature_num = random.randint(0, X.shape[1] - 1)

        # 按照某种规则挑选出一堆数据
        random_num = [0.1, 0.2, 0.3]
        # 漂移数据的混合比率
        drift_rate = int(batch_size * (1 - random_num[random.randint(0, len(random_num) - 1)]))
        # 特殊数据的取样比率
        border_rate = random_num[random.randint(0, len(random_num) - 1)]

        border = np.min(X[:, feature_num]) + border_rate * (np.max(X[:, feature_num]) - np.min(X[:, feature_num]))
        some_X = X[X[:, feature_num] < border]
        some_y = y[X[:, feature_num] < border]

        # 少补
        while some_X.shape[0] <= drift_rate:
            new_X, new_y = dataset_gen.next_sample(batch_size)
            some_X = np.concatenate([some_X, new_X[new_X[:, feature_num] < border]], axis=0)
            some_y = np.concatenate([some_y, new_y[new_X[:, feature_num] < border]], axis=0)
        # 多退
        some_X = some_X[:drift_rate, :]
        some_y = some_y[:drift_rate]
        # 重新生成数据
        X, y = dataset_gen.next_sample(batch_size - some_X.shape[0])
        drift_X = np.concatenate([X, some_X], axis=0)
        drift_y = np.concatenate([y, some_y], axis=0)
        # 打乱顺序
        drift_X, drift_y = shuffle(drift_X, drift_y, )
        return drift_X, drift_y


if __name__ == '__main__':
    make_drift(SEAGenerator(), 200, False, "SEA")


# Blinking X
def create_blinking_X_dataset(n_samples_per_concept=200, n_concepts=4):
    def labeling_a(n_samples_per_class):
        X0 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)),
                            axis=0).reshape(-1, 1)
        X0 = np.concatenate([X0, -1. * X0], axis=1)
        Y0 = np.array([0 for _ in range(n_samples_per_class)])

        X1 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)),
                            axis=0).reshape(-1, 1)
        X1 = np.concatenate([X1, 1. * X1], axis=1)
        Y1 = np.array([1 for _ in range(n_samples_per_class)])

        return np.concatenate([X0, X1], axis=0), np.concatenate([Y0, Y1], axis=0)

    def labeling_b(n_samples_per_class):
        X0 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)),
                            axis=0).reshape(-1, 1)
        X0 = np.concatenate([X0, -1. * X0], axis=1)
        Y0 = np.array([1 for _ in range(n_samples_per_class)])

        X1 = np.concatenate((np.arange(0.4, 20.4, step=20. / 50.), np.arange(-20., 0., step=20. / 50.)),
                            axis=0).reshape(-1, 1)
        X1 = np.concatenate([X1, 1. * X1], axis=1)
        Y1 = np.array([0 for _ in range(n_samples_per_class)])

        return np.concatenate([X0, X1], axis=0), np.concatenate([Y0, Y1], axis=0)

    # Start with labeling a, then switch to labeling b, then again to labeling a, ....
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    label_a = True
    for _ in range(n_concepts):
        data_stream_X, data_stream_Y = labeling_a(int(n_samples_per_concept / 2)) if label_a else labeling_b(
            int(n_samples_per_concept / 2))
        data_stream_X, data_stream_Y = shuffle(data_stream_X, data_stream_Y)
        label_a = not label_a
        t += n_samples_per_concept

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# Mixed RBF
def create_mixed_rbf_dataset(n_wrong_samples_per_class=120, n_correct_samples_per_class=240, n_concepts=4):
    def mixed_rbf_blob_data(n_wrong_samples, n_correct_samples):
        X = []
        Y = []

        centerA = np.array([[-2., 2.]])
        centerB = np.array([[2., -2.]])

        # Class A
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerA, cluster_std=0.5)
        X.append(x)
        Y.append([1 for _ in range(n_wrong_samples)])

        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerA, cluster_std=0.8)
        X.append(x)
        Y.append([0 for _ in range(n_correct_samples)])

        # Class B
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerB, cluster_std=0.5)
        X.append(x)
        Y.append([0 for _ in range(n_wrong_samples)])

        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerB, cluster_std=0.8)
        X.append(x)
        Y.append([1 for _ in range(n_correct_samples)])

        return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

    def unmixed_rbf_blob_data(n_wrong_samples, n_correct_samples):
        X = []
        Y = []

        centerA = np.array([[-2., 2.]])
        centerA2 = np.array([[2., 5.0]])
        centerB = np.array([[2., -2.]])
        centerB2 = np.array([[5.5, 2.]])

        # Class A
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerA2, cluster_std=0.5)
        X.append(x)
        Y.append([1 for _ in range(n_wrong_samples)])

        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerA, cluster_std=0.8)
        X.append(x)
        Y.append([0 for _ in range(n_correct_samples)])

        # Class B
        x, _ = make_blobs(n_samples=n_wrong_samples, n_features=2, centers=centerB, cluster_std=0.5)
        X.append(x)
        Y.append([0 for _ in range(n_wrong_samples)])

        x, _ = make_blobs(n_samples=n_correct_samples, n_features=2, centers=centerB2, cluster_std=0.8)
        X.append(x)
        Y.append([1 for _ in range(n_correct_samples)])

        return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

    # Start with a mixed sampels, unmix it, mix it again, ...
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    mixed = True
    for _ in range(n_concepts):
        data_stream_X, data_stream_Y = mixed_rbf_blob_data(n_wrong_samples_per_class,
                                                           n_correct_samples_per_class) if mixed else unmixed_rbf_blob_data(
            n_wrong_samples_per_class, n_correct_samples_per_class)
        data_stream_X, data_stream_Y = shuffle(data_stream_X, data_stream_Y)
        mixed = not mixed
        t += 2 * n_wrong_samples_per_class + n_correct_samples_per_class

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# Rotating hyperplane dataset
def create_rotating_hyperplane_dataset(n_samples_per_concept=200, concepts=np.arange(0.0, 5.0, 1.0)):
    def create_hyperplane_dataset(n_samples, n_dim=2, plane_angle=0.45):
        w = np.dot(np.array([[np.cos(plane_angle), -np.sin(plane_angle)], [np.sin(plane_angle), np.cos(plane_angle)]]),
                   np.array([1.0, 1.0]))
        X = np.random.uniform(-1.0, 1.0, (n_samples, n_dim))
        Y = np.array([1 if np.dot(x, w) >= 0 else 0 for x in X])

        return X, Y

    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for a in concepts:
        data_stream_X, data_stream_Y = create_hyperplane_dataset(n_samples=n_samples_per_concept, plane_angle=a)
        data_stream_X, data_stream_Y = shuffle(data_stream_X, data_stream_Y)
        t += n_samples_per_concept

        X_stream.append(data_stream_X)
        Y_stream.append(data_stream_Y)
        concept_drifts.append(t)
    concept_drifts.pop()

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# SEA
def create_sea_drift_dataset(n_samples_per_concept=500, concepts=[0, 1, 2, 3]):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    gen = SEAGenerator()
    gen.prepare_for_use()
    for _ in concepts:
        if t != 0:
            concept_drifts.append(t)

        X, y = make_drift(gen, n_samples_per_concept, False, "")
        X_stream.append(X)
        Y_stream.append(y)

        gen.generate_drift()

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array([])}


# STAGGER
def create_stagger_drift_dataset(n_samples_per_concept=500, n_concept_drifts=3):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    gen = STAGGERGenerator()
    gen.prepare_for_use()
    for _ in range(n_concept_drifts):
        if t != 0:
            concept_drifts.append(t)

        X, y = make_drift(gen, n_samples_per_concept, False, "")
        X_stream.append(X)
        Y_stream.append(y)

        gen.generate_drift()
        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array([])}


# Random rbf
def create_rbf_drift_dataset(n_samples_per_concept=500, n_concept_drifts=3):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for _ in range(n_concept_drifts):
        if t != 0:
            concept_drifts.append(t)
        # 每次漂移点创建一个数据生成器，使每次生成的数据不同
        gen = RandomRBFGenerator(n_features=4, n_centroids=10)
        gen.prepare_for_use()
        X, y = gen.next_sample(batch_size=n_samples_per_concept)
        X_stream.append(X)
        Y_stream.append(y)

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# Gaussians with color mixing
def create_mixing_gaussians_dataset(n_samples_per_concept=500, concepts=[(0, 1.), (1, 1.), (0, .5), (1, 1.), (0, 1.)]):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for concept_type, p in concepts:
        if t != 0:
            concept_drifts.append(t)

        X = np.concatenate((
            np.random.multivariate_normal((-3, -3), ((1, 0), (0, 1)), int(n_samples_per_concept * 75 / 200)),
            np.random.multivariate_normal((-3, 3), ((1, 0), (0, 1)), int(n_samples_per_concept * 25 / 200)),
            np.random.multivariate_normal((3, -3), ((1, 0), (0, 1)), int(n_samples_per_concept * 25 / 200)),
            np.random.multivariate_normal((3, 3), ((1, 0), (0, 1)), int(n_samples_per_concept * 75 / 200))))
        if concept_type == 0:
            y = np.concatenate((
                np.random.choice([-1, 1], size=int(n_samples_per_concept * 100 / 200), p=[p, 1 - p]),
                np.random.choice([-1, 1], size=int(n_samples_per_concept * 100 / 200), p=[1 - p, p])))
        else:
            y = np.concatenate((
                np.random.choice([-1, 1], size=int(n_samples_per_concept * 75 / 200), p=[p, 1 - p]),
                np.random.choice([-1, 1], size=int(n_samples_per_concept * 25 / 200), p=[1 - p, p]),
                np.random.choice([-1, 1], size=int(n_samples_per_concept * 25 / 200), p=[p, 1 - p]),
                np.random.choice([-1, 1], size=int(n_samples_per_concept * 75 / 200), p=[1 - p, p])))

        perm = np.random.permutation(X.shape[0])
        X_stream.append(X[perm, :])
        Y_stream.append(y[perm])

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# Two mixing Gaussians mixtures
def create_two_mixing_gaussians_dataset(n_samples_per_concept=500,
                                        concepts=[(1., .5), (.5, 1.), (.5, .5), (.5, 1.), (1., .5)]):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for p, q in concepts:
        if t != 0:
            concept_drifts.append(t)

        X = np.concatenate((
            np.random.multivariate_normal((-3, -3), ((1, 0), (0, 1)), int(n_samples_per_concept / 4)),
            np.random.multivariate_normal((-3, 3), ((1, 0), (0, 1)), int(n_samples_per_concept / 4)),
            np.random.multivariate_normal((3, -3), ((1, 0), (0, 1)), int(n_samples_per_concept / 4)),
            np.random.multivariate_normal((3, 3), ((1, 0), (0, 1)), int(n_samples_per_concept / 4))))
        y = np.concatenate((
            np.random.choice([-1, 1], size=int(n_samples_per_concept / 4), p=[p, 1 - p]),
            np.random.choice([-1, 1], size=int(n_samples_per_concept / 4), p=[1 - p, p]),
            np.random.choice([-1, 1], size=int(n_samples_per_concept / 4), p=[q, 1 - q]),
            np.random.choice([-1, 1], size=int(n_samples_per_concept / 4), p=[1 - q, q])))

        perm = np.random.permutation(X.shape[0])
        X_stream.append(X[perm, :]);
        Y_stream.append(y[perm])

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# mean change
def create_mean_change_dataset(epsilon, n_samples_per_concept=500, n_drifts=10):
    interval1 = [epsilon / 2, epsilon + 1e-100]
    interval2 = [-epsilon, -epsilon / 2 + 1e-100]
    standard_deviation1 = 0.2
    standard_deviation2 = 0.2
    correlation = 0.5
    cov = standard_deviation1 * standard_deviation2 * correlation
    cov_matrix = [[standard_deviation1 * standard_deviation1, cov], [cov, standard_deviation2 * standard_deviation2]]
    concept_drifts = []
    X_stream = []
    Y_stream = []
    for i in range(n_drifts):
        if i != 0:
            concept_drifts.append(i * n_samples_per_concept)

        u1 = np.random.uniform(interval1[0], interval1[1], 2)
        u2 = np.random.uniform(interval2[0], interval2[1], 2)
        mean1 = np.random.choice([u1[0], u2[0]])
        mean2 = np.random.choice([u1[1], u2[1]])
        data = np.random.multivariate_normal([mean1, mean2], cov_matrix, n_samples_per_concept)
        data_y = []
        for index in range(data.shape[0]):
            if data[index][0] > data[index][1]:
                data_y.append(1)
            else:
                data_y.append(0)
        data_y = np.array(data_y)
        X_stream.append(data)
        Y_stream.append(data_y)
    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# standard deviation change
def create_std_change_dataset(epsilon, n_samples_per_concept=500, n_drifts=10):
    mean1 = mean2 = 0.5
    correlation = 0.5
    interval1 = [epsilon / 2, epsilon + 1e-100]
    interval2 = [-epsilon, -epsilon / 2 + 1e-100]
    concept_drifts = []
    X_stream = []
    Y_stream = []
    for i in range(n_drifts):
        if i != 0:
            concept_drifts.append(i * n_samples_per_concept)
        standard_d1 = np.random.uniform(interval1[0], interval1[1], 2)
        standard_d2 = np.random.uniform(interval2[0], interval2[1], 2)
        std1 = np.random.choice([standard_d1[0], standard_d2[0]])
        std2 = np.random.choice([standard_d1[1], standard_d2[1]])
        cov = std1 * std2 * correlation
        cov_matrix = [[std1 * std1, cov], [cov, std2 * std2]]
        data = np.random.multivariate_normal([mean1, mean2], cov_matrix, n_samples_per_concept)
        data_y = []
        for index in range(data.shape[0]):
            if data[index][0] > data[index][1]:
                data_y.append(1)
            else:
                data_y.append(0)
        data_y = np.array(data_y)
        X_stream.append(data)
        Y_stream.append(data_y)
    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


#  correlation coefficient change
def create_rou_change_dataset(epsilon, n_samples_per_concept=500, n_drifts=10):
    mean1 = mean2 = 0.5
    std1 = std2 = 0.2
    interval1 = [epsilon / 2, epsilon + 1e-100]
    interval2 = [-epsilon, -epsilon / 2 + 1e-100]
    concept_drifts = []
    X_stream = []
    Y_stream = []
    for i in range(n_drifts):
        if i != 0:
            concept_drifts.append(i * n_samples_per_concept)

        rou1 = np.random.uniform(interval1[0], interval1[1], 1)
        rou2 = np.random.uniform(interval2[0], interval2[1], 1)
        rou = np.random.choice([rou1[0], rou2[0]])
        cov = std1 * std2 * rou
        cov_matrix = [[std1 * std1, cov], [cov, std2 * std2]]
        data = np.random.multivariate_normal([mean1, mean2], cov_matrix, n_samples_per_concept)
        data_y = []
        for index in range(data.shape[0]):
            if data[index][0] > data[index][1]:
                data_y.append(1)
            else:
                data_y.append(0)
        data_y = np.array(data_y)
        X_stream.append(data)
        Y_stream.append(data_y)
    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


# benchmark
from bdilab_detect.test import load_benchmark_data


def create_1CDT_dataset(n_max_length=16000):
    X, y = load_benchmark_data.read_benchmark_data("1CDT.csv")
    drift_size = 400

    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def create_1CHT_dataset(n_max_length=16000):
    X, y = load_benchmark_data.read_benchmark_data("1CHT.csv")
    drift_size = 400
    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def create_2CDT_dataset(n_max_length=16000):
    X, y = load_benchmark_data.read_benchmark_data("2CDT.csv")
    drift_size = 400
    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def create_2CHT_dataset(n_max_length=16000):
    X, y = load_benchmark_data.read_benchmark_data("2CHT.csv")
    drift_size = 400
    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def pre_benchmark2(dataset, n_max_length, drift_size, step):
    concept_drifts = [drift for drift in range(drift_size, n_max_length, drift_size)]

    data = dataset["data"]
    X = data[0]
    y = data[1]
    for i in range(len(concept_drifts)):
        X[i * drift_size:(i + 1) * drift_size] = X[i * step * drift_size:i * step * drift_size + drift_size]
        y[i * drift_size:(i + 1) * drift_size] = y[i * step * drift_size:i * step * drift_size + drift_size]
    X = X[0:n_max_length]
    y = y[0:n_max_length]
    data = (X, y)
    dataset["data"] = data
    dataset["drifts"] = np.array(concept_drifts)
    return dataset


def create_4CR_dataset(n_max_length=144400):
    X, y = load_benchmark_data.read_benchmark_data("4CR.csv")
    drift_size = 400
    step = 25
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_MG_2C_2D_dataset(n_max_length=200000):
    X, y = load_benchmark_data.read_benchmark_data("MG_2C_2D.csv")
    drift_size = 2000
    step = 20
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_UG_2C_2D_dataset(n_max_length=100000):
    X, y = load_benchmark_data.read_benchmark_data("UG_2C_2D.csv")
    drift_size = 1000
    step = 10
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_4CRE_V1_dataset(n_max_length=4000):
    X, y = load_benchmark_data.read_benchmark_data("4CRE-V1.csv")
    drift_size = 1000
    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def create_5CVT_dataset(n_max_length=40000):
    X, y = load_benchmark_data.read_benchmark_data("5CVT.csv")
    drift_size = 1000
    step = 2
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_4CE1CF_dataset(n_max_length=173250):
    X, y = load_benchmark_data.read_benchmark_data("4CE1CF.csv")
    drift_size = 750
    step = 35
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_benchmark_dataset(X, y, n_max_length, drift_size):
    concept_drifts = [drift for drift in range(drift_size, n_max_length, drift_size)]
    if X.shape[0] > n_max_length:
        X = X[0:n_max_length, :]
        y = y[0:n_max_length]
    data = (X, y.reshape(-1, 1))
    dataset = {"data": data, "drifts": np.array(concept_drifts)}

    return {"data": data, "drifts": np.array(concept_drifts)}


# Real world data
from bdilab_detect.test import load_rw_data


def create_weather_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_weather()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


def create_forest_cover_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_forest_cover_type()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


def create_electricity_market_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_electricity_market()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


def create_phishing_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_phishing()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


# 2011训练，在2012上检测
def create_bike_share_drift_dataset(n_concept_drifts=5):
    X = load_rw_data.read_data_bike_share()
    train_size = 0.5 * X.shape[0]
    flag = train_size
    concept_drifts = []
    for i in range(1, n_concept_drifts):
        if i in [1, 3, 5, 7, 8, 10, 12]:
            flag += 31 * 24
        elif i == 2:
            flag += 28 * 24
        else:
            flag += 30 * 24
        concept_drifts.append(flag)
    Y_stream = X['count'].values
    X_stream = X.drop(columns=['count']).values

    # data2012 = X[X['year'] == 2012]
    # data2011 = X[X['year'] == 2011]
    # data2012 = data2012.reindex(np.random.permutation(data2012.index))
    # train_data = data2012[0:train_size]
    # X_stream = pd.contact([train_data, data2011])
    # X_stream = X_stream.reset_index(drop=True)
    # y = X_stream['count'].valuse()

    return {"data": (X_stream, Y_stream.reshape(-1, 1)), "drifts": []}


def create_controlled_drift_dataset(X, y=None, n_max_length=1000, n_concept_drifts=3):
    return {"data": (X, y.reshape(-1, 1)), "drifts": []}


# Drift detection
class DriftDetectorSupervised():
    def __init__(self, clf, drift_detector, training_buffer_size):
        self.clf = clf
        self.drift_detector = drift_detector
        self.training_buffer_size = training_buffer_size
        self.X_training_buffer = []
        self.Y_training_buffer = []
        self.changes_detected = []

    def apply_to_stream(self, X_stream, Y_stream):
        self.changes_detected = []

        collect_samples = False
        T = len(X_stream)
        since = time.time()
        for t in range(T):
            x, y = X_stream[t, :], Y_stream[t, :]

            if collect_samples == False:
                self.drift_detector.add_element(self.clf.score(x, y))

                if self.drift_detector.detected_change():
                    self.changes_detected.append(t)

                    collect_samples = True
                    self.X_training_buffer = []
                    self.Y_training_buffer = []
            else:
                self.X_training_buffer.append(x)
                self.Y_training_buffer.append(y)

                if len(self.X_training_buffer) >= self.training_buffer_size:
                    collect_samples = False
                    self.clf.fit(np.array(self.X_training_buffer), np.array(self.Y_training_buffer))
        time_elapsed = time.time() - since
        return self.changes_detected, time_elapsed


class DriftDetectorUnsupervised():
    def __init__(self, drift_detector, batch_size):
        self.drift_detector = drift_detector
        self.batch_size = batch_size
        self.changes_detected = []
        self.detector_name = self.drift_detector.__class__.__name__

    def apply_to_stream(self, data_stream):
        since = time.time()
        self.changes_detected = []

        n_data_stream_samples = len(data_stream)

        t = 0
        while t < n_data_stream_samples:
            end_idx = t + self.batch_size
            if end_idx >= n_data_stream_samples:
                end_idx = n_data_stream_samples

            batch = data_stream[t:end_idx, :]
            self.drift_detector.add_batch(batch)

            if self.drift_detector.detected_change():
                self.changes_detected.append(t)
                print(self.detector_name + "检测到漂移啦")

            t += self.batch_size
        print(self.detector_name + "在" + str(self.changes_detected) + "处检测到漂移啦")
        time_elapsed = time.time() - since
        return self.changes_detected, time_elapsed


# Evaluation
def evaluate(true_concept_drifts, pred_concept_drifts, time_elapsed, tol=200):
    false_alarms = 0
    drift_detected = 0
    drift_not_detected = 0
    delays = []

    # Check for false alarms
    for t in pred_concept_drifts:
        b = False
        for dt in true_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                break
        if b is False:  # False alarm
            false_alarms += 1

    # Check for detected and undetected drifts
    for dt in true_concept_drifts:
        b = False
        for t in pred_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                drift_detected += 1
                delays.append(t - dt)
                break
        if b is False:
            drift_not_detected += 1

    return {"false_alarms": false_alarms, "drift_detected": drift_detected, "drift_not_detected": drift_not_detected,
            "delays": delays, "time_elapsed": time_elapsed}


# Evaluation
def evaluate_rw(data_desc, method, D, pred_concept_drifts, time_elapsed, batch_size, train_size=0.2,
                train_buffer_size=100):
    # 构造分类器
    model = AdaBoostClassifier()

    # 数据准备
    n_train = (int)(0.2 * D["data"][0].shape[0])
    X, Y = D["data"]
    X0, Y0 = X[0:n_train, :], Y[0:n_train, :]  # Training dataset
    model.fit(X0, Y0.ravel())

    collect_samples = False
    rw_scores = []
    data_indexs = []
    n_x_samples = len(X)
    X_training_buffer = []
    Y_training_buffer = []

    t = 0
    while t < n_x_samples:
        end_idx = t + batch_size
        if end_idx >= n_x_samples:
            end_idx = n_x_samples
        x_batch = X[t:end_idx, :]
        y_batch = Y[t:end_idx, :]
        if not collect_samples:

            y_pred = model.predict(x_batch)
            rw_scores.append(precision_score(y_batch, y_pred, average='weighted'))
            data_indexs.append(t)
            for pred_concept_drift in pred_concept_drifts:
                if t <= pred_concept_drift < end_idx:
                    collect_samples = True
                    X_training_buffer = []
                    Y_training_buffer = []
        else:
            y_pred = model.predict(x_batch)
            rw_scores.append(precision_score(y_batch, y_pred, average='weighted'))
            data_indexs.append(t)
            X_training_buffer.append(x_batch)
            Y_training_buffer.append(y_batch)
            if len(X_training_buffer) * len(x_batch) > train_buffer_size:
                X_training = np.concatenate(X_training_buffer, axis=0)
                Y_training = np.concatenate(Y_training_buffer, axis=0)
                model.fit(X_training, Y_training)
                collect_samples = False
        t += batch_size

    # 画出结果
    plt.xlabel('index')
    plt.ylabel('accuracy/p-value')
    plt.title(data_desc + "___" + method)
    plt.vlines(x=pred_concept_drifts, ymin=0.0, ymax=1, colors='r', linestyles='-',
               label='drift')
    plt.plot(data_indexs, rw_scores, lw=2, label='accuracy')
    plt.title(data_desc + "___" + method + "__" + str(np.average(rw_scores)))
    plt.show()
    print(method)
    print(pred_concept_drifts)
    print(data_indexs)
    print(rw_scores)
    return {"data_indexs": data_indexs, "scores": rw_scores}


# Classifier
from sklearn.svm import SVC


class Classifier():
    def __init__(self, model=SVC(C=1.0, kernel='linear')):
        self.model = model
        self.flip_score = False

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def score(self, x, y):
        s = int(self.model.predict([x]) == y)
        if self.flip_score == True:
            return 1 - s
        else:
            return s

    def score_set(self, X, y):
        return self.model.score(X, y.ravel())
