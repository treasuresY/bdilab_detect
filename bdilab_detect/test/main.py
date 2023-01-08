#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/15 12:02
# @Author  : ZWP
# @Desc    : 
# @File    : main.py
import numpy as np

from bdilab_detect.test.hdddm_test import HDDDM
from bdilab_detect.test.sddm_test import SDDM
from bdilab_detect.test.experiments_utils import create_rbf_drift_dataset, \
    DriftDetectorUnsupervised, evaluate


def test_dataset(all_dataset, methods):
    global run_record
    run_record += 1
    print("第" + str(run_record) + "次迭代")
    # 各个数据集的运行结果
    results = dict()
    for dataset in all_dataset:
        results[dataset] = dict()

    # 数据集数据，使用的实验工具包的数据集生成函数，内容一概是（名称，（x，y））的元组
    dataset_build = [

        ("RandomRBF", create_rbf_drift_dataset(n_samples_per_concept=500, n_concept_drifts=4)),
        # ("UG_2C_2D", create_UG_2C_2D_dataset(n_max_length=5000)),
        # ("MG_2C_2D", create_MG_2C_2D_dataset(n_max_length=5000)),
        # ("2CDT", create_2CDT_dataset(n_max_length=2000)),
        # ("2CHT", create_2CHT_dataset(n_max_length=2000)),
        # ("4CR", create_4CR_dataset(n_max_length=2000)),
        # ("4CRE-V1", create_4CRE_V1_dataset(5000)),
        # ("5CVT", create_5CVT_dataset(n_max_length=5000)),
        # ("4CE1CF", create_4CE1CF_dataset()),
    ]
    datasets = []
    for dataset in dataset_build:
        if dataset[0] in all_dataset:
            datasets.append(dataset)
    # Test all data sets
    # r_all_datasets = Parallel(n_jobs=4)(delayed(test_on_data_set)(data_desc, D, methods) for data_desc, D in datasets)
    r_all_datasets = [test_on_data_set(data_desc, D, methods) for data_desc, D in datasets]
    for r_data in r_all_datasets:
        for k in r_data.keys():
            results[k] = r_data[k]

    return results


# 对所有方法进行测试的地方
def test_on_data_set(data_desc, D, methods):
    r = {data_desc: dict()}
    for method in methods:
        r.get(data_desc)[method] = []

    training_buffer_size = 100  # Size of training buffer of the drift detector
    n_train = (int)(0.2 * D["data"][0].shape[0])  # Initial training set size

    concept_drifts = D["drifts"]
    X, Y = D["data"]
    data_stream = np.concatenate((X, Y.reshape(-1, 1)), axis=1)

    X0, Y0 = X[0:n_train, :], Y[0:n_train, :]  # Training dataset
    data0 = data_stream[0:n_train, :]

    # 为什么漂移数据点没有减去训练数据？？？？
    # 太傻逼了，设计漂移点位置时候有训练数据，检测点计数的时候他妈的没了，真绝了
    X_next, Y_next = X[n_train:, :], Y[n_train:, :]  # Test set
    data_next = data_stream[n_train:, :]
    # Run unsupervised drift detector

    if "HDDDM" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.005), batch_size=batch_size)
        changes_detected, time_elapsed = dd.apply_to_stream(data_stream)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["HDDDM"].append(scores)

    if "SDDM" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(SDDM(X0, Y0, 50, 0, alpha_ks), batch_size=batch_size)
        changes_detected, time_elapsed = dd.apply_to_stream(X)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["SDDM"].append(scores)

    return r


if __name__ == '__main__':
    tol = 90
    patience = 0
    alpha_tran = 0.02
    alpha_middle = 0.02
    alpha_ks = 0.008
    n_itr = 3
    batch_size = 50
    run_record = 0
    all_datasets = [

        "RandomRBF",

    ]
    methods = [
        "SDDM",

    ]

    # all_results = Parallel(n_jobs=-1)(delayed(test_dataset)(all_datasets, methods) for _ in range(n_itr))
    all_results = [test_dataset(all_datasets, methods) for _ in range(n_itr)]
