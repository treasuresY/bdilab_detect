# -*- coding: utf-8 -*-
from bdilab_detect.cd.hdddm import HDDDMDrift


class HDDDM():
    def __init__(self, X, gamma=1., alpha=None, use_mmd2=False, use_k2s_test=False):
        self.cd = HDDDMDrift(X)
        self.drift_detected = False

    def add_batch(self, X):
        self.drift_detected = False
        cd_pred = self.cd.predict(X)
        self.drift_detected = cd_pred["data"]["is_drift"]

    def detected_change(self):
        return self.drift_detected