from bdilab_detect.cd.sddm import SDDMDrift


class SDDM():
    def __init__(self, X, y, window_size, shap_class=0, alpha=0.01):
        self.cd = SDDMDrift(X, y,shap_class=0)
        self.drift_detected = False

    def add_batch(self, X):
        self.drift_detected = False
        cd_pred = self.cd.predict(X)
        self.drift_detected = cd_pred["data"]["is_drift"]

    def detected_change(self):
        return self.drift_detected
