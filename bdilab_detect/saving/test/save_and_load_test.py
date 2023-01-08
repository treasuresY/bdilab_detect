import numpy as np

from bdilab_detect.cd.hdddm import HDDDMDrift
from bdilab_detect.saving.loading import load_detector
from bdilab_detect.saving.saving import save_detector


def test_save_hdddm(tmp_path):
    np.random.seed(0)
    n = 750
    n_features = 5
    update_x_ref = {'reservoir_sampling': 1000}
    preprocess_at_init = True
    x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
    cd = HDDDMDrift(x_ref,
                    gamma=1.0,
                    alpha=None,
                    use_mmd2=False,
                    use_k2s_test=False,
                    )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)

    x = x_ref.copy()
    preds_batch = cd_load.predict(x)
    print(preds_batch)


if __name__ == '__main__':
    tmp_path = 'dill_file/'
    test_save_hdddm(tmp_path)
