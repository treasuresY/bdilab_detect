import numpy as np

from bdilab_detect.cd import SDDMDrift
from bdilab_detect.saving.loading import load_detector
from bdilab_detect.saving.saving import save_detector
from bdilab_detect.test.experiments_utils import create_rbf_drift_dataset


def preprocess_simple(x: np.ndarray):
    """
    Simple function to test serialization of generic Python function within preprocess_fn.
    """
    return x * 2.0


def test_save_sddm(tmp_path):
    data = create_rbf_drift_dataset(n_samples_per_concept=500, n_concept_drifts=6)['data']
    X, y = data
    x_ref = X[0:int(0.2 * X.shape[0]), :]
    y_ref = y[0:int(0.2 * y.shape[0]), :]
    cd = SDDMDrift(x_ref,
                   y_ref,
                   p_val=.01,
                   preprocess_fn=preprocess_simple,
                   window_size=50,
                   threshold=0.99,
                   shap_class=0,
                   x_ref_preprocessed=False,
                   preprocess_at_init=True,
                   backend='tensorflow',
                   preds_type='probs',
                   binarize_preds=False
                   )
    save_detector(cd, tmp_path)
    cd_load = load_detector(tmp_path)
    step = 50

    for i in range(int(0.2 * X.shape[0]), X.shape[0], step):
        stop = i+step
        if stop >= X.shape[0]:
            break
        x = X[i:stop, :]
        preds_batch = cd_load.predict(x)
        print(preds_batch)


if __name__ == '__main__':
    tmp_path = 'sddm/'
    test_save_sddm(tmp_path)
