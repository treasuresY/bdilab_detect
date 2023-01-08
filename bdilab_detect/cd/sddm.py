import numpy as np
from typing import Callable, Dict, Optional, Union
from bdilab_detect.utils.frameworks import has_pytorch, has_tensorflow, \
    BackendValidator, Framework
from bdilab_detect.base import DriftConfigMixin

# if has_pytorch:
#     from torch.utils.data import DataLoader
#     from bdilab_detect.cd.pytorch.classifier import ClassifierDriftTorch
#     from bdilab_detect.utils.pytorch.data import TorchDataset

if has_tensorflow:
    from bdilab_detect.cd.tensorflow.sddm import SDDMDriftTF


class SDDMDrift(DriftConfigMixin):
    def __init__(
            self,
            x_ref: np.ndarray,
            y_ref: np.ndarray,
            cnn_model=None,
            dtr_model=None,
            backend: str = 'tensorflow',
            p_val: float = .01,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            window_size=50,
            threshold=0.99,
            shap_class=0,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        super().__init__()

        # Set config
        self._set_config(locals())

        backend = backend.lower()
        BackendValidator(
            backend_options={Framework.TENSORFLOW: [Framework.TENSORFLOW],
                             Framework.PYTORCH: [Framework.PYTORCH],
                             Framework.SKLEARN: [Framework.SKLEARN]},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        # 这块他要把参数分成位置实参和关键字实参，我实在是不知道这有啥用，所以我就没干
        kwargs = locals()
        args = []
        if backend == Framework.TENSORFLOW:
            pop_kwargs = ['device', 'dataloader', 'use_calibration', 'calibration_kwargs', 'use_oob', 'self']
            [kwargs.pop(k, None) for k in pop_kwargs]
            self._detector = SDDMDriftTF(*args, **kwargs)
        self.meta = self._detector.meta
        self.meta['name'] = self.__class__.__name__

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True,
                return_distance: bool = True, return_probs: bool = True, return_model: bool = True) \
            -> Dict[str, Dict[str, Union[str, int, float, Callable]]]:
        """
        Predict whether a batch of data has drifted from the reference data.

        Parameters
        ----------
        x
            Batch of instances.
        return_p_val
            Whether to return the p-value of the test.
        return_distance
            Whether to return a notion of strength of the drift.
            K-S test stat if binarize_preds=False, otherwise relative error reduction.
        return_probs
            Whether to return the instance level classifier probabilities for the reference and test data
            (0=reference data, 1=test data).
        return_model
            Whether to return the updated model trained to discriminate reference and test instances.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries

         - 'meta' - has the model's metadata.

         - 'data' - contains the drift prediction and optionally the p-value, performance of the classifier \
        relative to its expectation under the no-change null, the out-of-fold classifier model \
        prediction probabilities on the reference and test data, and the trained model. \
        """
        return self._detector.predict(x, return_p_val, return_distance, return_probs, return_model)
