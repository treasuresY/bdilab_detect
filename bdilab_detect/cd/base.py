from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
from bdilab_detect.base import BaseDetector, concept_drift_dict
from bdilab_detect.cd.utils import get_input_shape, logger
import tensorflow as tf


class BaseSDDMDrift(BaseDetector):
    model: Union['tf.keras.Model', 'torch.nn.Module']

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            p_val: float = .05,
            window_size=50,
            threshold=0.99,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        A context-aware drift detector based on a conditional analogue of the maximum mean discrepancy (MMD).
        Only detects differences between samples that can not be attributed to differences between associated
        sets of contexts. p-values are computed using a conditional permutation test.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        p_val
            p-value used for the significance of the test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs probabilities or logits
        binarize_preds
            Whether to test for discrepency on soft (e.g. probs/logits) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        retrain_from_scratch
            Whether the classifier should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        seed
            Optional random seed for fold selection.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if p_val is None:
            logger.warning('No p-value set for the drift threshold. Need to set it to detect data drift.')

        # x_ref preprocessing
        self.preprocess_at_init = preprocess_at_init
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if self.preprocess_at_init and not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.alpha = p_val
        self.threshold = threshold
        self.window_size = window_size
        self.update_x_ref = update_x_ref
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)

        # define whether soft preds and optionally the stratified k-fold split
        self.preds_type = preds_type
        self.binarize_preds = binarize_preds

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta['online'] = False
        self.meta['data_type'] = data_type
        self.meta['detector_type'] = 'drift'
        self.meta['params'] = {'binarize_preds ': binarize_preds, 'preds_type': preds_type}

    def feat_selection(self, shap_values):
        feature_important = []
        feature_select = []
        for shap_index in range(shap_values.shape[1]):
            feature_shap_values = shap_values[:, shap_index]
            # 先求绝对值之后再求平均值，作为特征重要性
            feature_important.append(np.mean(abs(feature_shap_values)))
        feature_important = np.array(feature_important) / sum(feature_important)
        feature_important_zip = list(zip(range(shap_values.shape[1]), feature_important))
        feature_sorted = sorted(feature_important_zip, key=lambda x: x[1], reverse=True)
        important_sum = 0
        for shap_index, important in feature_sorted:
            important_sum += important
            feature_select.append(shap_index)
            if important_sum >= self.threshold:
                if len(feature_select) > 20:
                    feature_select = feature_select[:20]
                return feature_select

    # 数据预处理，实际上完全用的官方的数据预处理函数
    def preprocess(self, x: Union[np.ndarray, list]) -> Tuple[Union[np.ndarray, list], Union[np.ndarray, list]]:
        """
        Data preprocessing before computing the drift scores.
        Parameters
        ----------
        x
            Batch of instances.
        Returns
        -------
        Preprocessed reference data and new instances.
        """
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
            if not self.preprocess_at_init and not self.x_ref_preprocessed:
                x_ref = self.preprocess_fn(self.x_ref)
            else:
                x_ref = self.x_ref
            return x_ref, x  # type: ignore[return-value]
        else:
            return self.x_ref, x  # type: ignore[return-value]

    @abstractmethod
    def score(self, x: Union[np.ndarray, list]) \
            -> Tuple[float, float, np.ndarray, np.ndarray, Union[np.ndarray, list], Union[np.ndarray, list]]:
        pass

    def retrain(self, x):
        pass

    def predict(self, x: Union[np.ndarray, list], return_p_val: bool = True,
                return_distance: bool = True, return_probs: bool = True, return_model: bool = True) \
            -> Dict[str, Dict[str, Union[str, int, float, Callable]]]:

        # compute drift scores
        p_val = self.score(x)
        is_drift = False
        if sum(np.array(p_val) <= self.alpha) > 0:
            self.retrain(x)
            is_drift = True
        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = is_drift
        if return_p_val:
            cd['data']['p_val'] = min(p_val)
            cd['data']['threshold'] = self.alpha
        return cd
