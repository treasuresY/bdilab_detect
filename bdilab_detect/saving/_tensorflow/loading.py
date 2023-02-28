import logging
import os
from importlib import import_module
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union, Type

import dill
import joblib
import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.tree import DecisionTreeRegressor

from bdilab_detect.base import Detector
from bdilab_detect.cd import SDDMDrift
from bdilab_detect.models.tensorflow import TransformerEmbedding
from bdilab_detect.utils.frameworks import Framework
from bdilab_detect.saving._typing import VALID_DETECTORS
from bdilab_detect.version import __version__
from bdilab_detect.cd.hdddm import HDDDMDrift
from bdilab_detect.cd.tensorflow import HiddenOutput

logger = logging.getLogger(__name__)


def load_model(filepath: Union[str, os.PathLike],
               load_dir: str = 'model',
               custom_objects: dict = None,
               layer: Optional[int] = None,
               ) -> tf.keras.Model:
    """
    Load TensorFlow model.

    Parameters
    ----------
    filepath
        Saved model directory.
    load_dir
        Name of saved model folder within the filepath directory.
    custom_objects
        Optional custom objects when loading the TensorFlow model.
    layer
        Optional index of a hidden layer to extract. If not `None`, a
        :py:class:`~alibi_detect.cd.tensorflow.HiddenOutput` model is returned.

    Returns
    -------
    Loaded model.
    """
    # TODO - update this to accept tf format - later PR.
    model_dir = Path(filepath).joinpath(load_dir)
    # Check if model exists
    if 'model.h5' not in [f.name for f in model_dir.glob('[!.]*.h5')]:
        raise FileNotFoundError(f'No .h5 file found in {model_dir}.')
    model = tf.keras.models.load_model(model_dir.joinpath('model.h5'), custom_objects=custom_objects)
    # Optionally extract hidden layer
    if isinstance(layer, int):
        model = HiddenOutput(model, layer=layer)
    return model


def load_optimizer(cfg: dict) -> Union[Type[tf.keras.optimizers.Optimizer], tf.keras.optimizers.Optimizer]:
    """
    Loads a TensorFlow optimzier from a optimizer config dict.

    Parameters
    ----------
    cfg
        The optimizer config dict.

    Returns
    -------
    The loaded optimizer, either as an instantiated object (if `cfg` is a tensorflow optimizer config dict), otherwise
    as an uninstantiated class.
    """
    class_name = cfg.get('class_name')
    tf_config = cfg.get('config')
    if tf_config is not None:  # cfg is a tensorflow config dict
        return tf.keras.optimizers.deserialize(cfg)
    else:
        try:
            return getattr(import_module('tensorflow.keras.optimizers'), class_name)
        except AttributeError:
            raise ValueError(f"{class_name} is not a recognised optimizer in `tensorflow.keras.optimizers`.")


#######################################################################################################
# TODO: Everything below here is legacy loading code, and will be removed in the future
#######################################################################################################
def load_detector_legacy(filepath: Union[str, os.PathLike], suffix: str, **kwargs) -> Detector:
    """
    Legacy function to load outlier, drift or adversarial detectors stored dill or pickle files.

    Warning
    -------
    This function will be removed in a future version.

    Parameters
    ----------
    filepath
        Load directory.
    suffix
        File suffix for meta and state files. Either `'.dill'` or `'.pickle'`.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    warnings.warn('Loading of meta.dill and meta.pickle files will be removed in a future version.', DeprecationWarning)

    if kwargs:
        k = list(kwargs.keys())
    else:
        k = []

    # check if path exists
    filepath = Path(filepath)
    if not filepath.is_dir():
        raise FileNotFoundError(f'{filepath} does not exist.')

    # load metadata
    meta_dict = dill.load(open(filepath.joinpath('meta' + suffix), 'rb'))

    # check version
    try:
        if meta_dict['version'] != __version__:
            warnings.warn(f'Trying to load detector from version {meta_dict["version"]} when using version '
                          f'{__version__}. This may lead to breaking code or invalid results.')
    except KeyError:
        warnings.warn('Trying to load detector from an older version.'
                      'This may lead to breaking code or invalid results.')

    if 'backend' in list(meta_dict.keys()) and meta_dict['backend'] == Framework.PYTORCH:
        raise NotImplementedError('Detectors with PyTorch backend are not yet supported.')

    detector_name = meta_dict['name']
    if detector_name not in [detector for detector in VALID_DETECTORS]:
        raise NotImplementedError(f'{detector_name} is not supported by `load_detector`.')

    # load outlier detector specific parameters
    state_dict = dill.load(open(filepath.joinpath(detector_name + suffix), 'rb'))

    # initialize detector
    detector = None  # type: Optional[Detector]  # to avoid mypy errors

    if detector_name in ['SDDMDrift', 'HDDDMDrift']:
        emb, tokenizer = None, None
        if state_dict.get('other'):
            if state_dict.get('other').get('load_text_embedding'):
                emb, tokenizer = load_text_embed(filepath)
        # if state_dict['other']['load_text_embedding']:
        #     emb, tokenizer = load_text_embed(filepath)
        if detector_name == 'HDDDMDrift':
            load_fn = init_cd_hdddmdrift
        elif detector_name == 'SDDMDrift':
            cnn_model = load_model(filepath, 'model')
            dtr_model = load_sddm_dtr(filepath, load_dir='model')
            load_fn = partial(init_cd_sddmdrift, cnn_model, dtr_model)
        else:
            raise NotImplementedError
        model = None
        detector = load_fn(state_dict, model, emb, tokenizer, **kwargs)  # type: ignore[assignment]
    else:
        raise NotImplementedError

    # TODO - add tests back in!

    detector.meta = meta_dict
    logger.info('Finished loading detector.')
    return detector


def load_sddm_dtr(filepath: Union[str, os.PathLike],
                  load_dir: str = 'model') -> DecisionTreeRegressor:
    model_dir = Path(filepath).joinpath(load_dir)
    model = joblib.load(model_dir.joinpath('dtr_model.joblib'))
    return model


def init_preprocess(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                    emb, tokenizer, **kwargs) \
        -> Tuple[Optional[Callable], Optional[dict]]:
    """ Return preprocessing function and kwargs. """
    if kwargs:  # override defaults
        keys = list(kwargs.keys())
        preprocess_fn = kwargs['preprocess_fn'] if 'preprocess_fn' in keys else None
        preprocess_kwargs = kwargs['preprocess_kwargs'] if 'preprocess_kwargs' in keys else None
        return preprocess_fn, preprocess_kwargs
    elif model is not None and callable(state_dict['preprocess_fn']) \
            and isinstance(state_dict['preprocess_kwargs'], dict):
        preprocess_fn = state_dict['preprocess_fn']
        preprocess_kwargs = state_dict['preprocess_kwargs']
    else:
        return None, None

    keys = list(preprocess_kwargs.keys())

    if 'model' not in keys:
        raise ValueError('No model found for the preprocessing step.')

    preprocess_kwargs['model'] = model

    return preprocess_fn, preprocess_kwargs


def init_cd_sddmdrift(cnn_model: tf.keras.Model, dtr_Model: DecisionTreeRegressor, state_dict: Dict,
                      model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                      emb, tokenizer: Optional[Callable], **kwargs) \
        -> SDDMDrift:
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict['other'], model, emb, tokenizer, **kwargs)
    if callable(preprocess_fn) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    args = list(state_dict['args'].values()) + [cnn_model, dtr_Model]
    cd = SDDMDrift(*args, **state_dict['kwargs'])
    attrs = state_dict['other']
    cd._detector.n = attrs['n']
    return cd


def init_cd_hdddmdrift(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                       emb, tokenizer: Optional[Callable], **kwargs) \
        -> HDDDMDrift:
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict['other'], model, emb, tokenizer, **kwargs)
    if callable(preprocess_fn) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = HDDDMDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    return cd


def load_text_embed(filepath: Union[str, os.PathLike], load_dir: str = 'model') \
        -> Tuple[TransformerEmbedding, Callable]:
    """Legacy function to load text embedding."""
    model_dir = Path(filepath).joinpath(load_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()))
    args = dill.load(open(model_dir.joinpath('embedding.dill'), 'rb'))
    emb = TransformerEmbedding(
        str(model_dir.resolve()), embedding_type=args['embedding_type'], layers=args['layers']
    )
    return emb, tokenizer
