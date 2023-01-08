import logging
import os
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, Callable
from functools import partial

import joblib
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from bdilab_detect.cd.tensorflow import UAE, HiddenOutput
from bdilab_detect.cd.tensorflow.sddm import SDDMDriftTF
from bdilab_detect.models.tensorflow import TransformerEmbedding
from bdilab_detect.utils._types import Literal
import dill  # dispatch table setting not done here as done in top-level saving.py file
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer
# from tensorflow.keras.layers import Input, InputLayer

from bdilab_detect.cd import HDDDMDrift, SDDMDrift

logger = logging.getLogger(__name__)


def save_model(model: tf.keras.Model,
               filepath: Union[str, os.PathLike],
               save_dir: Union[str, os.PathLike] = 'model',
               save_format: Literal['tf', 'h5'] = 'h5') -> None:  # TODO - change to tf, later PR
    """
    Save TensorFlow model.

    Parameters
    ----------
    model
        The tf.keras.Model to save.
    filepath
        Save directory.
    save_dir
        Name of folder to save to within the filepath directory.
    save_format
        The format to save to. 'tf' to save to the newer SavedModel format, 'h5' to save to the lighter-weight
        legacy hdf5 format.
    """
    # create folder to save model in
    model_path = Path(filepath).joinpath(save_dir)
    if not model_path.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_path))
        model_path.mkdir(parents=True, exist_ok=True)

    # save model
    model_path = model_path.joinpath('model.h5') if save_format == 'h5' else model_path

    if isinstance(model, tf.keras.Model):
        model.save(model_path, save_format=save_format)
    else:
        raise ValueError('The extracted model to save is not a `tf.keras.Model`. Cannot save.')


def save_optimizer_config(optimizer: tf.keras.optimizers.Optimizer):
    """

    Parameters
    ----------
    optimizer
        The tensorflow optimizer to serialize.

    Returns
    -------
    The tensorflow optimizer's config dictionary.
    """
    return tf.keras.optimizers.serialize(optimizer)


#######################################################################################################
# TODO: Everything below here is legacy saving code, and will be removed in the future
#######################################################################################################
def save_embedding_legacy(embed,
                          embed_args: dict,
                          filepath: Path) -> None:
    """
    Save embeddings for text drift models.

    Parameters
    ----------
    embed
        Embedding model.
    embed_args
        Arguments for TransformerEmbedding module.
    filepath
        The save directory.
    """
    # create folder to save model in
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # Save embedding model
    logger.info('Saving embedding model to {}.'.format(filepath.joinpath('embedding.dill')))
    embed.save_pretrained(filepath)
    with open(filepath.joinpath('embedding.dill'), 'wb') as f:
        dill.dump(embed_args, f)


def save_detector_legacy(detector, filepath):
    detector_name = detector.meta['name']

    # save metadata
    logger.info('Saving metadata and detector to {}'.format(filepath))

    with open(filepath.joinpath('meta.dill'), 'wb') as f:
        dill.dump(detector.meta, f)
    model = None
    embed = None
    tokenizer = None
    embed_args = {}
    cnn_model = None
    dtr_model = None

    if isinstance(detector, HDDDMDrift):
        state_dict = state_hdddm(detector)
    elif isinstance(detector, SDDMDrift):
        state_dict, cnn_model, dtr_model, model, embed, embed_args, tokenizer = state_sddmdrift(detector)
    else:
        raise NotImplementedError('The %s detector does not have a legacy save method.' % detector_name)

    with open(filepath.joinpath(detector_name + '.dill'), 'wb') as f:
        dill.dump(state_dict, f)

    # save detector specific models
    if isinstance(detector, SDDMDrift):
        if model is not None:
            save_model(model, filepath, save_dir='encoder')
        if embed is not None:
            save_embedding_legacy(embed, embed_args, filepath)
        if tokenizer is not None:
            tokenizer.save_pretrained(filepath.joinpath('model'))
        if cnn_model is not None:
            save_model(cnn_model, filepath, save_dir='model')
        if dtr_model is not None:
            save_sddm_dtr(dtr_model, filepath, save_dir='model')


def preprocess_step_drift(cd: Union[SDDMDriftTF]) -> Tuple[
    Optional[Callable], Dict, Optional[tf.keras.Model],
    Optional[TransformerEmbedding], Dict, Optional[Callable], bool
]:
    # note: need to be able to dill tokenizers other than transformers
    preprocess_fn, preprocess_kwargs = None, {}
    model, embed, embed_args, tokenizer, load_emb = None, None, {}, None, False
    if isinstance(cd.preprocess_fn, partial):
        preprocess_fn = cd.preprocess_fn.func
        for k, v in cd.preprocess_fn.keywords.items():
            if isinstance(v, UAE):
                if isinstance(v.encoder.layers[0], TransformerEmbedding):  # text drift
                    # embedding
                    embed = v.encoder.layers[0].model
                    embed_args = dict(
                        embedding_type=v.encoder.layers[0].emb_type,
                        layers=v.encoder.layers[0].hs_emb.keywords['layers']
                    )
                    load_emb = True

                    # preprocessing encoder
                    inputs = Input(shape=cd.input_shape, dtype=tf.int64)
                    v.encoder.call(inputs)
                    shape_enc = (v.encoder.layers[0].output.shape[-1],)
                    layers = [InputLayer(input_shape=shape_enc)] + v.encoder.layers[1:]
                    model = tf.keras.Sequential(layers)
                    _ = model(tf.zeros((1,) + shape_enc))
                else:
                    model = v.encoder
                preprocess_kwargs['model'] = 'UAE'
            elif isinstance(v, HiddenOutput):
                model = v.model
                preprocess_kwargs['model'] = 'HiddenOutput'
            elif isinstance(v, tf.keras.Model):
                model = v
                preprocess_kwargs['model'] = 'custom'
            elif hasattr(v, '__module__'):
                if 'transformers' in v.__module__:  # transformers tokenizer
                    tokenizer = v
                    preprocess_kwargs[k] = v.__module__
            else:
                preprocess_kwargs[k] = v
    elif callable(cd.preprocess_fn):
        preprocess_fn = cd.preprocess_fn
    return preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb


def state_sddmdrift(cd: SDDMDrift) -> Tuple[
    Dict, tf.keras.Model, DecisionTreeRegressor,
    Optional[tf.keras.Model],
    Optional[TransformerEmbedding], Optional[Dict], Optional[Callable]
]:
    """
        SDDM drift detector parameters to save.
        Parameters
        ----------
        cd
            Drift detection object.
    """
    preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb = \
        preprocess_step_drift(cd._detector)
    state_dict = {
        "args": {
            'x_ref': cd._detector.x_ref,
            'y_ref': cd._detector.y_ref
        },
        "kwargs": {
            'p_val': cd._detector.p_val,
            'x_ref_preprocessed': True,
            'preprocess_at_init': cd._detector.preprocess_at_init,
            'update_x_ref': cd._detector.update_x_ref,
            'preds_type': cd._detector.preds_type,
            'binarize_preds': cd._detector.binarize_preds,
            'window_size': cd._detector.window_size,
            'shap_class': cd._detector.shap_class,
            'threshold': cd._detector.threshold,
        },
        "other": {
            'n': cd._detector.n,
            'load_text_embedding': load_emb,
            'preprocess_fn': preprocess_fn,
            'preprocess_kwargs': preprocess_kwargs
        }
    }
    cnn_model = cd._detector.shap_model.model
    dtr_model = cd._detector.shap_predict
    return state_dict, cnn_model, dtr_model, model, embed, embed_args, tokenizer


def state_hdddm(cd):
    state_dict = {
        'args':
            {
                'X_baseline': cd.X_baseline
            },
        'kwargs':
            {
                'gamma': cd.gamma,
                'alpha': cd.alpha,
                'use_mmd2': cd.use_mmd2,
                'use_k2s_test': cd.use_k2s_test,
            },
        'other':
            {

            }
    }
    return state_dict


# 暂时写在_tensorflow包下了
def save_sddm_dtr(model: BaseEstimator,
                  filepath: Union[str, os.PathLike],
                  save_dir: Union[str, os.PathLike] = 'model') -> None:
    """
        Save scikit-learn (and xgboost) models. Models are assumed to be a subclass of :class:`~sklearn.base.BaseEstimator`.

        Parameters
        ----------
        model
            The BaseEstimator to save.
        filepath
            Save directory.
        save_dir
            Name of folder to save to within the filepath directory.
        """
    # create folder to save model in
    model_path = Path(filepath).joinpath(save_dir)
    if not model_path.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_path))
        model_path.mkdir(parents=True, exist_ok=True)

    # save model
    model_path = model_path.joinpath('dtr_model.joblib')
    joblib.dump(model, model_path)
