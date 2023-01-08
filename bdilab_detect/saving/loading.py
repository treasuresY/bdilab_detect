import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING

import toml
from transformers import AutoTokenizer
from bdilab_detect.base import Detector, ConfigurableDetector
from bdilab_detect.saving._tensorflow.loading import load_detector_legacy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Fields to resolve in resolve_config ("resolve" meaning either load local artefact or resolve @registry, conversion to
# tuple, np.ndarray and np.dtype are dealt with separately).
# Note: For fields consisting of nested dicts, they must be listed in order from deepest to shallowest, so that the
# deepest fields are resolved first. e.g. 'preprocess_fn.src' must be resolved before 'preprocess_fn'.
FIELDS_TO_RESOLVE = [
    ['preprocess_fn', 'src'],
    ['preprocess_fn', 'model'],
    ['preprocess_fn', 'embedding'],
    ['preprocess_fn', 'tokenizer'],
    ['preprocess_fn', 'preprocess_batch_fn'],
    ['preprocess_fn'],
    ['x_ref'],
    ['c_ref'],
    ['model'],
    ['optimizer'],
    ['reg_loss_fn'],
    ['dataset'],
    ['kernel', 'src'],
    ['kernel', 'proj'],
    ['kernel', 'init_sigma_fn'],
    ['kernel', 'kernel_a', 'src'],
    ['kernel', 'kernel_a', 'init_sigma_fn'],
    ['kernel', 'kernel_b', 'src'],
    ['kernel', 'kernel_b', 'init_sigma_fn'],
    ['kernel'],
    ['x_kernel', 'src'],
    ['x_kernel', 'init_sigma_fn'],
    ['x_kernel'],
    ['c_kernel', 'src'],
    ['c_kernel', 'init_sigma_fn'],
    ['c_kernel'],
    ['initial_diffs'],
    ['tokenizer']
]

# Fields to convert from str to dtype
FIELDS_TO_DTYPE = [
    ['preprocess_fn', 'dtype']
]


def load_detector(filepath: Union[str, os.PathLike], **kwargs) -> Union[Detector, ConfigurableDetector]:
    """
    Load outlier, drift or adversarial detector.

    Parameters
    ----------
    filepath
        Load directory.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    filepath = Path(filepath)

    # Otherwise, if a directory, look for meta.dill, meta.pickle or config.toml inside it
    if filepath.is_dir():
        files = [str(f.name) for f in filepath.iterdir() if f.is_file()]

        if 'meta.dill' in files:
            return load_detector_legacy(filepath, '.dill', **kwargs)


    # No other file types are accepted, so if not dir raise error
    else:
        raise ValueError("load_detector accepts only a filepath to a directory, or a config.toml file.")


def _init_detector(cfg: dict) -> ConfigurableDetector:
    """
    Instantiates a detector from a fully resolved config dictionary.

    Parameters
    ----------
    cfg
        The detector's resolved config dictionary.

    Returns
    -------
    The instantiated detector.
    """
    detector_name = cfg.pop('name')

    # Instantiate the detector
    klass = getattr(import_module('alibi_detect.cd'), detector_name)
    detector = klass.from_config(cfg)
    logger.info('Instantiated drift detector {}'.format(detector_name))
    return detector


def _load_tokenizer_config(cfg: dict) -> AutoTokenizer:
    """
    Loads a text tokenizer from a tokenizer config dict.

    Parameters
    ----------
    cfg
        A tokenizer config dict. (see the pydantic schemas).

    Returns
    -------
    The loaded tokenizer.
    """
    src = cfg['src']
    kwargs = cfg['kwargs']
    src = Path(src)
    tokenizer = AutoTokenizer.from_pretrained(src, **kwargs)
    return tokenizer


def _get_nested_value(dic: dict, keys: list) -> Any:
    """
    Get a value from a nested dictionary.

    Parameters
    ----------
    dic
        The dictionary.
    keys
        List of keys to "walk" to nested value.
        For example, to extract the value `dic['key1']['key2']['key3']`, set `keys = ['key1', 'key2', 'key3']`.

    Returns
    -------
    The nested value specified by `keys`.
    """
    for key in keys:
        try:
            dic = dic[key]
        except (TypeError, KeyError):
            return None
    return dic


def _set_nested_value(dic: dict, keys: list, value: Any):
    """
    Set a value in a nested dictionary.

    Parameters
    ----------
    dic
        The dictionary.
    keys
        List of keys to "walk" to nested value.
        For example, to set the value `dic['key1']['key2']['key3']`, set `keys = ['key1', 'key2', 'key3']`.
    value
        The value to set.
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def read_config(filepath: Union[os.PathLike, str]) -> dict:
    """
    This function reads a detector toml config file and returns a dict specifying the detector.

    Parameters
    ----------
    filepath
        The filepath to the config.toml file.

    Returns
    -------
        Parsed toml dictionary.
    """
    filepath = Path(filepath)

    cfg = dict(toml.load(filepath))  # toml.load types return as MutableMapping, force to dict
    logger.info('Loaded config file from {}'.format(str(filepath)))

    # This is necessary as no None/null in toml spec., and missing values are set to defaults set in pydantic models.
    # But we sometimes need to explicitly spec as None.
    cfg = _replace(cfg, "None", None)

    return cfg


def _replace(cfg: dict, orig: Optional[str], new: Optional[str]) -> dict:
    """
    Recursively traverse a nested dictionary and replace values.

    Parameters
    ----------
    cfg
        The dictionary.
    orig
        Original value to search.
    new
        Value to replace original with.

    Returns
    -------
    The updated dictionary.
    """
    for k, v in cfg.items():
        if isinstance(v == orig, bool) and v == orig:
            cfg[k] = new
        elif isinstance(v, dict):
            _replace(v, orig, new)
    return cfg


def _prepend_cfg_filepaths(cfg: dict, prepend_dir: Path):
    """
    Recursively traverse through a nested dictionary and prepend a directory to any filepaths.

    Parameters
    ----------
    cfg
        The dictionary.
    prepend_dir
        The filepath to prepend to any filepaths in the dictionary.

    Returns
    -------
    The updated config dictionary.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            v = prepend_dir.joinpath(Path(v))
            if v.is_file() or v.is_dir():  # Update if prepending config_dir made config value a real filepath
                cfg[k] = str(v)
        elif isinstance(v, dict):
            _prepend_cfg_filepaths(v, prepend_dir)
