import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Union, TYPE_CHECKING
import dill
from transformers import PreTrainedTokenizerBase

from bdilab_detect.saving._typing import VALID_DETECTORS
from bdilab_detect.utils.frameworks import Framework
from bdilab_detect.base import Detector, ConfigurableDetector
from bdilab_detect.saving._tensorflow.saving import save_detector_legacy

if TYPE_CHECKING:
    pass

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

logger = logging.getLogger(__name__)

X_REF_FILENAME = 'x_ref.npy'
C_REF_FILENAME = 'c_ref.npy'


def save_detector(
        detector,
        filepath: Union[str, os.PathLike], legacy: bool = False
) -> None:
    """
    Save outlier, drift or adversarial detector.

    Parameters
    ----------
    detector
        Detector object.
    filepath
        Save directory.
    legacy
        Whether to save in the legacy .dill format instead of via a config.toml file. Default is `False`.
        This option will be removed in a future version.
    """
    if legacy:
        warnings.warn('The `legacy` option will be removed in a future version.', DeprecationWarning)

    if 'backend' in list(detector.meta.keys()) and detector.meta['backend'] == Framework.KEOPS:
        raise NotImplementedError('Saving detectors with keops backend is not yet supported.')

    # TODO: Replace .__args__ w/ typing.get_args() once Python 3.7 dropped (and remove type ignore below)
    detector_name = detector.__class__.__name__
    if detector_name not in [detector for detector in VALID_DETECTORS]:
        raise NotImplementedError(f'{detector_name} is not supported by `save_detector`.')

    # Saving is wrapped in a try, with cleanup in except. To prevent a half-saved detector remaining upon error.
    filepath = Path(filepath)
    try:
        # Create directory if it doesn't exist
        if not filepath.is_dir():
            logger.warning('Directory {} does not exist and is now created.'.format(filepath))
            filepath.mkdir(parents=True, exist_ok=True)

        # If a drift detector, wrap drift detector save method

        save_detector_legacy(detector, filepath)

    except Exception as error:
        # Get a list of all existing files in `filepath` (so we know what not to cleanup if an error occurs)
        orig_files = set(filepath.iterdir())
        _cleanup_filepath(orig_files, filepath)
        raise RuntimeError(f'Saving failed. The save directory {filepath} has been cleaned.') from error

    logger.info('finished saving.')


def _cleanup_filepath(orig_files: set, filepath: Path):
    """
    Cleans up the `filepath` directory in the event of a saving failure.

    Parameters
    ----------
    orig_files
        Set of original files (not to delete).
    filepath
        The directory to clean up.
    """
    # Find new files
    new_files = set(filepath.iterdir())
    files_to_rm = new_files - orig_files
    # Delete new files
    for file in files_to_rm:
        if file.is_dir():
            shutil.rmtree(file)
        elif file.is_file():
            file.unlink()

    # Delete filepath directory if it is now empty
    if filepath is not None:
        if not any(filepath.iterdir()):
            filepath.rmdir()


def _path2str(cfg: dict, absolute: bool = False) -> dict:
    """
    Private function to traverse a config dict and convert pathlib Path's to strings.

    Parameters
    ----------
    cfg
        The config dict.
    absolute
        Whether to convert to absolute filepaths.

    Returns
    -------
    The converted config dict.
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            _path2str(v, absolute)
        elif isinstance(v, Path):
            if absolute:
                v = v.resolve()
            cfg.update({k: str(v.as_posix())})
    return cfg


def _int2str_keys(dikt: dict) -> dict:
    """
    Private function to traverse a dict and convert any dict's with int keys to str keys (e.g.
    `categories_per_feature` kwarg for `TabularDrift`.

    Parameters
    ----------
    dikt
        The dictionary.

    Returns
    -------
    The converted dictionary.
    """
    dikt_copy = dikt.copy()
    for k, v in dikt.items():
        if isinstance(k, int):
            dikt_copy[str(k)] = dikt[k]
            dikt_copy.pop(k)
        if isinstance(v, dict):
            dikt_copy[k] = _int2str_keys(v)
    return dikt_copy


def _save_tokenizer_config(tokenizer: PreTrainedTokenizerBase,
                           base_path: Path,
                           path: Path = Path('')) -> dict:
    """
    Saves HuggingFace tokenizers.

    Parameters
    ----------
    tokenizer
        The tokenizer.
    base_path
        Base filepath to save to.
    path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    The tokenizer config dict.
    """
    # create folder to save model in
    filepath = base_path.joinpath(path).joinpath('tokenizer')
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    cfg_token = {}
    logger.info('Saving tokenizer to {}.'.format(filepath))
    tokenizer.save_pretrained(filepath)
    cfg_token.update({'src': path.joinpath('tokenizer')})
    return cfg_token
