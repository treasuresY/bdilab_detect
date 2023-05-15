from typing import Union, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    # Import the true objects directly for typechecking. (See note in CONTRIBUTING.md in Optional Dependencies section)
    from bdilab_detect.cd.hdddm import HDDDMDrift
    from bdilab_detect.cd.sddm import SDDMDrift
Data = Union[
    'BaseDetector',
    'HDDDMDrift',
    'SDDMDrift'
]
