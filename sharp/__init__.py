"""
Implementation of the ShaRP algorithm (Shapley for Rankings and Preferences).

``sharp`` is a library containing the implementation of the ShaRP
algorithm (Shapley for Rankings and Preferences), a framework that can be used
to explain the contributions of features to different aspects of a ranked
outcome, based on Shapley values.

Subpackages
-----------
qoi
utils
"""

from . import qoi
from . import utils
from .base import ShaRP
from ._version import __version__

__all__ = ["qoi", "utils", "ShaRP", "__version__"]
