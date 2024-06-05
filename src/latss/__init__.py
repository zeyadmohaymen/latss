from importlib.metadata import version
__version__ = version("latss")

from latss.latss import LATSS
from latss.data.load_data import load_data, unpack_fifs