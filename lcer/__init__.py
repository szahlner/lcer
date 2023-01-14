import os

from lcer.her import HER
from lcer.sac import SAC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


__all__ = ["HER", "SAC"]
