__version__ = '0.1.0'

import os
import shutil
from os.path import join as pjoin


def _maybe_mkdirs(dirpath):
    """ Create all parent folders if needed. """
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    return dirpath


# _default_alfworld_cache = os.path.expanduser("~/.cache/alfworld")
# ALFWORLD_CACHE = _maybe_mkdirs(os.getenv("ALFWORLD_DATA", _default_alfworld_cache))
# DETECTORS_CACHE = _maybe_mkdirs(pjoin(ALFWORLD_CACHE, "detectors"))
# AGENTS_CACHE = _maybe_mkdirs(pjoin(ALFWORLD_CACHE, "agents"))
# LOGIC_CACHE = _maybe_mkdirs(pjoin(ALFWORLD_CACHE, "logic"))
# os.environ["ALFWORLD_DATA"] = ALFWORLD_CACHE  # Set the environment variable, in case it wasn't.

BUILTIN_DATA_PATH = pjoin(os.path.dirname(__file__), "data")
ALFRED_PDDL_PATH = pjoin(BUILTIN_DATA_PATH, 'alfred.pddl')
ALFRED_TWL2_PATH = pjoin(BUILTIN_DATA_PATH, 'alfred.twl2')
