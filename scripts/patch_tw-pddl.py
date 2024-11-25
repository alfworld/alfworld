# This script is to patch the 2.1.1 game.tw-pddl files to bring up to 2.1.2.
# This script will add a new "help" action and separate the "PutObject" action
#  into two separate actions: PutObjectInContainer and PutObjectOnSupporter.
# This script will also add the corresponding grammar for the new actions.
# This script also patch the grammar to fix a typo in the go-to feedback.
# The script will create a backup of the original file before patching.

import os
import json
from glob import glob
from os.path import join as pjoin
from tqdm import tqdm

from alfworld.info import ALFWORLD_DATA
import os
import json

import tqdm


HELP_ACTION_PDDL = """\
(:action help
    :parameters (?a - agent)
    :precondition
        ()
    :effect
        (and
            (checked ?a)
        )
)
"""

HELP_GRAMMAR = """\

action help {
    template :: "help";
    feedback :: "\nAvailable commands:\n  look:                             look around your current location\n  inventory:                        check your current inventory\n  go to (receptacle):               move to a receptacle\n  open (receptacle):                open a receptacle\n  close (receptacle):               close a receptacle\n  take (object) from (receptacle):  take an object from a receptacle\n  move (object) to (receptacle):  place an object in or on a receptacle\n  examine (something):              examine a receptacle or an object\n  use (object):                     use an object\n  heat (object) with (receptacle):  heat an object using a receptacle\n  clean (object) with (receptacle): clean an object using a receptacle\n  cool (object) with (receptacle):  cool an object using a receptacle\n  slice (object) with (object):     slice an object using a sharp object\n";
}
"""

def patch_twpddl(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    # Make backup if doesn't exist
    if not os.path.exists(filename + ".bak"):
        with open(filename + ".bak", "w") as f:
            json.dump(data, f)

    # Always start from backup.
    with open(filename + ".bak") as f:
        data = json.load(f)

    # Patch domain pddl.
    before, after = data["pddl_domain"].rsplit(")", 1)
    data["pddl_domain"] = f"{before}{HELP_ACTION_PDDL}){after}"

    # Patch grammar.
    data["grammar"] = data["grammar"].replace("You arrive at {lend.name}.", "You arrive at {r.name}.")
    data["grammar"] = data["grammar"].replace("put {o} in/on {r}", "move {o} to {r}")
    data["grammar"] = data["grammar"].replace("You put the {o.name} in/on the {r.name}.", "You move the {o.name} to the {r.name}.")
    data["grammar"] += HELP_GRAMMAR

    with open(filename, "w") as f:
        json.dump(data, f)


filenames = glob(pjoin(ALFWORLD_DATA, "json_2.1.1/**/**/**/*.tw-pddl"))

for filename in tqdm(filenames):
   patch_twpddl(filename)
