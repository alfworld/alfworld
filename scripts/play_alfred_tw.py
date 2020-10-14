import os
import random
import json
import glob
import hashlib
import argparse
from termcolor import colored

import textworld
from textworld.agents import HumanAgent

import gym
import textworld.gym

import sys
sys.path.append(os.environ["ALFRED_ROOT"])
from agents.utils.misc import Demangler, get_templated_task_desc, clean_alfred_facts, add_task_to_grammar


class AlfredDemangler(textworld.core.Wrapper):

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._game.infos)
        for info in self._game.infos.values():
            info.name = demangler.demangle_alfred_name(info.id)

def main(args):
    GAME_LOGIC = {
        "pddl_domain": open(args.domain).read(),
        "grammar": "\n".join(open(f).read() for f in glob.glob("data/textworld_data/logic/*.twl2")),
    }

    # Load state and trajectory files
    pddl_file = os.path.join(args.problem, 'initial_state.pddl')
    json_file = os.path.join(args.problem, 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)
    GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)

    gamefiles = []
    gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
    json.dump(gamedata, open(gamefile, "w"))
    gamefiles.append(gamefile)

    # Register a new Gym environment.
    infos = textworld.EnvInfos(won=True, admissible_commands=True)
    env_id = textworld.gym.register_game(gamefile, infos,
                                         max_episode_steps=1000000,
                                         wrappers=[AlfredDemangler])

    env = gym.make(env_id)
    obs, infos = env.reset()


    agent = HumanAgent(True)
    agent.reset(env)

    while True:
        print(obs)
        cmd = agent.act(infos, 0, False)
        if cmd == "STATE":
            print("\n".join(sorted(map(str, clean_alfred_facts(obs._facts)))))
            continue

        elif cmd == "ipdb":
            from ipdb import set_trace; set_trace()
            continue

        obs, score, done, infos = env.step(cmd)

        if done:
            print("You won!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="Path to folder containing pddl and traj_data files")
    parser.add_argument("--domain",
                        default=os.environ.get("ALFRED_ROOT", ".") + "/gen/planner/domains/PutTaskExtended_domain.pddl",
                        help="Path to a PDDL file describing the domain."
                             " Default: `%(default)s`.")
    args = parser.parse_args()
    main(args)
