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
from utils.misc import Demangler, get_templated_task_desc, clean_alfred_facts, add_task_to_grammar
from eval_tw_agent import AlfredDemangler


def main(args):
    GAME_LOGIC = {
        "pddl_domain": open(args.domain).read(),
        "grammar": "\n".join(open(f).read() for f in glob.glob("textworld_data/logic/*.twl2")),
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

    # Interact with the ALFRED environment.
    # obs += "\n\nYour task is to: %s\n" % task_desc

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
        if args.verbose:
            print(colored("\n".join(sorted(map(str, clean_alfred_facts(obs.effects)))), "yellow"))

        if done:
            print("You won!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="Path to a folder contain initial state PDDL and traj_info.")
    parser.add_argument("--domain",
                        default=os.environ.get("ALFRED_ROOT", ".") + "/gen/planner/domains/PutTaskExtended_domain.pddl",
                        help="Path to a PDDL file describing the domain."
                             " Default: `%(default)s`.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
