import os
import re
import time
import random
import json
import glob
import hashlib
import argparse
from termcolor import colored
import pprint
import numpy as np

import textworld
from textworld.agents import HumanAgent, RandomCommandAgent
from textworld.logic import Proposition, Variable

import sys
sys.path.append(os.environ["ALFRED_ROOT"])
from utils.misc import Demangler, get_templated_task_desc, clean_alfred_facts


class RandomAgent:
    def __init__(self, batch_size, seed=1234):
        self.batch_size = batch_size
        self.seed = seed
        self.rngs = [np.random.RandomState(self.seed + i) for i in range(batch_size)]

    def act(self, obs, rewards, dones, infos):
        return [rng.choice(commands) for rng, commands in zip(self.rngs, infos["admissible_commands"])]


class AlfredDemangler(textworld.core.Wrapper):

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._game.infos)
        for info in self._game.infos.values():
            info.name = demangler.demangle_alfred_name(info.id)


def batch_main(args):
    import gym
    import textworld.gym

    GAME_LOGIC = {
        "pddl_domain": open(args.domain).read(),
        "grammar": "\n".join(open(f).read() for f in glob.glob("textworld_data/logic/*.twl2")),
    }

    # Iterate through task folders and build .tw-pddl game files.
    gamefiles = []
    for root, _, files in os.walk(args.data, topdown=False):
        if 'traj_data.json' not in files:
            continue

        if 'movable' in root:
            continue

        # Load state and trajectory files
        pddl_file = os.path.join(root, 'initial_state.pddl')

        if not os.path.exists(pddl_file):
            if args.verbose:
                print("Missing files. Skipping %s" % root)

            continue

        # To avoid making .tw game file, we are going to load the gamedata directly.
        gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
        gamefile = os.path.join(root, 'game.tw-pddl')
        json.dump(gamedata, open(gamefile, "w"))
        gamefiles.append(gamefile)

    # Register a new Gym environment.
    infos = textworld.EnvInfos(won=True, admissible_commands=True)
    env_id = textworld.gym.register_games(gamefiles, infos,
                                          batch_size=args.batch_size,
                                          asynchronous=True,
                                          max_episode_steps=args.max_episode_len,
                                          wrappers=[AlfredDemangler])

    # Make a random agent.
    agent = RandomAgent(batch_size=args.batch_size, seed=args.random_seed)

    # Launch Gym environment.
    env = gym.make(env_id)
    obs, infos = env.reset()

    dones = [False] * args.batch_size
    rewards = [0] * args.batch_size
    while not all(dones):
        cmds = agent.act(obs, rewards, dones, infos)
        obs, rewards, dones, infos = env.step(cmds)
        print(obs[0])

    print(infos["won"])


def main(args):

    GAME_LOGIC = {
        "pddl_domain": open(args.domain).read(),
        "grammar": "\n".join(open(f).read() for f in glob.glob("textworld_data/logic/*.twl2")),
    }

    infos = textworld.EnvInfos(admissible_commands=True)
    env = textworld.envs.PddlEnv(infos)
    env = AlfredDemangler(env)

    # Store results
    results = {
        'num_tasks': 0,
        'num_success': 0,
        'num_failures': 0,
        'total_actions': 0,
        'actions_per_sec': 0.0,
        'tasks': []
    }
    exec_time = 0.0

    agent = RandomCommandAgent(seed=args.random_seed)

    # Iterate through task folders
    for root, _, files in os.walk(args.data, topdown=False):
        if 'traj_data.json' not in files:
            continue

        if 'movable' in root:
            continue

        for eps in range(args.num_episodes_per_game):
            print("Game: %s, Episode: %d" % (root, eps))

            # Load state and trajectory files
            pddl_file = os.path.join(root, 'initial_state.pddl')
            json_file = os.path.join(root, 'traj_data.json')

            if not os.path.exists(pddl_file) or not os.path.exists(pddl_file):
                if args.verbose:
                    print("Missing files. Skipping %s" % root)
                continue

            with open(json_file, 'r') as f:
                traj_data = json.load(f)

            # To avoid making .tw game file, we are going to load the gamedata directly.
            gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())

            # Load task/domain
            try:
                env.load(gamedata)
            except Exception as e:
                print(e)
                print("Task is unsolvable or trivial")
                from ipdb import set_trace; set_trace()
                break

            agent.reset(env)

            # Interact with the ALFRED environment.
            obs = env.reset()
            task_desc = get_templated_task_desc(traj_data)
            obs.feedback += "\n\nYour task is to: %s\n" % task_desc

            # Task desc
            task = {
                'goal_str': task_desc,
                'trial_id': root,
                'intro': obs.feedback,
                'success': False,
                'episode_len': 0
            }

            step = 0
            while True:
                print(obs.feedback)
                cmd = agent.act(obs, 0, False)

                stime = time.time()
                obs, _, done = env.step(cmd)
                etime = time.time()
                exec_time += (etime - stime)

                step += 1
                if args.verbose:
                    print(colored("\n".join(sorted(map(str, clean_alfred_facts(obs.effects)))), "yellow"))

                if done:
                    print("Task completed! You won!")
                    break

                if step >= args.max_episode_len:
                    print("Failed. Exceeded max episode length")
                    break

            task['episode_len'] = step
            task['success'] = True if done else False

            results['num_tasks'] += 1
            results['total_actions'] += step
            if done:
                results['num_success'] += 1
            else:
                results['num_failures'] += 1
            results['actions_per_sec'] = float(results['total_actions']) / exec_time
            results['tasks'].append(task)

            for k in results.keys():
                if 'tasks' not in k:
                    print("%s: %s" % (k, str(results[k])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",
                        default=os.environ.get("ALFRED_ROOT", ".") + "/gen/planner/domains/PutTaskExtended_domain.pddl",
                        help="Path to a PDDL file describing the domain."
                             " Default: `%(default)s`.")
    parser.add_argument("--data", default=os.environ.get("ALFRED_ROOT", ".") + "data/json_2.1.0/valid_unseen")
    parser.add_argument("--max_episode_len", type=int, default=100)
    parser.add_argument("--num_episodes_per_game", type=int, default=1000)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    #main(args)
    batch_main(args)
