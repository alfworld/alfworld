#!/usr/bin/env python

import os
import sys
import json
import glob
import random
import argparse
from os.path import join as pjoin

import alfworld.agents
from alfworld.info import ALFWORLD_DATA
from alfworld.env.thor_env import ThorEnv
from alfworld.agents.detector.mrcnn import load_pretrained_model
from alfworld.agents.controller import OracleAgent, OracleAStarAgent, MaskRCNNAgent, MaskRCNNAStarAgent

prompt_toolkit_available = False
try:
    # For command line history and autocompletion.
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory
    prompt_toolkit_available = sys.stdout.isatty()
except ImportError:
    pass

try:
    # For command line history when prompt_toolkit is not available.
    import readline  # noqa: F401
except ImportError:
    pass


def setup_scene(env, traj_data, r_idx, args, reward_type='dense'):
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))

    # print goal instr
    print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    # setup task for reward
    env.set_task(traj_data, args, reward_type=reward_type)


def main(args):
    print(f"Playing '{args.problem}'.")

    # start THOR
    env = ThorEnv()

    # load traj_data
    root = args.problem
    json_file = os.path.join(root, 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)

    # setup scene
    setup_scene(env, traj_data, 0, args)

    # choose controller
    if args.controller == "oracle":
        AgentModule = OracleAgent
        agent = AgentModule(env, traj_data, traj_root=root, load_receps=args.load_receps, debug=args.debug)
    elif args.controller == "oracle_astar":
        AgentModule = OracleAStarAgent
        agent = AgentModule(env, traj_data, traj_root=root, load_receps=args.load_receps, debug=args.debug)
    elif args.controller == "mrcnn":
        AgentModule = MaskRCNNAgent
        mask_rcnn = load_pretrained_model(pjoin(ALFWORLD_DATA, "detectors", "mrcnn.pth"))
        agent = AgentModule(env, traj_data, traj_root=root,
                            pretrained_model=mask_rcnn,
                            load_receps=args.load_receps, debug=args.debug)
    elif args.controller == "mrcnn_astar":
        AgentModule = MaskRCNNAStarAgent
        mask_rcnn = load_pretrained_model(pjoin(ALFWORLD_DATA, "detectors", "mrcnn.pth"))
        agent = AgentModule(env, traj_data, traj_root=root,
                            pretrained_model=mask_rcnn,
                            load_receps=args.load_receps, debug=args.debug)
    else:
        raise NotImplementedError()

    history = None
    if prompt_toolkit_available:
        history = InMemoryHistory()

    print(agent.feedback)
    while True:
        if prompt_toolkit_available:
            actions_completer = None
            admissible_commands = agent.get_admissible_commands()
            actions_completer = WordCompleter(admissible_commands, ignore_case=True, sentence=True)
            cmd = prompt('> ', completer=actions_completer, history=history, enable_history_search=True)
        else:
            cmd = input('> ')

        if cmd == "ipdb":
            from ipdb import set_trace; set_trace()
            continue

        agent.step(cmd)
        if not args.debug:
            print(agent.feedback)

        done = env.get_goal_satisfied()
        if done:
            print("You won!")
            break


if __name__ == "__main__":
    description = "Play the abstract text version of an ALFRED environment."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("problem", nargs="?", default=None,
                        help="Path to a folder containing PDDL and traj_data files."
                             f"Default: pick one at random found in {ALFWORLD_DATA}")
    parser.add_argument("--controller", default="oracle", choices=["oracle", "oracle_astar", "mrcnn", "mrcnn_astar"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--load_receps', action="store_true")
    parser.add_argument('--reward_config', type=str, default=pjoin(alfworld.agents.__path__[0], 'config', 'rewards.json'))
    args = parser.parse_args()

    if args.problem is None:
        problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)

        # Remove problem which contains movable receptacles.
        problems = [p for p in problems if "movable_recep" not in p]

        if len(problems) == 0:
            raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
        
        args.problem = os.path.dirname(random.choice(problems))

    if "movable_recep" in args.problem:
        raise ValueError("This problem contains movable receptacles, which is not supported by ALFWorld.")

    main(args)
