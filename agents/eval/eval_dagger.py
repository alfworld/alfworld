import datetime
import os
import random
import time
import copy
import json
import glob
import importlib
import numpy as np

import sys
sys.path.append(os.environ['ALFRED_ROOT'])
from agent.agent import DAggerAgent
import modules.generic as generic
import eval.evaluate as evaluate
from environment import AlfredTWEnv, AlfredThorEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train():

    config = generic.load_config()
    agent = DAggerAgent(config)

    id_eval_env, num_id_eval_game = None, 0
    ood_eval_env, num_ood_eval_game = None, 0
    if agent.run_eval:
        # in distribution
        if config['dataset']['eval_id_data_path'] is not None:
            alfred_env = getattr(importlib.import_module("environment"), config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_in_distribution")
            id_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_id_eval_game = alfred_env.num_games
        # out of distribution
        if config['dataset']['eval_ood_data_path'] is not None:
            alfred_env = getattr(importlib.import_module("environment"), config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
            ood_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_ood_eval_game = alfred_env.num_games

    output_dir = os.getenv('PT_OUTPUT_DIR', '/tmp') if agent.philly else config["general"]["save_path"]
    data_dir = os.environ['PT_DATA_DIR'] if agent.philly else config["general"]["save_path"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json_file_name = agent.experiment_tag.replace(" ", "_")

    # load model from checkpoint
    load_from_tag_id, load_from_tag_ood = agent.load_from_tag.split("<<<SEP>>>")

    if id_eval_env is None:
        print("id eval env is None")
        exit(0)

    if os.path.exists(load_from_tag_id + "_id.pt"):
        agent.load_pretrained_model(load_from_tag_id + "_id.pt")
    else:
        print("file not found: " + load_from_tag_id)
        exit(0)

    id_eval_game_points, id_eval_game_step = 0.0, 0.0
    id_eval_res = evaluate.evaluate_dagger(id_eval_env, agent, num_id_eval_game)
    id_eval_game_points, id_eval_game_step = id_eval_res['average_points'], id_eval_res['average_steps']

    if ood_eval_env is None:
        print("ood eval env is None")
        exit(0)

    if os.path.exists(load_from_tag_ood + "_ood.pt"):
        agent.load_pretrained_model(load_from_tag_ood + "_ood.pt")
    else:
        print("file not found: " + load_from_tag_ood)
        exit(0)

    ood_eval_game_points, ood_eval_game_step = 0.0, 0.0
    ood_eval_res = evaluate.evaluate_dagger(ood_eval_env, agent, num_ood_eval_game)
    ood_eval_game_points, ood_eval_game_step = ood_eval_res['average_points'], ood_eval_res['average_steps']

    # write accuracies down into file
    _s = json.dumps({"id eval game points": str(id_eval_game_points),
                     "id eval steps": str(id_eval_game_step),
                     "ood eval game points": str(ood_eval_game_points),
                     "ood eval steps": str(ood_eval_game_step)})
    with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
        outfile.write(_s + '\n')
        outfile.flush()


if __name__ == '__main__':
    train()
