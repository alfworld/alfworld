import datetime
import os
import sys
import random
import time
import copy
import json
import glob
from os.path import join as pjoin

import numpy as np

from alfworld.info import ALFWORLD_DATA
import alfworld.agents.environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.agent import TextDAggerAgent

os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_or_eval = "train"

def collect_data(task_types):

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    config['general']['training']['batch_size'] = 32
    config['general']['evaluate']['batch_size'] = 1
    config['general']['observation_pool_capacity'] = 5
    config['general']['training_method'] = 'dagger'
    config['env']['task_types'] = task_types
    # config['env']['expert_type'] = "planner"

    if train_or_eval == "train":
        config['dataset']['data_path'] = pjoin(ALFWORLD_DATA, "json_2.1.1", "train")
    else:
        config['dataset']['data_path'] = pjoin(ALFWORLD_DATA, "json_2.1.1", "valid_seen")

    agent = TextDAggerAgent(config)
    alfred_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval="train")
    env = alfred_env.init_env(batch_size=agent.batch_size)
    num_game = alfred_env.num_games
    env.seed(42)
    np.random.seed(42)

    output_dir = config["general"]["save_path"]
    data_dir = config["general"]["save_path"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    episode_no = 0
    collected_data = []

    while(True):
        if episode_no >= num_game:
            break
        obs, infos = env.reset()
        game_names = infos["extra.gamefile"]
        batch_size = len(obs)

        agent.train()
        agent.init(batch_size)

        execute_actions = []
        prev_step_dones = []
        for _ in range(batch_size):
            execute_actions.append("restart")
            prev_step_dones.append(0.0)

        observation_strings = list(obs)
        task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)
        task_desc_strings = agent.preprocess_task(task_desc_strings)
        observation_strings = agent.preprocess_observation(observation_strings)
        first_sight_strings = copy.deepcopy(observation_strings)
        agent.observation_pool.push_first_sight(first_sight_strings)
        observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, execute_actions)]  # appending the chosen action at previous step into the observation
        episode_data = [[] for _ in range(batch_size)]
        still_running = [1.0 for _ in range(batch_size)]

        for step_no in range(200):
            # push obs into observation pool
            agent.observation_pool.push_batch(observation_strings)
            # get most recent k observations
            most_recent_observation_strings = agent.observation_pool.get()
            expert_actions = []
            for b in range(batch_size):
                if "extra.expert_plan" in infos and len(infos["extra.expert_plan"][b]) > 0:
                    next_action = infos["extra.expert_plan"][b][0]
                    expert_actions.append(next_action)
                else:
                    expert_actions.append("look")
            execute_actions = expert_actions

            for b in range(batch_size):
                if still_running[b] == 0:
                    continue
                _d = {"step_id": step_no, "obs": most_recent_observation_strings[b], "action": execute_actions[b]}
                episode_data[b].append(_d)

            obs, _, dones, infos = env.step(execute_actions)
            dones = [float(item) for item in dones]

            observation_strings = list(obs)
            observation_strings = agent.preprocess_observation(observation_strings)
            observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, execute_actions)]  # appending the chosen action at previous step into the observation

            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        time_2 = datetime.datetime.now()
        for b in range(batch_size):
            print("Episode: {:3d} | {:s} | time spent: {:s} | used steps: {:s}".format(episode_no + b, game_names[b], str(time_2 - time_1).rsplit(".")[0], str(len(episode_data[b]))))

        for b in range(batch_size):
            if len(collected_data) >= num_game:
                continue
            try:
                if len(episode_data[b]) >= 200:
                    continue
                collected_data.append({"g": "/".join(game_names[b].split("/")[-3:-1]), "g_id": episode_no, "task": task_desc_strings[b], "steps": episode_data[b]})
            except:
                pass

        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size

    with open("../data/seq2seq_data/tw_alfred_seq2seq_" + train_or_eval + "_task" + "-".join([str(item) for item in task_types]) + "_hc.json", 'w', encoding='utf-8') as json_file:
        json.dump({"data": collected_data}, json_file)


if __name__ == '__main__':
    if not os.path.exists("./data/seq2seq_data"):
        os.makedirs("./data/seq2seq_data")

    for task in [[1], [2], [3], [4], [5], [6]]:
        collect_data(task)
