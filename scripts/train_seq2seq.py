import datetime
import os
import json
import importlib
import numpy as np
from tqdm import tqdm

import sys

import alfworld.agents.environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.agent import TextDAggerAgent
from alfworld.agents.eval import evaluate_dagger
from alfworld.agents.modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_TRAIN_STEP = 50000
REPORT_FREQUENCY = 1000


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    agent = TextDAggerAgent(config)

    id_eval_env, num_id_eval_game = None, 0
    ood_eval_env, num_ood_eval_game = None, 0
    if agent.run_eval:
        # in distribution
        if config['dataset']['eval_id_data_path'] is not None:
            alfred_env = getattr(alfworld.agents.environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_in_distribution")
            id_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_id_eval_game = alfred_env.num_games
        # out of distribution
        if config['dataset']['eval_ood_data_path'] is not None:
            alfred_env = getattr(alfworld.agents.environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
            ood_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_ood_eval_game = alfred_env.num_games

    output_dir = config["general"]["save_path"]
    data_dir = config["general"]["save_path"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        reward_win, step_win = None, None
        loss_win = None
        viz_loss = []
        viz_id_eval_game_points, viz_id_eval_step = [], []
        viz_ood_eval_game_points, viz_ood_eval_step = [], []

    episode_no = 0
    running_avg_dagger_loss = HistoryScoreCache(capacity=500)

    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_performance_so_far, best_ood_performance_so_far = 0.0, 0.0

    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    # load dataset
    # push experience into replay buffer (dagger)
    task_types = config['env']['task_types']
    for tt in task_types:
        train_dataset = json.load(open(os.path.join(data_dir, "../data/seq2seq_data/", "tw_alfred_seq2seq_train_task" + str(tt) + "_hc.json"), 'r'))
        train_dataset = train_dataset["data"]
        for episode in tqdm(train_dataset):
            steps = episode["steps"]
            task = episode["task"]
            trajectory = []
            for i in range(len(steps)):
                obs = steps[i]["obs"]
                action = steps[i]["action"]
                trajectory.append([obs, task, None, action, None])
            agent.dagger_memory.push(trajectory)


    while(True):
        if episode_no > MAX_TRAIN_STEP:
            break

        agent.train()
        for i in range(4):  # average #step is 20, this is to keep same #updates with dagger
            dagger_loss = agent.update_dagger()
            if dagger_loss is not None:
                running_avg_dagger_loss.push(dagger_loss)

        report = (episode_no % REPORT_FREQUENCY == 0)
        episode_no += 10
        if not report:
            continue
        time_2 = datetime.datetime.now()
        print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], running_avg_dagger_loss.get_avg()))

        # evaluate
        id_eval_game_points, id_eval_game_step = 0.0, 0.0
        ood_eval_game_points, ood_eval_game_step = 0.0, 0.0
        if agent.run_eval:
            if id_eval_env is not None:
                id_eval_res = evaluate_dagger(id_eval_env, agent, num_id_eval_game)
                id_eval_game_points, id_eval_game_step = id_eval_res['average_points'], id_eval_res['average_steps']
            if ood_eval_env is not None:
                ood_eval_res = evaluate_dagger(ood_eval_env, agent, num_ood_eval_game)
                ood_eval_game_points, ood_eval_game_step = ood_eval_res['average_points'], ood_eval_res['average_steps']
            if id_eval_game_points >= best_performance_so_far:
                best_performance_so_far = id_eval_game_points
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_id.pt")
            if ood_eval_game_points >= best_ood_performance_so_far:
                best_ood_performance_so_far = ood_eval_game_points
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_ood.pt")
        else:
            if 1000 - running_avg_dagger_loss.get_avg() >= best_performance_so_far:
                best_performance_so_far = 1000 - running_avg_dagger_loss.get_avg()
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + ".pt")

        # plot using visdom
        if config["general"]["visdom"]:
            viz_loss.append(running_avg_dagger_loss.get_avg())
            viz_id_eval_game_points.append(id_eval_game_points)
            viz_id_eval_step.append(id_eval_game_step)
            viz_ood_eval_game_points.append(ood_eval_game_points)
            viz_ood_eval_step.append(ood_eval_game_step)
            viz_x = np.arange(len(viz_id_eval_game_points)).tolist()

            if reward_win is None:
                reward_win = viz.line(X=viz_x, Y=viz_id_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_id_eval_game_points"),
                        name="id eval game points")
                viz.line(X=viz_x, Y=viz_ood_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_ood_eval_game_points"),
                            win=reward_win, update='append', name="ood eval game points")
            else:
                viz.line(X=[len(viz_id_eval_game_points) - 1], Y=[viz_id_eval_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_id_eval_game_points"),
                            win=reward_win,
                            update='append', name="id eval game points")
                viz.line(X=[len(viz_ood_eval_game_points) - 1], Y=[viz_ood_eval_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_ood_eval_game_points"),
                            win=reward_win,
                            update='append', name="ood eval game points")

            if step_win is None:
                step_win = viz.line(X=viz_x, Y=viz_id_eval_step,
                                    opts=dict(title=agent.experiment_tag + "_id_eval_step"),
                                    name="id eval step")
                viz.line(X=viz_x, Y=viz_ood_eval_step,
                            opts=dict(title=agent.experiment_tag + "_ood_eval_step"),
                            win=step_win, update='append', name="ood eval step")
            else:
                viz.line(X=[len(viz_id_eval_step) - 1], Y=[viz_id_eval_step[-1]],
                            opts=dict(title=agent.experiment_tag + "_id_eval_step"),
                            win=step_win,
                            update='append', name="id eval step")
                viz.line(X=[len(viz_ood_eval_step) - 1], Y=[viz_ood_eval_step[-1]],
                            opts=dict(title=agent.experiment_tag + "_ood_eval_step"),
                            win=step_win,
                            update='append', name="ood eval step")

            if loss_win is None:
                loss_win = viz.line(X=viz_x, Y=viz_loss,
                                    opts=dict(title=agent.experiment_tag + "_loss"),
                                    name="loss")
            else:
                viz.line(X=[len(viz_loss) - 1], Y=[viz_loss[-1]],
                            opts=dict(title=agent.experiment_tag + "_loss"),
                            win=loss_win,
                            update='append', name="loss")

        # write accuracies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "time spent seconds": (time_2 - time_1).seconds,
                         "episodes": episode_no,
                         "loss": str(running_avg_dagger_loss.get_avg()),
                         "id eval game points": str(id_eval_game_points),
                         "id eval steps": str(id_eval_game_step),
                         "ood eval game points": str(ood_eval_game_points),
                         "ood eval steps": str(ood_eval_game_step)})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()
    agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_final.pt")


if __name__ == '__main__':
    train()
