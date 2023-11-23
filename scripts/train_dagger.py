import datetime
import os
import copy
import json
import importlib
import numpy as np

import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import alfworld.agents.environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.eval import evaluate_dagger
from alfworld.agents.agent import TextDAggerAgent
from alfworld.agents.utils.misc import extract_admissible_commands
from alfworld.agents.modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory


def train():

    time_1 = datetime.datetime.now()
    step_time = []
    config = generic.load_config()
    agent = TextDAggerAgent(config)
    alfred_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval="train")
    env = alfred_env.init_env(batch_size=agent.batch_size)

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
        viz_game_points, viz_game_step, viz_loss = [], [], []
        viz_student_points, viz_student_step = [], []
        viz_id_eval_game_points, viz_id_eval_step = [], []
        viz_ood_eval_game_points, viz_ood_eval_step = [], []

    step_in_total = 0
    episode_no = 0
    running_avg_game_points = HistoryScoreCache(capacity=500)
    running_avg_student_points = HistoryScoreCache(capacity=500)
    running_avg_game_steps = HistoryScoreCache(capacity=500)
    running_avg_student_steps = HistoryScoreCache(capacity=500)
    running_avg_dagger_loss = HistoryScoreCache(capacity=500)

    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_performance_so_far, best_ood_performance_so_far = 0.0, 0.0

    # load model from checkpoint
    if agent.load_pretrained:
        print("Checking {} for {}".format(data_dir, agent.load_from_tag))
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    while(True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        game_names = infos["extra.gamefile"]
        batch_size = len(obs)

        agent.train()
        agent.init(batch_size)
        previous_dynamics = None

        execute_actions = []
        prev_step_dones, prev_rewards = [], []
        for _ in range(batch_size):
            execute_actions.append("restart")
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)

        observation_strings = list(obs)
        task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)
        task_desc_strings = agent.preprocess_task(task_desc_strings)
        observation_strings = agent.preprocess_observation(observation_strings)
        first_sight_strings = copy.deepcopy(observation_strings)
        agent.observation_pool.push_first_sight(first_sight_strings)
        if agent.action_space == "exhaustive":
            action_candidate_list = [extract_admissible_commands(intro, obs) for intro, obs in zip(first_sight_strings, observation_strings)]
        else:
            action_candidate_list = list(infos["admissible_commands"])
        action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
        observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, execute_actions)]  # appending the chosen action at previous step into the observation

        # it requires to store sequences of transitions into memory with order,
        # so we use a cache to keep what agents returns, and push them into memory
        # altogether in the end of game.
        transition_cache = []
        still_running_mask = []
        sequence_game_points = []
        print_actions = []
        report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= (episode_no - batch_size) % agent.report_frequency)

        for step_no in range(agent.max_nb_steps_per_episode):
            # push obs into observation pool
            agent.observation_pool.push_batch(observation_strings)
            # get most recent k observations
            most_recent_observation_strings = agent.observation_pool.get()

            # predict actions
            if agent.action_space == "generation":
                agent_actions, current_dynamics = agent.command_generation_greedy_generation(most_recent_observation_strings, task_desc_strings, previous_dynamics)
            elif agent.action_space in ["admissible", "exhaustive"]:
                agent_actions, _, current_dynamics = agent.admissible_commands_greedy_generation(most_recent_observation_strings, task_desc_strings, action_candidate_list, previous_dynamics)
            else:
                raise NotImplementedError()

            random_actions, _ = agent.admissible_commands_random_generation(action_candidate_list)

            expert_actions, expert_indices = [], []
            for b in range(batch_size):
                if "extra.expert_plan" in infos and len(infos["extra.expert_plan"][b]) > 0:
                    next_action = infos["extra.expert_plan"][b][0]
                    expert_actions.append(next_action)
                    expert_indices.append(action_candidate_list[b].index(next_action) if agent.action_space in ["admissible", "exhaustive"] else -1)
                else:
                    expert_actions.append("look")
                    expert_indices.append(action_candidate_list[b].index("look") if agent.action_space in ["admissible", "exhaustive"] else -1)

            from_which = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            execute_actions = []
            for b in range(batch_size):
                if not report and from_which[b] <= agent.fraction_assist:
                    execute_actions.append(expert_actions[b])
                elif not report and from_which[b] <= agent.fraction_assist + agent.fraction_random:
                    execute_actions.append(random_actions[b])
                else:
                    execute_actions.append(agent_actions[b])

            replay_info = [most_recent_observation_strings, task_desc_strings, action_candidate_list, expert_actions, expert_indices]
            transition_cache.append(replay_info)

            env_step_start_time = datetime.datetime.now()
            obs, _, dones, infos = env.step(execute_actions)
            env_step_stop_time = datetime.datetime.now()
            step_time.append((env_step_stop_time-env_step_start_time).microseconds / (float(batch_size)))

            scores = [float(item) for item in infos["won"]]
            dones = [float(item) for item in dones]

            observation_strings = list(obs)
            observation_strings = agent.preprocess_observation(observation_strings)
            if agent.action_space == "exhaustive":
                action_candidate_list = [extract_admissible_commands(intro, obs) for intro, obs in zip(first_sight_strings, observation_strings)]
            else:
                action_candidate_list = list(infos["admissible_commands"])
            action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
            observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, execute_actions)]  # appending the chosen action at previous step into the observation
            previous_dynamics = current_dynamics

            if step_in_total % agent.dagger_update_per_k_game_steps == 0:
                dagger_loss = agent.update_dagger()
                if dagger_loss is not None:
                    running_avg_dagger_loss.push(dagger_loss)

            if step_no == agent.max_nb_steps_per_episode - 1:
                # terminate the game because DQN requires one extra step
                dones = [1.0 for _ in dones]

            step_in_total += 1
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # list of float
            prev_rewards = scores
            sequence_game_points.append(step_rewards)
            still_running_mask.append(still_running)
            print_actions.append(execute_actions[0] if still_running[0] else "--")

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        still_running_mask_np = np.array(still_running_mask)
        game_points_np = np.array(sequence_game_points) * still_running_mask_np  # step x batch

        # push experience into replay buffer (dagger)
        if not report:
            for b in range(batch_size):
                trajectory = []
                for i in range(len(transition_cache)):
                    observation_strings, task_strings, action_candidate_list, expert_actions, expert_indices = transition_cache[i]
                    trajectory.append([observation_strings[b], task_strings[b], action_candidate_list[b],
                                       expert_actions[b], expert_indices[b]])
                    if still_running_mask_np[i][b] == 0.0:
                        break
                agent.dagger_memory.push(trajectory)

        for b in range(batch_size):
            if report:
                running_avg_student_points.push(np.sum(game_points_np, 0)[b])
                running_avg_student_steps.push(np.sum(still_running_mask_np, 0)[b])
            else:
                running_avg_game_points.push(np.sum(game_points_np, 0)[b])
                running_avg_game_steps.push(np.sum(still_running_mask_np, 0)[b])

        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size

        if not report:
            continue
        time_2 = datetime.datetime.now()
        time_spent_seconds = (time_2-time_1).seconds
        eps_per_sec = float(episode_no) / time_spent_seconds
        avg_step_time = np.mean(np.array(step_time))
        print("Model: {:s} | Episode: {:3d} | {:s} | time spent: {:s} | eps/sec : {:2.3f} | avg step time: {:2.10f} | loss: {:2.3f} | game points: {:2.3f} | used steps: {:2.3f} | student points: {:2.3f} | student steps: {:2.3f} | fraction assist: {:2.3f} | fraction random: {:2.3f}".format(agent.experiment_tag, episode_no, game_names[0], str(time_2 - time_1).rsplit(".")[0], eps_per_sec, avg_step_time, running_avg_dagger_loss.get_avg(), running_avg_game_points.get_avg(), running_avg_game_steps.get_avg(), running_avg_student_points.get_avg(), running_avg_student_steps.get_avg(), agent.fraction_assist, agent.fraction_random))
        # print(game_id + ":    " + " | ".join(print_actions))
        print(" | ".join(print_actions))

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
            if running_avg_student_points.get_avg() >= best_performance_so_far:
                best_performance_so_far = running_avg_student_points.get_avg()
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + ".pt")

        # plot using visdom
        if config["general"]["visdom"]:
            viz_game_points.append(running_avg_game_points.get_avg())
            viz_game_step.append(running_avg_game_steps.get_avg())
            viz_student_points.append(running_avg_student_points.get_avg())
            viz_student_step.append(running_avg_student_steps.get_avg())
            viz_loss.append(running_avg_dagger_loss.get_avg())
            viz_id_eval_game_points.append(id_eval_game_points)
            viz_id_eval_step.append(id_eval_game_step)
            viz_ood_eval_game_points.append(ood_eval_game_points)
            viz_ood_eval_step.append(ood_eval_game_step)
            viz_x = np.arange(len(viz_game_points)).tolist()

            if reward_win is None:
                reward_win = viz.line(X=viz_x, Y=viz_game_points,
                            opts=dict(title=agent.experiment_tag + "_game_points"),
                        name="game points")
                viz.line(X=viz_x, Y=viz_student_points,
                            opts=dict(title=agent.experiment_tag + "_student_points"),
                            win=reward_win, update='append', name="student points")
                viz.line(X=viz_x, Y=viz_id_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_id_eval_game_points"),
                            win=reward_win, update='append', name="id eval game points")
                viz.line(X=viz_x, Y=viz_ood_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_ood_eval_game_points"),
                            win=reward_win, update='append', name="ood eval game points")
            else:
                viz.line(X=[len(viz_game_points) - 1], Y=[viz_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_game_points"),
                            win=reward_win,
                            update='append', name="game points")
                viz.line(X=[len(viz_student_points) - 1], Y=[viz_student_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_student_points"),
                            win=reward_win,
                            update='append', name="student points")
                viz.line(X=[len(viz_id_eval_game_points) - 1], Y=[viz_id_eval_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_id_eval_game_points"),
                            win=reward_win,
                            update='append', name="id eval game points")
                viz.line(X=[len(viz_ood_eval_game_points) - 1], Y=[viz_ood_eval_game_points[-1]],
                            opts=dict(title=agent.experiment_tag + "_ood_eval_game_points"),
                            win=reward_win,
                            update='append', name="ood eval game points")

            if step_win is None:
                step_win = viz.line(X=viz_x, Y=viz_game_step,
                                    opts=dict(title=agent.experiment_tag + "_game_step"),
                                    name="game step")
                viz.line(X=viz_x, Y=viz_student_step,
                            opts=dict(title=agent.experiment_tag + "_student_step"),
                            win=step_win, update='append', name="student step")
                viz.line(X=viz_x, Y=viz_id_eval_step,
                            opts=dict(title=agent.experiment_tag + "_id_eval_step"),
                            win=step_win, update='append', name="id eval step")
                viz.line(X=viz_x, Y=viz_ood_eval_step,
                            opts=dict(title=agent.experiment_tag + "_ood_eval_step"),
                            win=step_win, update='append', name="ood eval step")
            else:
                viz.line(X=[len(viz_game_step) - 1], Y=[viz_game_step[-1]],
                            opts=dict(title=agent.experiment_tag + "_game_step"),
                            win=step_win,
                            update='append', name="game step")
                viz.line(X=[len(viz_student_step) - 1], Y=[viz_student_step[-1]],
                            opts=dict(title=agent.experiment_tag + "_student_step"),
                            win=step_win,
                            update='append', name="student step")
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
                         "time spent seconds":  time_spent_seconds,
                         "episodes": episode_no,
                         "episodes per second": eps_per_sec,
                         "loss": str(running_avg_dagger_loss.get_avg()),
                         "train game points": str(running_avg_game_points.get_avg()),
                         "train game steps": str(running_avg_game_steps.get_avg()),
                         "train student points": str(running_avg_student_points.get_avg()),
                         "train student steps": str(running_avg_student_steps.get_avg()),
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
