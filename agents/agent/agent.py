import os
import random
import copy
import codecs
import spacy
import operator
import logging
from os.path import join as pjoin
from queue import PriorityQueue

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

import modules.memory as memory
from modules.model import Policy
from modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode
from modules.layers import NegativeLogLoss, masked_mean, compute_mask, GetGenerationQValue


class ObservationPool(object):

    def __init__(self, capacity=1):
        if capacity == 0:
            self.capacity = 1
            self.disable_observation = True
        elif capacity > 0:
            self.capacity = capacity
            self.disable_observation = False
        else:
            raise NotImplementedError

    def identical_with_history(self, new_stuff, list_of_old_stuff):
        for i in range(len(list_of_old_stuff)):
            if new_stuff == list_of_old_stuff[i]:
                return True
        return False

    def push_batch(self, stuff):
        assert len(stuff) == len(self.memory)
        for i in range(len(stuff)):
            self.push_one(i, stuff[i])

    def push_one(self, which, stuff):
        assert which < len(self.memory)
        if len(self.memory[which]) == 0 and stuff.endswith("restart"):
            return
        if self.disable_observation:
            if "[SEP]" in stuff:
                action = stuff.split("[SEP]", 1)[-1].strip()
                self.memory[which].append(action)
        else:
            if "Nothing happens" in stuff:
                return
            if not self.identical_with_history(stuff, self.memory[which]):
                self.memory[which].append(stuff)
        if len(self.memory[which]) > self.capacity:
            self.memory[which] = self.memory[which][-self.capacity:]

    def push_first_sight(self, stuff):
        assert len(stuff) == len(self.memory)
        for i in range(len(stuff)):
            self.first_sight.append(stuff[i])

    def get(self, which=None):
        if which is not None:
            assert which < len(self.memory)
            output = [self.first_sight[which]]
            for idx in range(len(self.memory[which])):
                output.append(self.memory[which][idx])
            return " [SEP] ".join(output)

        output = []
        for i in range(len(self.memory)):
            output.append(self.get(which=i))
        return output

    def reset(self, batch_size):
        self.memory = []
        self.first_sight = []
        for _ in range(batch_size):
            self.memory.append([])

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, config):
        self.mode = "train"
        self.config = config
        print(self.config)
        self.load_config()

        # bert tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        self.word2id = self.tokenizer.get_vocab()
        self.word_vocab = {value:key for key, value in self.word2id.items()}
        bert_model = DistilBertModel.from_pretrained('distilbert-base-cased')
        bert_model.transformer = None
        bert_model.encoder = None
        for param in bert_model.parameters():
            param.requires_grad = False

        self.online_net = Policy(config=self.config, bert_model=bert_model, word_vocab_size=len(self.word2id))
        self.target_net = Policy(config=self.config, bert_model=bert_model, word_vocab_size=len(self.word2id))
        self.online_net.train()
        self.target_net.train()
        self.update_target_net()
        for param in self.target_net.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.online_net.cuda()
            self.target_net.cuda()

        # optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config['general']['training']['optimizer']['learning_rate'])
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']

        # losses
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def load_config(self):
        self.task = self.config['general']['task']
        self.philly = self.config['general']['philly']
        self.observation_pool_capacity = self.config['general']['observation_pool_capacity']
        self.observation_pool = ObservationPool(capacity=self.observation_pool_capacity)
        self.hide_init_receptacles = self.config['general']['hide_init_receptacles']
        self.training_method = self.config['general']['training_method']

        self.init_learning_rate = self.config['general']['training']['optimizer']['learning_rate']
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']
        self.batch_size = self.config['general']['training']['batch_size']
        self.max_episode = self.config['general']['training']['max_episode']
        self.smoothing_eps = self.config['general']['training']['smoothing_eps']

        self.run_eval = self.config['general']['evaluate']['run_eval']
        self.eval_batch_size = self.config['general']['evaluate']['batch_size']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.experiment_tag = self.config['general']['checkpoint']['experiment_tag']
        self.report_frequency = self.config['general']['checkpoint']['report_frequency']
        self.load_pretrained = self.config['general']['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['general']['checkpoint']['load_from_tag']

        self.recurrent = self.config['general']['model']['recurrent']

        # RL specific
        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['rl']['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['rl']['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['rl']['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.epsilon_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes, initial_p=self.epsilon_anneal_from, final_p=self.epsilon_anneal_to)
        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0
        # replay buffer and updates
        self.accumulate_reward_from_final = self.config['rl']['replay']['accumulate_reward_from_final']
        self.discount_gamma_game_reward = self.config['rl']['replay']['discount_gamma_game_reward']
        self.discount_gamma_count_reward = self.config['rl']['replay']['discount_gamma_count_reward']
        self.discount_gamma_novel_object_reward = self.config['rl']['replay']['discount_gamma_novel_object_reward']
        self.replay_batch_size = self.config['rl']['replay']['replay_batch_size']
        self.dqn_memory = memory.PrioritizedReplayMemory(self.config['rl']['replay']['replay_memory_capacity'],
                                                         priority_fraction=self.config['rl']['replay']['replay_memory_priority_fraction'],
                                                         discount_gamma_game_reward=self.discount_gamma_game_reward,
                                                         discount_gamma_count_reward=self.discount_gamma_count_reward,
                                                         discount_gamma_novel_object_reward=self.discount_gamma_novel_object_reward,
                                                         accumulate_reward_from_final=self.accumulate_reward_from_final)
        self.update_per_k_game_steps = self.config['rl']['replay']['update_per_k_game_steps']
        self.multi_step = self.config['rl']['replay']['multi_step']
        self.count_reward_lambda = self.config['rl']['replay']['count_reward_lambda']
        self.novel_object_reward_lambda = self.config['rl']['replay']['novel_object_reward_lambda']
        self.rl_replay_sample_history_length = self.config['rl']['replay']['replay_sample_history_length']
        self.rl_replay_sample_update_from = self.config['rl']['replay']['replay_sample_update_from']
        # rl train and eval
        self.learn_start_from_this_episode = self.config['rl']['training']['learn_start_from_this_episode']
        self.target_net_update_frequency = self.config['rl']['training']['target_net_update_frequency']

        # dagger
        self.fraction_assist_anneal_episodes = self.config['dagger']['fraction_assist']['fraction_assist_anneal_episodes']
        self.fraction_assist_anneal_from = self.config['dagger']['fraction_assist']['fraction_assist_anneal_from']
        self.fraction_assist_anneal_to = self.config['dagger']['fraction_assist']['fraction_assist_anneal_to']
        self.fraction_assist = self.fraction_assist_anneal_from
        self.fraction_assist_scheduler = LinearSchedule(schedule_timesteps=self.fraction_assist_anneal_episodes, initial_p=self.fraction_assist_anneal_from, final_p=self.fraction_assist_anneal_to)

        self.fraction_random_anneal_episodes = self.config['dagger']['fraction_random']['fraction_random_anneal_episodes']
        self.fraction_random_anneal_from = self.config['dagger']['fraction_random']['fraction_random_anneal_from']
        self.fraction_random_anneal_to = self.config['dagger']['fraction_random']['fraction_random_anneal_to']
        self.fraction_random = self.fraction_random_anneal_from
        self.fraction_random_scheduler = LinearSchedule(schedule_timesteps=self.fraction_random_anneal_episodes, initial_p=self.fraction_random_anneal_from, final_p=self.fraction_random_anneal_to)

        self.dagger_memory = memory.DaggerReplayMemory(self.config['dagger']['replay']['replay_memory_capacity'])
        self.dagger_update_per_k_game_steps = self.config['dagger']['replay']['update_per_k_game_steps']
        self.dagger_replay_batch_size = self.config['dagger']['replay']['replay_batch_size']
        self.dagger_replay_sample_history_length = self.config['dagger']['replay']['replay_sample_history_length']
        self.dagger_replay_sample_update_from = self.config['dagger']['replay']['replay_sample_update_from']

        if self.training_method == "dagger":
            self.max_target_length = self.config['dagger']['max_target_length']
            self.generate_top_k = self.config['dagger']['generate_top_k']
            self.beam_width = self.config['dagger']['beam_width']
            self.action_space = self.config['dagger']['action_space']
            self.max_nb_steps_per_episode = self.config['dagger']['training']['max_nb_steps_per_episode']
            self.unstick_by_beam_search = self.config['dagger']['unstick_by_beam_search']
        elif self.training_method == "dqn":
            self.max_target_length = self.config['rl']['max_target_length']
            self.generate_top_k = self.config['rl']['generate_top_k']
            self.beam_width = self.config['rl']['beam_width']
            self.action_space = self.config['rl']['action_space']
            self.max_nb_steps_per_episode = self.config['rl']['training']['max_nb_steps_per_episode']
            self.unstick_by_beam_search = None
        else:
            raise NotImplementedError

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()

    def update_target_net(self):
        if self.target_net is not None:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')

            model_dict = self.online_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
            print("The loaded parameters are:")
            keys = [key for key in pretrained_dict]
            print(", ".join(keys))
            print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def init(self, batch_size):
        self.observation_pool.reset(batch_size)

    def get_word_input(self, input_strings):
        word_id_list = [self.tokenizer.encode(item, add_special_tokens=False) for item in input_strings]
        return self.get_word_input_from_ids(word_id_list)

    def get_word_input_from_ids(self, word_id_list):
        input_word = pad_sequences(word_id_list, maxlen=max_len(word_id_list) + 3, dtype='int32')  # 3 --> see layer.DepthwiseSeparableConv.padding
        input_word = to_pt(input_word, self.use_cuda)
        return input_word

    def get_action_candidate_representations(self, action_candidate_list, use_model="online"):
        # in case there are too many candidates in certain data point, we compute their candidate representations by small batches
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        res_representations = torch.zeros(batch_size, max_num_candidate, self.online_net.block_hidden_dim)
        res_mask = torch.zeros(batch_size, max_num_candidate)
        if self.use_cuda:
            res_representations = res_representations.cuda()
            res_mask = res_mask.cuda()

        squeezed_candidate_list, from_which_original_batch = [], []
        for b in range(batch_size):
            squeezed_candidate_list += action_candidate_list[b]
            for i in range(len(action_candidate_list[b])):
                from_which_original_batch.append((b, i))

        tmp_batch_size = 64
        n_tmp_batches = (len(squeezed_candidate_list) + tmp_batch_size - 1) // tmp_batch_size
        for tmp_batch_id in range(n_tmp_batches):
            tmp_batch_cand = squeezed_candidate_list[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]  # tmp_batch of candidates
            tmp_batch_from = from_which_original_batch[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]

            tmp_batch_cand_representation_sequence, tmp_batch_cand_mask = self.encode_text(tmp_batch_cand, use_model=use_model)  # tmp_batch x num_word x hid, tmp_batch x num_word

            # masked mean the num_word dimension
            _mask = torch.sum(tmp_batch_cand_mask, -1)  # batch
            tmp_batch_cand_representation = torch.sum(tmp_batch_cand_representation_sequence, -2)  # batch x hid
            tmp = torch.eq(_mask, 0).float()
            if tmp_batch_cand_representation.is_cuda:
                tmp = tmp.cuda()
            _mask = _mask + tmp
            tmp_batch_cand_representation = tmp_batch_cand_representation / _mask.unsqueeze(-1)  # batch x hid
            tmp_batch_cand_mask = tmp_batch_cand_mask.byte().any(-1).float()  # batch

            for i in range(len(tmp_batch_from)):
                res_representations[tmp_batch_from[i][0], tmp_batch_from[i][1], :] = tmp_batch_cand_representation[i]
                res_mask[tmp_batch_from[i][0], tmp_batch_from[i][1]] = tmp_batch_cand_mask[i]

        return res_representations, res_mask

    def choose_model(self, use_model="online"):
        if use_model == "online":
            model = self.online_net
        elif use_model == "target":
            model = self.target_net
        else:
            raise NotImplementedError
        return model

    def encode_text(self, observation_strings, use_model):
        model = self.choose_model(use_model)
        input_obs = self.get_word_input(observation_strings)
        # encode
        obs_encoding_sequence, obs_mask = model.encode_text(input_obs)
        return obs_encoding_sequence, obs_mask

    def finish_of_episode(self, episode_no, batch_size):
        # fraction_assist annealing
        self.fraction_assist = self.fraction_assist_scheduler.value(episode_no)
        self.fraction_assist = max(self.fraction_assist, 0.0)

        # fraction_random annealing
        self.fraction_random = self.fraction_random_scheduler.value(episode_no)
        self.fraction_random = max(self.fraction_random, 0.0)

        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(episode_no - self.learn_start_from_this_episode)
            self.epsilon = max(self.epsilon, 0.0)

    def preprocess_task(self, task_strings):
        return [preproc(item) for item in task_strings]

    def preprocess_observation(self, observation_strings):
        res = []
        for i in range(len(observation_strings)):
            obs = observation_strings[i]
            obs = preproc(obs)
            # sort objects
            if "you see" in obs:
                # -= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a armchair 2, a diningtable 1, a diningtable 2, a coffeetable 1, a armchair 1, a tvstand 1, a sidetable 2, a garbagecan 1, a sidetable 1, a sofa 1, a drawer 1, and a drawer 2.
                before_you_see, after_you_see = obs.split("you see", 1)
                before_you_see, after_you_see = before_you_see.strip(), after_you_see.strip()
                before_you_see = " ".join([before_you_see, "you see"])
                object_list, after_period = after_you_see.split(".", 1)
                object_list, after_period = object_list.strip(), after_period.strip()
                after_period += "."
                object_list = object_list.replace(", and ", ", ")
                object_list = object_list.split(", ")
                object_list = sorted(object_list)
                if len(object_list) == 1:
                    object_string = object_list[0]
                else:
                    object_string = ", ".join(object_list[:-1]) + ", and " + object_list[-1]
                obs = " ".join([before_you_see, object_string]) + after_period
            res.append(obs)
        return res

    def preprocess_action_candidates(self, action_candidate_list):
        batch_size = len(action_candidate_list)
        preproced_action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item) for item in action_candidate_list[b]]
            preproced_action_candidate_list.append(ac)
        return preproced_action_candidate_list

    def preprocessing(self, observation_strings, action_candidate_list):
        preproced_observation_strings = self.preprocess_observation(observation_strings)
        preproced_action_candidate_list = self.preprocess_action_candidates(action_candidate_list)
        return preproced_observation_strings, preproced_action_candidate_list

    def get_task_and_obs(self, observation_strings):
        batch_size = len(observation_strings)
        task_desc_strings, no_goal_observation_strings = [], []
        for b in range(batch_size):
            task_desc = observation_strings[b].partition("Your task is to: ")[-1]
            no_goal_obs_str = observation_strings[b].replace("Your task is to: %s" % task_desc, "")
            if self.hide_init_receptacles:
                no_goal_obs_str = no_goal_obs_str.partition("Looking quickly around you, you see")[0]
            task_desc_strings.append(task_desc)
            no_goal_observation_strings.append(no_goal_obs_str)
        return task_desc_strings, no_goal_observation_strings

    def encode(self, observation_strings, use_model):
        obs_enc_seq, obs_mask = self.encode_text(observation_strings, use_model=use_model)
        return obs_enc_seq, obs_mask

    def action_scoring(self, action_candidate_list, h_obs, obs_mask, h_td, td_mask, previous_dynamics, use_model=None):
        model = self.choose_model(use_model)
        average_action_candidate_representations, action_candidate_mask = self.get_action_candidate_representations(action_candidate_list, use_model=use_model)  # batch x num_cand x hid, batch x num_cand
        aggregated_obs_representation = model.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

        if self.recurrent:
            averaged_representation = model.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
            current_dynamics = model.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else model.rnncell(averaged_representation)
        else:
            current_dynamics = None

        action_scores, action_masks = model.score_actions(average_action_candidate_representations, action_candidate_mask,
                                                          aggregated_obs_representation, obs_mask, current_dynamics)  # batch x num_actions
        return action_scores, action_masks, current_dynamics

    def beam_search_candidate_scoring(self, action_candidate_list, aggregated_obs_representation, obs_mask, current_dynamics, use_model=None):
        model = self.choose_model(use_model)
        average_action_candidate_representations, action_candidate_mask = self.get_action_candidate_representations(action_candidate_list, use_model=use_model)  # batch x num_cand x hid, batch x num_cand

        action_scores, action_masks = model.score_actions(average_action_candidate_representations, action_candidate_mask,
                                                          aggregated_obs_representation, obs_mask, current_dynamics, fix_shared_components=True)  # batch x num_actions
        return action_scores, action_masks


class DQNAgent(Agent):
    # action scoring stuff (Deep Q-Learning)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.training_method == "dqn"

    def choose_random_action(self, action_rank, action_unpadded=None):
        """
        Select an action randomly.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(len(action_unpadded[j])))
            indices = np.array(indices)
        return indices

    def choose_maxQ_action(self, action_rank, action_mask=None):
        """
        Generate an action by maximum q values.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        action_indices = torch.argmax(action_rank, -1)  # batch
        return to_np(action_indices)

    # choosing from list of admissible commands
    def admissible_commands_act_greedy(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics):
        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics

    def admissible_commands_act_random(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics):
        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, current_dynamics

    def admissible_commands_act(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics, random=False):

        with torch.no_grad():
            if self.mode == "eval":
                return self.admissible_commands_act_greedy(observation_strings, task_desc_strings, action_candidate_list, previous_dynamics)
            if random:
                return self.admissible_commands_act_random(observation_strings, task_desc_strings, action_candidate_list, previous_dynamics)
            batch_size = len(observation_strings)

            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")

            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics

    # choosing from output of beam search (without re-compute some intermediate representations)
    def beam_search_choice_act_greedy(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            action_candidate_list, current_dynamics, obs_mask, aggregated_obs_representation = self.command_generation_by_beam_search(observation_strings, task_desc_strings, previous_dynamics)
            action_scores, action_masks = self.beam_search_candidate_scoring(action_candidate_list,
                                                                             aggregated_obs_representation,
                                                                             obs_mask,
                                                                             current_dynamics,
                                                                             use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics, action_candidate_list

    def beam_search_choice_act_random(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            action_candidate_list, current_dynamics, obs_mask, aggregated_obs_representation = self.command_generation_by_beam_search(observation_strings, task_desc_strings, previous_dynamics)
            action_scores, _ = self.beam_search_candidate_scoring(action_candidate_list,
                                                                  aggregated_obs_representation,
                                                                  obs_mask,
                                                                  current_dynamics,
                                                                  use_model="online")
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, current_dynamics, action_candidate_list

    def beam_search_choice_act(self, observation_strings, task_desc_strings, previous_dynamics, random=False):
        with torch.no_grad():
            if self.mode == "eval":
                return self.beam_search_choice_act_greedy(observation_strings, task_desc_strings, previous_dynamics)
            if random:
                return self.beam_search_choice_act_random(observation_strings, task_desc_strings, previous_dynamics)
            batch_size = len(observation_strings)
            action_candidate_list, current_dynamics, obs_mask, aggregated_obs_representation = self.command_generation_by_beam_search(observation_strings, task_desc_strings, previous_dynamics)
            action_scores, action_masks = self.beam_search_candidate_scoring(action_candidate_list,
                                                                             aggregated_obs_representation,
                                                                             obs_mask,
                                                                             current_dynamics,
                                                                             use_model="online")

            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics, action_candidate_list

    # generating token by token
    def command_generation_by_beam_search(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_strings)
            beam_width = self.beam_width
            generate_top_k = self.generate_top_k
            chosen_actions = []

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):
                # starts from CLS tokens
                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = input_obs[b: b + 1]  # 1 x obs_len
                __obs_mask = obs_mask[b: b + 1]  # 1 x obs_len
                __aggregated_obs_representation = aggregated_obs_representation[b: b + 1]  # 1 x obs_len x hid
                if current_dynamics is not None:
                    __current_dynamics = current_dynamics[b: b + 1]  # 1 x hid
                else:
                    __current_dynamics = None
                ended_nodes = []
                
                # starting node -  previous node, input target, logp, length
                node = BeamSearchNode(None, __input_target_list, 0, 1)
                nodes_queue = PriorityQueue()
                # start the queue
                nodes_queue.put((node.val, node))
                queue_size = 1
                while(True):
                    # give up when decoding takes too long
                    if queue_size > 2000:
                        break
                    # fetch the best node
                    score, n = nodes_queue.get()
                    __input_target_list = n.input_target

                    if (n.input_target[-1] == self.word2id["[SEP]"] or n.length >= self.max_target_length) and n.previous_node != None:
                        ended_nodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(ended_nodes) >= generate_top_k:
                            break
                        else:
                            continue

                    input_target = pad_sequences([__input_target_list], dtype='int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)
                    # decode for one step using decoder
                    pred = self.online_net.decode(input_target, target_mask, __aggregated_obs_representation, __obs_mask, __current_dynamics, __input_obs)  # 1 x target_length x vocab
                    pred = pred[0][-1].cpu()
                    gt_zero = torch.gt(pred, 0.0).float()  # vocab
                    epsilon = torch.le(pred, 0.0).float() * 1e-8  # vocab
                    log_pred = torch.log(pred + epsilon) * gt_zero  # vocab

                    top_beam_width_log_probs, top_beam_width_indicies = torch.topk(log_pred, beam_width)
                    next_nodes = []

                    for new_k in range(beam_width):
                        pos = top_beam_width_indicies[new_k]
                        log_p = top_beam_width_log_probs[new_k].item()
                        node = BeamSearchNode(n, __input_target_list + [pos], n.log_prob + log_p, n.length + 1)
                        next_nodes.append((node.val, node))

                    # put them into queue
                    for i in range(len(next_nodes)):
                        score, nn = next_nodes[i]
                        nodes_queue.put((score, nn))
                    # increase qsize
                    queue_size += len(next_nodes) - 1

                # choose n best paths
                if len(ended_nodes) == 0:
                    ended_nodes = [nodes_queue.get() for _ in range(generate_top_k)]

                utterances = []
                for score, n in sorted(ended_nodes, key=operator.itemgetter(0)):
                    utte = n.input_target
                    utte_string = self.tokenizer.decode(utte)
                    utterances.append(utte_string)

                utterances = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in utterances]
                utterances = [item.replace(" in / on ", " in/on " ) for item in utterances]
                chosen_actions.append(utterances)
            return chosen_actions, current_dynamics, obs_mask, aggregated_obs_representation

    def command_generation_act_greedy(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_strings)

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            # greedy generation
            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)  # batch x target_length x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            chosen_actions = [self.tokenizer.decode(item) for item in input_target_list]
            chosen_actions = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in chosen_actions]
            chosen_actions = [item.replace(" in / on ", " in/on " ) for item in chosen_actions]
            chosen_indices = [item[1:] for item in input_target_list]
            for i in range(len(chosen_indices)):
                if chosen_indices[i][-1] == self.word2id["[SEP]"]:
                    chosen_indices[i] = chosen_indices[i][:-1]
            return chosen_actions, chosen_indices, current_dynamics

    def command_generation_act_random(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():

            batch_size = len(observation_strings)
            beam_width = self.beam_width
            generate_top_k = self.generate_top_k
            chosen_actions, chosen_indices = [], []

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):
                # starts from CLS tokens
                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = input_obs[b: b + 1]  # 1 x obs_len
                __obs_mask = obs_mask[b: b + 1]  # 1 x obs_len
                __aggregated_obs_representation = aggregated_obs_representation[b: b + 1]  # 1 x obs_len x hid
                if current_dynamics is not None:
                    __current_dynamics = current_dynamics[b: b + 1]  # 1 x hid
                else:
                    __current_dynamics = None
                ended_nodes = []
                # starting node -  previous node, input target, logp, length
                node = BeamSearchNode(None, __input_target_list, 0, 1)
                nodes_queue = PriorityQueue()
                # start the queue
                nodes_queue.put((node.val, node))
                queue_size = 1

                while(True):
                    # give up when decoding takes too long
                    if queue_size > 2000:
                        break
                    # fetch the best node
                    score, n = nodes_queue.get()
                    __input_target_list = n.input_target

                    if (n.input_target[-1] == self.word2id["[SEP]"] or n.length >= self.max_target_length) and n.previous_node != None:
                        ended_nodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(ended_nodes) >= generate_top_k:
                            break
                        else:
                            continue

                    input_target = pad_sequences([__input_target_list], dtype='int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)
                    # decode for one step using decoder
                    pred = self.online_net.decode(input_target, target_mask, __aggregated_obs_representation, __obs_mask, __current_dynamics, __input_obs)  # 1 x target_length x vocab
                    pred = pred[0][-1].cpu()
                    gt_zero = torch.gt(pred, 0.0).float()  # vocab
                    epsilon = torch.le(pred, 0.0).float() * 1e-8  # vocab
                    log_pred = torch.log(pred + epsilon) * gt_zero  # vocab

                    top_beam_width_log_probs, top_beam_width_indicies = torch.topk(log_pred, beam_width)
                    next_nodes = []

                    for new_k in range(beam_width):
                        pos = top_beam_width_indicies[new_k]
                        log_p = top_beam_width_log_probs[new_k].item()
                        node = BeamSearchNode(n, __input_target_list + [pos], n.log_prob + log_p, n.length + 1)
                        next_nodes.append((node.val, node))

                    # put them into queue
                    for i in range(len(next_nodes)):
                        score, nn = next_nodes[i]
                        nodes_queue.put((score, nn))
                    # increase qsize
                    queue_size += len(next_nodes) - 1

                # choose n best paths
                if len(ended_nodes) == 0:
                    ended_nodes = [nodes_queue.get() for _ in range(generate_top_k)]

                indicies, utterances = [], []
                for score, n in sorted(ended_nodes, key=operator.itemgetter(0)):
                    utte = n.input_target
                    utte_string = self.tokenizer.decode(utte)
                    utterances.append(utte_string)
                    indicies.append(utte)

                utterances = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in utterances]
                utterances = [item.replace(" in / on ", " in/on " ) for item in utterances]
                indicies = [item[1:] for item in indicies]
                for i in range(len(indicies)):
                    if indicies[i][-1] == self.word2id["[SEP]"]:
                        indicies[i] = indicies[i][:-1]

                # sample one from all generated beams
                rand_idx = np.random.choice(len(indicies))
                chosen_actions.append(utterances[rand_idx])
                chosen_indices.append(indicies[rand_idx])
            return chosen_actions, chosen_indices, current_dynamics

    def command_generation_act(self, observation_strings, task_desc_strings, previous_dynamics, random=False):
        with torch.no_grad():
            if self.mode == "eval":
                return self.command_generation_act_greedy(observation_strings, task_desc_strings, previous_dynamics)
            if random:
                return self.command_generation_act_random(observation_strings, task_desc_strings, previous_dynamics)
            batch_size = len(observation_strings)
            greedy_actions, greedy_indices, greedy_current_dynamics = self.command_generation_act_greedy(observation_strings, task_desc_strings, previous_dynamics)
            
            # random number for epsilon greedy
            chosen_actions, chosen_indices, current_dynamics = [], [], []
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            for b in range(batch_size):
                if rand_num[b] < self.epsilon:
                    # random
                    random_actions, random_indices, random_current_dynamics = self.command_generation_act_random(observation_strings[b: b + 1], task_desc_strings[b: b + 1], None if previous_dynamics is None else previous_dynamics[b: b + 1])
                    chosen_actions.append(random_actions[0])
                    chosen_indices.append(random_indices[0])
                    if self.recurrent:
                        current_dynamics.append(random_current_dynamics[0])
                else:
                    # greedy
                    chosen_actions.append(greedy_actions[b])
                    chosen_indices.append(greedy_indices[b])
                    if self.recurrent:
                        current_dynamics.append(greedy_current_dynamics[b])
            current_dynamics = torch.stack(current_dynamics, 0) if self.recurrent else None  # batch x hidden
            return chosen_actions, chosen_indices, current_dynamics

    # update: admissible commands
    def get_dqn_loss_admissible_commands(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.get_batch(self.replay_batch_size, multi_step=self.multi_step)
        if data is None:
            return None, None

        obs_list, task_list, candidate_list, action_indices, rewards, next_obs_list, next_candidate_list, actual_ns = data
        if self.use_cuda:
            rewards = rewards.cuda()

        h_obs, obs_mask = self.encode(obs_list, use_model="online")
        h_td, td_mask = self.encode(task_list, use_model="online")
        action_scores, _, _ = self.action_scoring(candidate_list,
                                                  h_obs, obs_mask,
                                                  h_td, td_mask,
                                                  None,
                                                  use_model="online")

        # ps_a
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="online")
            next_action_scores, next_action_masks, _ = self.action_scoring(next_candidate_list,
                                                                           h_obs, obs_mask,
                                                                           h_td.detach(), td_mask.detach(),
                                                                           None,
                                                                           use_model="online")

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="target")
            h_td_t, td_mask_t = self.encode(task_list, use_model="target")
            next_action_scores, _, _ = self.action_scoring(next_candidate_list,
                                                           h_obs, obs_mask,
                                                           h_td_t, td_mask_t,
                                                           None,
                                                           use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value

    def get_drqn_loss_admissible_commands(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.dqn_memory.get_batch_of_sequences(self.replay_batch_size, sample_history_length=self.rl_replay_sample_history_length)
        if data is None:
            return None, None

        seq_obs, task, seq_candidates, seq_chosen_indices, seq_reward, seq_next_obs, seq_next_candidates = data
        loss_list, q_value_list = [], []
        prev_dynamics = None

        h_td, td_mask = self.encode(task, use_model="online")
        with torch.no_grad():
            h_td_t, td_mask_t = self.encode(task, use_model="target")

        for step_no in range(self.rl_replay_sample_history_length):
            obs, candidates, chosen_indices, reward, next_obs, next_candidates = seq_obs[step_no], seq_candidates[step_no], seq_chosen_indices[step_no], seq_reward[step_no], seq_next_obs[step_no], seq_next_candidates[step_no]
            if self.use_cuda:
                reward = reward.cuda()

            h_obs, obs_mask = self.encode(obs, use_model="online")
            action_scores, _, current_dynamics = self.action_scoring(candidates, h_obs, obs_mask, h_td, td_mask,
                                                                     prev_dynamics, use_model="online")
            # ps_a
            chosen_indices = to_pt(chosen_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            q_value = ez_gather_dim_1(action_scores, chosen_indices).squeeze(1)  # batch

            prev_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.rl_replay_sample_update_from:
                q_value = q_value.detach()
                prev_dynamics = prev_dynamics.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)

                h_obs, obs_mask = self.encode(next_obs, use_model="online")
                next_action_scores, next_action_masks, _ = self.action_scoring(next_candidates, h_obs, obs_mask, h_td, td_mask,
                                                                               prev_dynamics, use_model="online")

                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
                next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)

                # pns # Probabilities p(s_t+n, ·; θtarget)
                h_obs, obs_mask = self.encode(next_obs, use_model="target")
                next_action_scores, _, _ = self.action_scoring(next_candidates, h_obs, obs_mask, h_td_t, td_mask_t,
                                                               prev_dynamics, use_model="target")

                # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
                
            reward = reward + next_q_value * self.discount_gamma_game_reward  # batch
            loss = F.smooth_l1_loss(q_value, reward)  # 1
            loss_list.append(loss)
            q_value_list.append(q_value)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()

        return loss, q_value

    def update_dqn_admissible_commands(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            dqn_loss, q_value = self.get_drqn_loss_admissible_commands()
        else:
            dqn_loss, q_value = self.get_dqn_loss_admissible_commands()

        if dqn_loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    # update: beam search choice
    def get_dqn_loss_beam_search_choice(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.get_batch(self.replay_batch_size, multi_step=self.multi_step)
        if data is None:
            return None, None

        obs_list, task_list, candidate_list, action_indices, rewards, next_obs_list, next_candidate_list, actual_ns = data
        if self.use_cuda:
            rewards = rewards.cuda()

        with torch.no_grad():
            h_obs, obs_mask = self.encode(obs_list, use_model="online")
            h_td, td_mask = self.encode(task_list, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
        action_scores, _ = self.beam_search_candidate_scoring(candidate_list, aggregated_obs_representation, obs_mask, None, use_model="online")

        # ps_a
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
            next_action_scores, next_action_masks = self.beam_search_candidate_scoring(next_candidate_list, aggregated_obs_representation, obs_mask, None, use_model="online")

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="target")
            h_td_t, td_mask_t = self.encode(task_list, use_model="target")
            aggregated_obs_representation = self.target_net.aggretate_information(h_obs, obs_mask, h_td_t, td_mask_t)  # batch x obs_length x hid
            next_action_scores, _ = self.beam_search_candidate_scoring(next_candidate_list, aggregated_obs_representation, obs_mask, None, use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value

    def get_drqn_loss_beam_search_choice(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.dqn_memory.get_batch_of_sequences(self.replay_batch_size, sample_history_length=self.rl_replay_sample_history_length)
        if data is None:
            return None, None

        seq_obs, task, seq_candidates, seq_chosen_indices, seq_reward, seq_next_obs, seq_next_candidates = data
        loss_list, q_value_list = [], []
        prev_dynamics = None

        with torch.no_grad():
            h_td, td_mask = self.encode(task, use_model="online")
            h_td_t, td_mask_t = self.encode(task, use_model="target")

        for step_no in range(self.rl_replay_sample_history_length):
            obs, candidates, chosen_indices, reward, next_obs, next_candidates = seq_obs[step_no], seq_candidates[step_no], seq_chosen_indices[step_no], seq_reward[step_no], seq_next_obs[step_no], seq_next_candidates[step_no]
            if self.use_cuda:
                reward = reward.cuda()

            with torch.no_grad():
                h_obs, obs_mask = self.encode(obs, use_model="online")
                aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, prev_dynamics) if prev_dynamics is not None else self.online_net.rnncell(averaged_representation)
            action_scores, _ = self.beam_search_candidate_scoring(candidates, aggregated_obs_representation, obs_mask, current_dynamics, use_model="online")
            # ps_a
            chosen_indices = to_pt(chosen_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            q_value = ez_gather_dim_1(action_scores, chosen_indices).squeeze(1)  # batch

            prev_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.rl_replay_sample_update_from:
                q_value = q_value.detach()
                prev_dynamics = prev_dynamics.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)

                h_obs, obs_mask = self.encode(next_obs, use_model="online")
                aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                next_dynamics = self.online_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.online_net.rnncell(averaged_representation)
                next_action_scores, next_action_masks = self.beam_search_candidate_scoring(next_candidates, aggregated_obs_representation, obs_mask, next_dynamics, use_model="online")

                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
                next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)

                # pns # Probabilities p(s_t+n, ·; θtarget)
                h_obs, obs_mask = self.encode(next_obs, use_model="target")
                aggregated_obs_representation = self.target_net.aggretate_information(h_obs, obs_mask, h_td_t, td_mask_t)  # batch x obs_length x hid
                averaged_representation = self.target_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                next_dynamics = self.target_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.target_net.rnncell(averaged_representation)
                next_action_scores, _ = self.beam_search_candidate_scoring(next_candidates, aggregated_obs_representation, obs_mask, next_dynamics, use_model="target")

                # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
                
            reward = reward + next_q_value * self.discount_gamma_game_reward  # batch
            loss = F.smooth_l1_loss(q_value, reward)  # 1
            loss_list.append(loss)
            q_value_list.append(q_value)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()

        return loss, q_value

    def update_dqn_beam_search_choice(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            dqn_loss, q_value = self.get_drqn_loss_beam_search_choice()
        else:
            dqn_loss, q_value = self.get_dqn_loss_beam_search_choice()

        if dqn_loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    # update: command generation
    def get_dqn_loss_command_generation(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.get_batch(self.replay_batch_size, multi_step=self.multi_step)
        if data is None:
            return None, None

        observation_strings, task_desc_strings, _, action_indices, rewards, next_observation_strings, _, actual_ns = data
        batch_size = len(observation_strings)
        if self.use_cuda:
            rewards = rewards.cuda()

        input_target = [[self.word2id["[CLS]"]] + item for item in action_indices]
        ground_truth = [item + [self.word2id["[SEP]"]] for item in action_indices]
        input_target = self.get_word_input_from_ids(input_target)
        ground_truth = self.get_word_input_from_ids(ground_truth)

        input_obs = self.get_word_input(observation_strings)
        next_input_obs = self.get_word_input(next_observation_strings)
        h_obs, obs_mask = self.encode(observation_strings, use_model="online")
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")
        aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

        target_mask = compute_mask(input_target)  # mask of ground truth should be the same
        pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, None, input_obs)  # batch x target_length x vocab
        q_value = GetGenerationQValue(pred * target_mask.unsqueeze(-1), ground_truth, target_mask)

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="online")
            next_aggregated_obs_representation = self.online_net.aggretate_information(next_h_obs, next_obs_mask, h_td, td_mask)  # batch x obs_length x hid

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            # greedy generation
            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):
                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                pred = self.online_net.decode(input_target, target_mask, next_aggregated_obs_representation, next_obs_mask, None, next_input_obs)  # batch x target_length x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            chosen_indices = [item[1:] for item in input_target_list]
            for i in range(len(chosen_indices)):
                if chosen_indices[i][-1] == self.word2id["[SEP]"]:
                    chosen_indices[i] = chosen_indices[i][:-1]

            # pns # Probabilities p(s_t+n, ·; θtarget)
            next_input_target = [[self.word2id["[CLS]"]] + item for item in chosen_indices]
            next_ground_truth = [item + [self.word2id["[SEP]"]] for item in chosen_indices]
            next_input_target = self.get_word_input_from_ids(next_input_target)
            next_ground_truth = self.get_word_input_from_ids(next_ground_truth)

            next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="target")
            next_h_td, next_td_mask = self.encode(task_desc_strings, use_model="target")
            next_aggregated_obs_representation = self.target_net.aggretate_information(next_h_obs, next_obs_mask, next_h_td, next_td_mask)  # batch x obs_length x hid

            next_target_mask = compute_mask(next_input_target)  # mask of ground truth should be the same
            next_pred = self.target_net.decode(next_input_target, next_target_mask, next_aggregated_obs_representation, next_obs_mask, None, next_input_obs)  # batch x target_length x vocab
            next_q_value = GetGenerationQValue(next_pred * next_target_mask.unsqueeze(-1), next_ground_truth, next_target_mask)  # batch

            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value

    def get_drqn_loss_command_generation(self):
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.dqn_memory.get_batch_of_sequences(self.replay_batch_size, sample_history_length=self.rl_replay_sample_history_length)
        if data is None:
            return None, None

        seq_obs, task_desc_strings, _, seq_chosen_indices, seq_reward, seq_next_obs, _ = data
        batch_size = len(seq_obs[0])
        loss_list, q_value_list = [], []
        previous_dynamics = None

        h_td, td_mask = self.encode(task_desc_strings, use_model="online")
        with torch.no_grad():
            h_td_t, td_mask_t = self.encode(task_desc_strings, use_model="target")

        for step_no in range(self.rl_replay_sample_history_length):
            observation_strings, action_indices, reward, next_observation_strings = seq_obs[step_no], seq_chosen_indices[step_no], seq_reward[step_no], seq_next_obs[step_no]
            if self.use_cuda:
                reward = reward.cuda()

            input_target = [[self.word2id["[CLS]"]] + item for item in action_indices]
            ground_truth = [item + [self.word2id["[SEP]"]] for item in action_indices]
            input_target = self.get_word_input_from_ids(input_target)
            ground_truth = self.get_word_input_from_ids(ground_truth)

            input_obs = self.get_word_input(observation_strings)
            next_input_obs = self.get_word_input(next_observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
            current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)

            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)  # batch x target_length x vocab
            q_value = GetGenerationQValue(pred * target_mask.unsqueeze(-1), ground_truth, target_mask)

            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.rl_replay_sample_update_from:
                q_value = q_value.detach()
                previous_dynamics = previous_dynamics.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)
                next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="online")
                next_aggregated_obs_representation = self.online_net.aggretate_information(next_h_obs, next_obs_mask, h_td, td_mask)  # batch x obs_length x hid
                next_averaged_representation = self.online_net.masked_mean(next_aggregated_obs_representation, next_obs_mask)  # batch x hid
                next_dynamics = self.online_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.online_net.rnncell(next_averaged_representation)

                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                # greedy generation
                input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
                eos = np.zeros(batch_size)
                for _ in range(self.max_target_length):
                    input_target = copy.deepcopy(input_target_list)
                    input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                    pred = self.online_net.decode(input_target, target_mask, next_aggregated_obs_representation, next_obs_mask, next_dynamics, next_input_obs)  # batch x target_length x vocab
                    # pointer softmax
                    pred = to_np(pred[:, -1])  # batch x vocab
                    pred = np.argmax(pred, -1)  # batch
                    for b in range(batch_size):
                        new_stuff = [pred[b]] if eos[b] == 0 else []
                        input_target_list[b] = input_target_list[b] + new_stuff
                        if pred[b] == self.word2id["[SEP]"]:
                            eos[b] = 1
                    if np.sum(eos) == batch_size:
                        break
                chosen_indices = [item[1:] for item in input_target_list]
                for i in range(len(chosen_indices)):
                    if chosen_indices[i][-1] == self.word2id["[SEP]"]:
                        chosen_indices[i] = chosen_indices[i][:-1]

                # pns # Probabilities p(s_t+n, ·; θtarget)
                next_input_target = [[self.word2id["[CLS]"]] + item for item in chosen_indices]
                next_ground_truth = [item + [self.word2id["[SEP]"]] for item in chosen_indices]
                next_input_target = self.get_word_input_from_ids(next_input_target)
                next_ground_truth = self.get_word_input_from_ids(next_ground_truth)

                next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="target")
                next_aggregated_obs_representation = self.target_net.aggretate_information(next_h_obs, next_obs_mask, h_td_t, td_mask_t)  # batch x obs_length x hid
                next_averaged_representation = self.target_net.masked_mean(next_aggregated_obs_representation, next_obs_mask)  # batch x hid
                next_dynamics = self.target_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.target_net.rnncell(next_averaged_representation)

                next_target_mask = compute_mask(next_input_target)  # mask of ground truth should be the same
                next_pred = self.target_net.decode(next_input_target, next_target_mask, next_aggregated_obs_representation, next_obs_mask, next_dynamics, next_input_obs)  # batch x target_length x vocab
                next_q_value = GetGenerationQValue(next_pred * next_target_mask.unsqueeze(-1), next_ground_truth, next_target_mask)  # batch

            reward = reward + next_q_value * self.discount_gamma_game_reward  # batch
            loss = F.smooth_l1_loss(q_value, reward)  # 1
            loss_list.append(loss)
            q_value_list.append(q_value)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()
        return loss, q_value

    def update_dqn_command_generation(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            dqn_loss, q_value = self.get_drqn_loss_command_generation()
        else:
            dqn_loss, q_value = self.get_dqn_loss_command_generation()

        if dqn_loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    def update_dqn(self):
        if self.action_space == "generation":
            return self.update_dqn_command_generation()
        elif self.action_space == "beam_search_choice":
            return self.update_dqn_beam_search_choice()
        elif self.action_space in ["admissible", "exhaustive"]:
            return self.update_dqn_admissible_commands()
        else:
            raise NotImplementedError()


class DAggerAgent(Agent):
    # dagger stuff

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.training_method == "dagger"

    def choose_softmax_action(self, action_rank, action_mask=None):
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        pred_softmax = torch.log_softmax(action_rank, dim=1)
        action_indices = torch.argmax(pred_softmax, -1)  # batch
        return pred_softmax, to_np(action_indices)

    def update_dagger(self):
        if self.recurrent:
            return self.train_dagger_recurrent()
        else:
            return self.train_dagger()

    # without recurrency
    def train_dagger(self):

        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        transitions = self.dagger_memory.sample(self.dagger_replay_batch_size)
        if transitions is None:
            return None
        batch = memory.dagger_transition(*zip(*transitions))

        if self.action_space == "generation":
            return self.command_generation_teacher_force(batch.observation_list, batch.task_list, batch.target_list)
        elif self.action_space in ["admissible", "exhaustive"]:
            return self.admissible_commands_teacher_force(batch.observation_list, batch.task_list, batch.action_candidate_list, batch.target_indices)
        else:
            raise NotImplementedError()

    def admissible_commands_teacher_force(self, observation_strings, task_desc_strings, action_candidate_list, target_indices):
        expert_indicies = to_pt(np.array(target_indices), enable_cuda=self.use_cuda, type='long')

        h_obs, obs_mask = self.encode(observation_strings, use_model="online")
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")

        action_scores, _, _ = self.action_scoring(action_candidate_list,
                                                  h_obs, obs_mask,
                                                  h_td, td_mask,
                                                  None,
                                                  use_model="online")

        # softmax and cross-entropy
        loss = self.cross_entropy_loss(action_scores, expert_indicies)
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(loss)

    def command_generation_teacher_force(self, observation_strings, task_desc_strings, target_strings):
        input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in target_strings]
        output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in target_strings]

        input_obs = self.get_word_input(observation_strings)
        h_obs, obs_mask = self.encode(observation_strings, use_model="online")
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")
        
        aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

        input_target = self.get_word_input(input_target_strings)
        ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
        target_mask = compute_mask(input_target)  # mask of ground truth should be the same
        pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, None, input_obs)  # batch x target_length x vocab

        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
        loss = torch.mean(batch_loss)

        if loss is None:
            return None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(loss)

    # with recurrency
    def train_dagger_recurrent(self):
        
        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        sequence_of_transitions, contains_first_step = self.dagger_memory.sample_sequence(self.dagger_replay_batch_size, self.dagger_replay_sample_history_length)
        if sequence_of_transitions is None:
            return None

        batches = []
        for transitions in sequence_of_transitions:
            batch = memory.dagger_transition(*zip(*transitions))
            batches.append(batch)

        if self.action_space == "generation":
            return self.command_generation_recurrent_teacher_force([batch.observation_list for batch in batches], [batch.task_list for batch in batches], [batch.target_list for batch in batches], contains_first_step)
        elif self.action_space in ["admissible", "exhaustive"]:
            return self.admissible_commands_recurrent_teacher_force([batch.observation_list for batch in batches], [batch.task_list for batch in batches], [batch.action_candidate_list for batch in batches], [batch.target_indices for batch in batches], contains_first_step)
        else:
            raise NotImplementedError()

    def admissible_commands_recurrent_teacher_force(self, seq_observation_strings, seq_task_desc_strings, seq_action_candidate_list, seq_target_indices, contains_first_step=False):
        loss_list = []
        previous_dynamics = None
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        for step_no in range(self.dagger_replay_sample_history_length):
            expert_indicies = to_pt(np.array(seq_target_indices[step_no]), enable_cuda=self.use_cuda, type='long')
            h_obs, obs_mask = self.encode(seq_observation_strings[step_no], use_model="online")
            action_scores, _, current_dynamics = self.action_scoring(seq_action_candidate_list[step_no], h_obs, obs_mask, h_td, td_mask,
                                                                     previous_dynamics, use_model="online")
            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.dagger_replay_sample_update_from:
                previous_dynamics = previous_dynamics.detach()
                continue

            # softmax and cross-entropy
            loss = self.cross_entropy_loss(action_scores, expert_indicies)
            loss_list.append(loss)
        loss = torch.stack(loss_list).mean()
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(loss)

    def command_generation_recurrent_teacher_force(self, seq_observation_strings, seq_task_desc_strings, seq_target_strings, contains_first_step=False):
        loss_list = []
        previous_dynamics = None
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        for step_no in range(self.dagger_replay_sample_history_length):
            input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in seq_target_strings[step_no]]
            output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in seq_target_strings[step_no]]

            input_obs = self.get_word_input(seq_observation_strings[step_no])
            h_obs, obs_mask = self.encode(seq_observation_strings[step_no], use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
            current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)

            input_target = self.get_word_input(input_target_strings)
            ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)  # batch x target_length x vocab

            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.dagger_replay_sample_update_from:
                previous_dynamics = previous_dynamics.detach()
                continue

            batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
            loss = torch.mean(batch_loss)
            loss_list.append(loss)

        loss = torch.stack(loss_list).mean()
        if loss is None:
            return None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(loss)

    # free generation
    def admissible_commands_greedy_generation(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics):

        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")

            _, chosen_indices = self.choose_softmax_action(action_scores, action_masks)
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, current_dynamics

    def command_generation_greedy_generation(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_strings)

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            # greedy generation
            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)  # batch x target_length x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            res = [self.tokenizer.decode(item) for item in input_target_list]
            res = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in res]
            res = [item.replace(" in / on ", " in/on " ) for item in res]
            return res, current_dynamics

    def command_generation_beam_search_generation(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():

            batch_size = len(observation_strings)
            beam_width = self.beam_width
            if beam_width == 1:
                res, current_dynamics = self.command_generation_greedy_generation(observation_strings, task_desc_strings, previous_dynamics)
                res = [[item] for item in res]
                return res, current_dynamics
            generate_top_k = self.generate_top_k
            res = []

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):

                # starts from CLS tokens
                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = input_obs[b: b + 1]  # 1 x obs_len
                __obs_mask = obs_mask[b: b + 1]  # 1 x obs_len
                __aggregated_obs_representation = aggregated_obs_representation[b: b + 1]  # 1 x obs_len x hid
                if current_dynamics is not None:
                    __current_dynamics = current_dynamics[b: b + 1]  # 1 x hid
                else:
                    __current_dynamics = None
                ended_nodes = []

                # starting node -  previous node, input target, logp, length
                node = BeamSearchNode(None, __input_target_list, 0, 1)
                nodes_queue = PriorityQueue()
                # start the queue
                nodes_queue.put((node.val, node))
                queue_size = 1

                while(True):
                    # give up when decoding takes too long
                    if queue_size > 2000:
                        break

                    # fetch the best node
                    score, n = nodes_queue.get()
                    __input_target_list = n.input_target

                    if (n.input_target[-1] == self.word2id["[SEP]"] or n.length >= self.max_target_length) and n.previous_node != None:
                        ended_nodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(ended_nodes) >= generate_top_k:
                            break
                        else:
                            continue

                    input_target = pad_sequences([__input_target_list], dtype='int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)
                    # decode for one step using decoder
                    pred = self.online_net.decode(input_target, target_mask, __aggregated_obs_representation, __obs_mask, __current_dynamics, __input_obs)  # 1 x target_length x vocab
                    pred = pred[0][-1].cpu()
                    gt_zero = torch.gt(pred, 0.0).float()  # vocab
                    epsilon = torch.le(pred, 0.0).float() * 1e-8  # vocab
                    log_pred = torch.log(pred + epsilon) * gt_zero  # vocab

                    top_beam_width_log_probs, top_beam_width_indicies = torch.topk(log_pred, beam_width)
                    next_nodes = []

                    for new_k in range(beam_width):
                        pos = top_beam_width_indicies[new_k]
                        log_p = top_beam_width_log_probs[new_k].item()
                        node = BeamSearchNode(n, __input_target_list + [pos], n.log_prob + log_p, n.length + 1)
                        next_nodes.append((node.val, node))

                    # put them into queue
                    for i in range(len(next_nodes)):
                        score, nn = next_nodes[i]
                        nodes_queue.put((score, nn))
                    # increase qsize
                    queue_size += len(next_nodes) - 1

                # choose n best paths
                if len(ended_nodes) == 0:
                    ended_nodes = [nodes_queue.get() for _ in range(generate_top_k)]

                utterances = []
                for score, n in sorted(ended_nodes, key=operator.itemgetter(0)):
                    utte = n.input_target
                    utte_string = self.tokenizer.decode(utte)
                    utterances.append(utte_string)
                utterances = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in utterances]
                utterances = [item.replace(" in / on ", " in/on " ) for item in utterances]
                res.append(utterances)
            return res, current_dynamics

    # random generation
    def admissible_commands_random_generation(self, action_candidate_list):

        chosen_actions, chosen_indicies = [], []
        for i in range(len(action_candidate_list)):
            _action_idx = np.random.choice(len(action_candidate_list[i]))
            chosen_indicies.append(_action_idx)
            chosen_actions.append(action_candidate_list[i][_action_idx])
        return chosen_actions, chosen_indicies
