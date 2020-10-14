import os
import json
import glob
import numpy as np

import time
import traceback
import threading
from queue import Queue
from threading import Thread
import sys
import random
import importlib

sys.path.append(os.environ['ALFRED_ROOT'])
from utils.misc import Demangler, get_templated_task_desc, add_task_to_grammar
from env.thor_env import ThorEnv
from agents.expert import HandCodedThorAgent, HandCodedAgentTimeout
from detector.mrcnn import load_pretrained_model

import textworld
import textworld.agents
import textworld.gym
import gym
TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}


class AlfredDemangler(textworld.core.Wrapper):

    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._game.infos, shuffle=self.shuffle)
        for info in self._game.infos.values():
            info.name = demangler.demangle_alfred_name(info.id)


class AlfredInfos(textworld.core.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamefile = None

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self._gamefile = args[0]

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        state["extra.gamefile"] = self._gamefile
        return state


class AlfredTWEnv(object):

    def __init__(self, config, train_eval="train"):
        print("Initializing AlfredTWEnv...")
        self.config = config
        self.train_eval = train_eval

        self.goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']
        self.get_game_logic()
        self.gen_game_files(regen_game_files=self.config['env']['regen_game_files'])

        self.random_seed = 42

    def seed(self, num):
        self.random_seed = num

    def gen_game_files(self, regen_game_files=False, verbose=False):
        def log(info):
            if verbose:
                print(info)

        self.game_files = []

        if self.train_eval == "train":
            data_path = self.config['dataset']['data_path']
        elif self.train_eval == "eval_in_distribution":
            data_path = self.config['dataset']['eval_id_data_path']
        elif self.train_eval == "eval_out_of_distribution":
            data_path = self.config['dataset']['eval_ood_data_path']
        print("Checking for solvable games...")

        # get task types
        assert len(self.config['env']['task_types']) > 0
        task_types = []
        for tt_id in self.config['env']['task_types']:
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        env = None
        count = 0
        for root, dirs, files in os.walk(data_path, topdown=False):
            if 'traj_data.json' in files:
                count += 1

                # Filenames
                pddl_path = os.path.join(root, 'initial_state.pddl')
                json_path = os.path.join(root, 'traj_data.json')
                game_file_path = os.path.join(root, "game.tw-pddl")

                # Skip if no PDDL file
                if not os.path.exists(pddl_path):
                    log("Skipping %s, PDDL file is missing" % root)
                    continue

                if 'movable' in root or 'Sliced' in root:
                    log("Movable & slice trajs not supported %s" % (root))
                    continue

                # Get goal description
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)

                # Check for any task_type constraints
                if not traj_data['task_type'] in task_types:
                    log("Skipping task type")
                    continue

                # Add task description to grammar
                grammar = add_task_to_grammar(self.game_logic['grammar'], traj_data, goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)

                # Check if a game file exists
                if not regen_game_files and os.path.exists(game_file_path):
                    with open(game_file_path, 'r') as f:
                        gamedata = json.load(f)

                    # Check if previously checked if solvable
                    if 'solvable' in gamedata:
                        if not gamedata['solvable']:
                            log("Skipping known %s, unsolvable game!" % root)
                            continue
                        else:
                            # write task desc to tw.game-pddl file
                            gamedata['grammar'] = grammar
                            if self.goal_desc_human_anns_prob > 0:
                                json.dump(gamedata, open(game_file_path, 'w'))
                            self.game_files.append(game_file_path)
                            continue

                # To avoid making .tw game file, we are going to load the gamedata directly.
                gamedata = dict(pddl_domain=self.game_logic['pddl_domain'],
                                grammar=grammar,
                                pddl_problem=open(pddl_path).read(),
                                solvable=False)
                json.dump(gamedata, open(game_file_path, "w"))

                # Check if game is solvable (expensive) and save it in the gamedata
                if not env:
                    demangler = AlfredDemangler(shuffle=False)
                    env = textworld.start(game_file_path, wrappers=[demangler])
                gamedata['solvable'], err, expert_steps = self.is_solvable(env, game_file_path)
                json.dump(gamedata, open(game_file_path, "w"))

                # Skip unsolvable games
                if not gamedata['solvable']:
                    continue

                # Add to game file list
                self.game_files.append(game_file_path)

                # Print solvable
                log("%s (%d steps), %d/%d solvable games" % (game_file_path, expert_steps, len(self.game_files), count))

        print("Overall we have %s games" % (str(len(self.game_files))))
        self.num_games = len(self.game_files)

        if self.train_eval == "train":
            num_train_games = self.config['dataset']['num_train_games'] if 'num_train_games' in self.config['dataset'] else len(self.game_files)
            self.game_files = self.game_files[:num_train_games]
            self.num_games = len(self.game_files)
            print("Training with %d games" % (len(self.game_files)))
        else:
            num_eval_games = self.config['dataset']['num_eval_games'] if 'num_eval_games' in self.config['dataset'] else len(self.game_files)
            self.game_files = self.game_files[:num_eval_games]
            self.num_games = len(self.game_files)
            print("Evaluating with %d games" % (len(self.game_files)))

    def get_game_logic(self):
        self.game_logic = {"pddl_domain": open(self.config['pddl']['domain']).read(),
                           "grammar": "\n".join(open(f).read() for f in glob.glob(os.path.join(os.environ['ALFRED_ROOT'], "textworld_data/logic/*.twl2")))}

    def is_solvable(self, env, game_file_path,
                    random_perturb=True, random_start=10, random_prob_after_state=0.15):
        done = False
        steps = 0
        try:
            env.load(game_file_path)
            env.infos.expert_type = "handcoded"
            env.infos.expert_plan = True

            game_state = env.reset()
            while not done:
                expert_action = game_state['expert_plan'][0]
                random_action = random.choice(game_state.admissible_commands)

                command = expert_action
                if random_perturb:
                    if steps <= random_start or random.random() < random_prob_after_state:
                        command = random_action

                game_state, reward, done = env.step(command)
                steps += 1
        except Exception as e:
            print("Unsolvable: %s (%s)" % (str(e), game_file_path))
            return False, str(e), steps
        return True, "", steps

    def init_env(self, batch_size):
        # Register a new Gym environment.
        training_method = self.config["general"]["training_method"]
        expert_type = self.config["env"]["expert_type"]
        if training_method == "dqn":
            infos = textworld.EnvInfos(won=True, admissible_commands=True, expert_type=expert_type, expert_plan=False, extras=["gamefile"])
            max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            expert_plan = True if self.train_eval == "train" else False
            infos = textworld.EnvInfos(won=True, admissible_commands=True, expert_type=expert_type, expert_plan=expert_plan, extras=["gamefile"])
            max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]
        else:
            raise NotImplementedError

        domain_randomization = self.config["env"]["domain_randomization"]
        if self.train_eval != "train":
            domain_randomization = False
        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        env_id = textworld.gym.register_games(self.game_files, infos,
                                              batch_size=batch_size,
                                              asynchronous=True,
                                              max_episode_steps=max_nb_steps_per_episode,
                                              wrappers=[alfred_demangler, AlfredInfos])
        # Launch Gym environment.
        env = gym.make(env_id)
        return env


class AlfredThorEnv(object):

    class Thor(threading.Thread):
        def __init__(self, queue, train_eval="train"):
            Thread.__init__(self)
            self.action_queue = queue
            self.mask_rcnn = None
            self.env =  None
            self.train_eval = train_eval
            self.controller_type = "oracle"

        def run(self):
            while True:
                action, reset, task_file = self.action_queue.get()
                try:
                    if reset:
                        self.reset(task_file)
                    else:
                        self.step(action)
                finally:
                    self.action_queue.task_done()

        def init_env(self, config):
            self.config = config

            screen_height = config['env']['thor']['screen_height']
            screen_width = config['env']['thor']['screen_width']
            smooth_nav = config['env']['thor']['smooth_nav']
            save_frames_to_disk = config['env']['thor']['save_frames_to_disk']

            if not self.env:
                self.env = ThorEnv(player_screen_height=screen_height,
                                   player_screen_width=screen_width,
                                   smooth_nav=smooth_nav,
                                   save_frames_to_disk=save_frames_to_disk)
            self.controller_type = self.config['controller']['type']
            self._done = False
            self._res = ()
            self._feedback = ""
            self.expert = HandCodedThorAgent(self.env, max_steps=200)
            self.prev_command = ""
            self.load_mask_rcnn()

        def load_mask_rcnn(self):
            # load pretrained MaskRCNN model if required
            if 'mrcnn' in self.config['controller']['type'] and not self.mask_rcnn:
                model_path = os.path.join(os.environ['ALFRED_ROOT'],
                                          self.config['mask_rcnn']['pretrained_model_path'])
                self.mask_rcnn = load_pretrained_model(model_path)

        def set_task(self, task_file):
            self.task_file = task_file
            self.traj_root = os.path.dirname(task_file)
            with open(task_file, 'r') as f:
                self.traj_data = json.load(f)

        def reset(self, task_file):
            assert self.env
            assert self.controller_type

            self.set_task(task_file)

            # scene setup
            scene_num = self.traj_data['scene']['scene_num']
            object_poses = self.traj_data['scene']['object_poses']
            dirty_and_empty = self.traj_data['scene']['dirty_and_empty']
            object_toggles = self.traj_data['scene']['object_toggles']
            scene_name = 'FloorPlan%d' % scene_num
            self.env.reset(scene_name)
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            # recording
            save_frames_path = self.config['env']['thor']['save_frames_path']
            self.env.save_frames_path = os.path.join(save_frames_path, self.traj_root.replace('../', ''))

            # initialize to start position
            self.env.step(dict(self.traj_data['scene']['init_action']))            # print goal instr
            task_desc = get_templated_task_desc(self.traj_data)
            print("Task: %s" % task_desc)
            # print("Task: %s" % (self.traj_data['turk_annotations']['anns'][0]['task_desc']))

            # setup task for reward
            class args: pass
            args.reward_config = os.path.join(os.environ['ALFRED_ROOT'], 'models/config/rewards.json')
            self.env.set_task(self.traj_data, args, reward_type='dense')

            # set controller
            self.controller_type = self.config['controller']['type']
            self.goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']
            load_receps = self.config['controller']['load_receps']
            debug = self.config['controller']['debug']

            if self.controller_type == 'oracle':
                from models.embodied.oracle import OracleAgent
                self.controller = OracleAgent(self.env, self.traj_data, self.traj_root,
                                              load_receps=load_receps, debug=debug,
                                              goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)
            elif self.controller_type == 'oracle_astar':
                from models.embodied.oracle_astar import OracleAStarAgent
                self.controller = OracleAStarAgent(self.env, self.traj_data, self.traj_root,
                                                   load_receps=load_receps, debug=debug,
                                                   goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)
            elif self.controller_type == 'mrcnn':
                from models.embodied.mrcnn import MaskRCNNAgent
                self.controller = MaskRCNNAgent(self.env, self.traj_data, self.traj_root,
                                                pretrained_model=self.mask_rcnn,
                                                load_receps=load_receps, debug=debug,
                                                goal_desc_human_anns_prob=self.goal_desc_human_anns_prob,
                                                save_detections_to_disk=self.env.save_frames_to_disk, save_detections_path=self.env.save_frames_path)
            elif self.controller_type == 'mrcnn_astar':
                from models.embodied.mrcnn_astar import MaskRCNNAStarAgent
                self.controller = MaskRCNNAStarAgent(self.env, self.traj_data, self.traj_root,
                                                     pretrained_model=self.mask_rcnn,
                                                     load_receps=load_receps, debug=debug,
                                                     goal_desc_human_anns_prob=self.goal_desc_human_anns_prob,
                                                     save_detections_to_disk=self.env.save_frames_to_disk, save_detections_path=self.env.save_frames_path)
            else:
                raise NotImplementedError()

            # zero steps
            self.steps = 0

            # reset expert state
            self.expert.reset(task_file)
            self.prev_command = ""

            # return intro text
            self._feedback = self.controller.feedback
            self._res = self.get_info()

            return self._feedback

        def step(self, action):
            if not self._done:
                # take action
                self.prev_command = str(action)
                self._feedback = self.controller.step(action)
                self._res = self.get_info()
                if self.env.save_frames_to_disk:
                    self.record_action(action)
            self.steps += 1

        def get_results(self):
            return self._res

        def record_action(self, action):
            txt_file = os.path.join(self.env.save_frames_path, 'action.txt')
            with open(txt_file, 'a+') as f:
                f.write("%s\r\n" % str(action))

        def get_info(self):
            won = self.env.get_goal_satisfied()
            pcs = self.env.get_goal_conditions_met()
            goal_condition_success_rate = pcs[0] / float(pcs[1])
            acs = self.controller.get_admissible_commands()

            # expert action
            if self.train_eval == "train":
                game_state = {
                    'admissible_commands': acs,
                    'feedback': self._feedback,
                    'won': won
                }
                expert_actions = ["look"]
                try:
                    if not self.prev_command:
                        self.expert.observe(game_state['feedback'])
                    else:
                        next_action = self.expert.act(game_state, 0, won, self.prev_command)
                        if next_action in acs:
                            expert_actions = [next_action]
                except HandCodedAgentTimeout:
                    print("Expert Timeout")
                except Exception as e:
                    print(e)
                    traceback.print_exc()
            else:
                expert_actions = []

            training_method = self.config["general"]["training_method"]
            if training_method == "dqn":
                max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
            elif training_method == "dagger":
                max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]
            else:
                raise NotImplementedError
            self._done = won or self.steps > max_nb_steps_per_episode
            return (self._feedback, self._done, acs, won, goal_condition_success_rate, expert_actions)

        def get_last_frame(self):
            return self.env.last_event.frame[:,:,::-1]

        def get_exploration_frames(self):
            return self.controller.get_exploration_frames()

    def __init__(self, config, train_eval="train"):
        print("Initialize AlfredThorEnv...")
        self.config = config
        self.train_eval = train_eval
        self.random_seed = 123
        self.batch_size = 1
        self.envs = []
        self.action_queues = []
        self.get_env_paths()

    def close(self):
        for env in self.envs:
            env.env.stop()

    def seed(self, seed):
        self.random_seed = seed

    def get_env_paths(self):
        self.json_file_list = []

        if self.train_eval == "train":
            data_path = self.config['dataset']['data_path']
        elif self.train_eval == "eval_in_distribution":
            data_path = self.config['dataset']['eval_id_data_path']
        elif self.train_eval == "eval_out_of_distribution":
            data_path = self.config['dataset']['eval_ood_data_path']
        else:
            raise Exception("Invalid split. Must be either train or eval")

        # get task types
        assert len(self.config['env']['task_types']) > 0
        task_types = []
        for tt_id in self.config['env']['task_types']:
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        for root, dirs, files in os.walk(data_path, topdown=False):
            if 'traj_data.json' in files:
                # Skip movable and slice objects object tasks
                if 'movable' in root or 'Sliced' in root:
                    continue

                # File paths
                json_path = os.path.join(root, 'traj_data.json')
                game_file_path = os.path.join(root, "game.tw-pddl")

                # Load trajectory file
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)

                # Check for any task_type constraints
                if not traj_data['task_type'] in task_types:
                    continue

                self.json_file_list.append(json_path)

                # # Only add solvable games
                # if os.path.exists(game_file_path):
                #     with open(game_file_path, 'r') as f:
                #         gamedata = json.load(f)
                #
                #     if 'solvable' in gamedata and gamedata['solvable']:
                #         self.json_file_list.append(json_path)

        print("Overall we have %s games..." % (str(len(self.json_file_list))))
        self.num_games = len(self.json_file_list)

        if self.train_eval == "train":
            num_train_games = self.config['dataset']['num_train_games'] if 'num_train_games' in self.config['dataset'] else len(self.json_file_list)
            self.json_file_list = self.json_file_list[:num_train_games]
            self.num_games = len(self.json_file_list)
            print("Training with %d games" % (len(self.json_file_list)))
        else:
            num_eval_games = self.config['dataset']['num_eval_games'] if 'num_eval_games' in self.config['dataset'] else len(self.json_file_list)
            self.json_file_list = self.json_file_list[:num_eval_games]
            self.num_games = len(self.json_file_list)
            print("Evaluating with %d games" % (len(self.json_file_list)))

    def init_env(self, batch_size):
        self.get_env_paths()
        self.batch_size = batch_size
        self.action_queues = []
        self.task_order = [""] * self.batch_size
        for n in range(self.batch_size):
            queue = Queue()
            env = self.Thor(queue, self.train_eval)
            self.action_queues.append(queue)
            self.envs.append(env)
            env.daemon = True
            env.start()
            env.init_env(self.config)
        return self

    def reset(self):
        # set tasks
        batch_size = self.batch_size
        # reset envs

        if self.train_eval == 'train':
            tasks = random.sample(self.json_file_list, k=batch_size)
        else:
            if len(self.json_file_list)-batch_size > batch_size:
                tasks = [self.json_file_list.pop(random.randrange(len(self.json_file_list))) for _ in range(batch_size)]
            else:
                tasks = random.sample(self.json_file_list, k=batch_size)
                self.get_env_paths()

        for n in range(batch_size):
            self.action_queues[n].put((None, True, tasks[n]))

        obs, dones, infos = self.wait_and_get_info()
        return obs, infos

    def step(self, actions):
        '''
        executes actions in parallel and waits for all env to finish
        '''

        batch_size = self.batch_size
        for n in range(batch_size):
            self.action_queues[n].put((actions[n], False, ""))

        obs, dones, infos = self.wait_and_get_info()
        return obs, None, dones, infos

    def wait_and_get_info(self):
        obs, dones, admissible_commands, wons, gamefiles, expert_plans, gc_srs = [], [], [], [], [], [], []

        # wait for all threads
        for n in range(self.batch_size):
            self.action_queues[n].join()
            feedback, done, acs, won, gc_sr, expert_actions = self.envs[n].get_results()
            obs.append(feedback)
            dones.append(done)
            admissible_commands.append(acs)
            wons.append(won)
            gc_srs.append(gc_sr)
            gamefiles.append(self.envs[n].traj_root)
            expert_plans.append(expert_actions)

        infos = {'admissible_commands': admissible_commands,
                 'won': wons,
                 'goal_condition_success_rate': gc_srs,
                 'extra.gamefile': gamefiles,
                 'expert_plan': expert_plans}
        return obs, dones, infos

    def get_frames(self):
        images = []
        for n in range(self.batch_size):
            images.append(self.envs[n].get_last_frame())
        return np.array(images)

    def get_exploration_frames(self):
        images = []
        for n in range(self.batch_size):
            images.append(self.envs[n].get_exploration_frames())
        return images

class AlfredHybrid(object):

    def __init__(self, config, train_eval="train"):
        print("Setting up AlfredHybrid env")
        self.hybrid_start_eps = config["env"]["hybrid"]["start_eps"]
        self.hybrid_thor_prob = config["env"]["hybrid"]["thor_prob"]

        self.config = config
        self.train_eval = train_eval

        self.curr_env = "tw"
        self.eval_mode = config["env"]["hybrid"]["eval_mode"]
        self.num_resets = 0

    def choose_env(self):
        if self.curr_env == "thor":
            return self.thor
        else:
            return self.tw

    def init_env(self, batch_size):
        AlfredTWEnv = getattr(importlib.import_module("environment"), "AlfredTWEnv")(self.config, train_eval=self.train_eval)
        AlfredThorEnv = getattr(importlib.import_module("environment"), "AlfredThorEnv")(self.config, train_eval=self.train_eval)

        self.batch_size = batch_size
        self.tw = AlfredTWEnv.init_env(batch_size)
        self.thor = AlfredThorEnv.init_env(batch_size)
        return self

    def seed(self, num):
        env = self.choose_env()
        return env.seed(num)

    def step(self, actions):
        env = self.choose_env()
        return env.step(actions)

    def reset(self):
        if "eval" in self.train_eval:
            assert(self.eval_mode in ['tw', 'thor'])
            self.curr_env = self.eval_mode
        else:
            if self.num_resets >= self.hybrid_start_eps:
                self.curr_env = "thor" if random.random() < self.hybrid_thor_prob else "tw"
            else:
                self.curr_env = "tw"
        env = self.choose_env()
        obs, infos = env.reset()
        self.num_resets += self.batch_size
        return obs, infos