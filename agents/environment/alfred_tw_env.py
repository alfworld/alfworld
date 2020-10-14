import os
import sys
import json
import glob
import random

sys.path.append(os.environ['ALFRED_ROOT'])
from agents.utils.misc import Demangler, get_templated_task_desc, add_task_to_grammar

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
            num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_train_games]
            self.num_games = len(self.game_files)
            print("Training with %d games" % (len(self.game_files)))
        else:
            num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_eval_games]
            self.num_games = len(self.game_files)
            print("Evaluating with %d games" % (len(self.game_files)))

    def get_game_logic(self):
        self.game_logic = {"pddl_domain": open(self.config['pddl']['domain']).read(),
                           "grammar": "\n".join(open(f).read() for f in glob.glob(os.path.join(os.environ['ALFRED_ROOT'], "data/textworld_data/logic/*.twl2")))}

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
