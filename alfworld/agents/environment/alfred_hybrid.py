import random
import importlib

import alfworld.agents.environment


class AlfredHybrid(object):
    '''
    Hybrid training manager for switching between AlfredTWEnv and AlfredThorEnv
    '''

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
        AlfredTWEnv = getattr(alfworld.agents.environment, "AlfredTWEnv")(self.config, train_eval=self.train_eval)
        AlfredThorEnv = getattr(alfworld.agents.environment, "AlfredThorEnv")(self.config, train_eval=self.train_eval)

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