import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from eval_task import EvalTask
from env.thor_env import ThorEnv
import torch.multiprocessing as mp


class Leaderboard(EvalTask):
    '''
    dump action-sequences for leaderboard eval
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, splits, seen_actseqs, unseen_actseqs):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, splits, seen_actseqs, unseen_actseqs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, splits, seen_actseqs, unseen_actseqs):
        # reset model
        model.reset()

        # setup scene
        cls.setup_scene(env, traj_data, r_idx, args)

        # extract language features
        feat = model.featurize([traj_data], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        done, success = False, False
        actions = list()
        fails = 0
        t = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

            # forward model
            m_out = model.step(feat)
            m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            # check if <<stop>> was predicted
            if m_pred['action_low'] == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break

            # get action and mask
            action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
            mask = np.squeeze(mask, axis=0) if model.has_interaction(action) else None

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, api_action = env.va_interact(action, interact_mask=mask, smooth_nav=False)

            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # save action
            if api_action is not None:
                actions.append(api_action)

            # next time-step
            t += 1

        # actseq
        seen_ids = [t['task'] for t in splits['tests_seen']]
        actseq = {traj_data['task_id']: actions}

        # log action sequences
        lock.acquire()

        if traj_data['task_id'] in seen_ids:
            seen_actseqs.append(actseq)
        else:
            unseen_actseqs.append(actseq)

        lock.release()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
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

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        seen_files, unseen_files = self.splits['tests_seen'], self.splits['tests_unseen']

        # add seen trajectories to queue
        for traj in seen_files:
            task_queue.put(traj)

        # add unseen trajectories to queue
        for traj in unseen_files:
            task_queue.put(traj)

        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        self.model.test_mode = True
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
                                                       self.splits, self.seen_actseqs, self.unseen_actseqs))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # save
        self.save_results()

    def create_stats(self):
        '''
        storage for seen and unseen actseqs
        '''
        self.seen_actseqs, self.unseen_actseqs = self.manager.list(), self.manager.list()

    def save_results(self):
        '''
        save actseqs as JSONs
        '''
        results = {'tests_seen': list(self.seen_actseqs),
                   'tests_unseen': list(self.unseen_actseqs)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'tests_actseqs_dump_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_2.1.0")
    parser.add_argument('--model_path', type=str, default="model.pth")
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)

    # parse arguments
    args = parser.parse_args()

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    # leaderboard dump
    eval = Leaderboard(args, manager)

    # start threads
    eval.spawn_threads()