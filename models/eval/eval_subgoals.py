import os
import sys
import json
import numpy as np
from PIL import Image
from datetime import datetime
from env.thor_env import ThorEnv
from eval import Eval


class EvalSubgoals(Eval):
    '''
    evaluate subgoals by teacher-forching expert demonstrations
    '''

    # subgoal types
    ALL_SUBGOALS = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject']

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        # make subgoals list
        subgoals_to_evaluate = cls.ALL_SUBGOALS if args.subgoals.lower() == "all" else args.subgoals.split(',')
        subgoals_to_evaluate = [sg for sg in subgoals_to_evaluate if sg in cls.ALL_SUBGOALS]
        print ("Subgoals to evaluate: %s" % str(subgoals_to_evaluate))

        # create empty stats per subgoal
        for sg in subgoals_to_evaluate:
            successes[sg] = list()
            failures[sg] = list()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                subgoal_idxs = [sg['high_idx'] for sg in traj['plan']['high_pddl'] if sg['discrete_action']['action'] in subgoals_to_evaluate]
                for eval_idx in subgoal_idxs:
                    print("No. of trajectories left: %d" % (task_queue.qsize()))
                    cls.evaluate(env, model, eval_idx, r_idx, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, model, eval_idx, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # expert demonstration to reach eval_idx-1
        expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]

        # subgoal info
        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
        subgoal_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]

        # print subgoal info
        print("Evaluating: %s\nSubgoal %s (%d)\nInstr: %s" % (traj_data['root'], subgoal_action, eval_idx, subgoal_instr))

        # extract language features
        feat = model.featurize([traj_data], load_mask=False)

        # previous action for teacher-forcing during expert execution (None is used for initialization)
        prev_action = None

        done, subgoal_success = False, False
        fails = 0
        t = 0
        reward = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps + len(expert_init_actions):
                break

            # extract visual feats
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

            # expert teacher-forcing upto subgoal
            if t < len(expert_init_actions):
                # get expert action
                action = expert_init_actions[t]
                subgoal_completed = traj_data['plan']['low_actions'][t+1]['high_idx'] != traj_data['plan']['low_actions'][t]['high_idx']
                compressed_mask = action['args']['mask'] if 'mask' in action['args'] else None
                mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None

                # forward model
                if not args.skip_model_unroll_with_expert:
                    model.step(feat, prev_action=prev_action)
                    prev_action = action['action'] if not args.no_teacher_force_unroll_with_expert else None

                # execute expert action
                success, _, _, err, _ = env.va_interact(action['action'], interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                if not success:
                    print ("expert initialization failed")
                    break

                # update transition reward
                _, _ = env.get_transition_reward()

            # subgoal evaluation
            else:
                # forward model
                m_out = model.step(feat, prev_action=prev_action)
                m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
                m_pred = list(m_pred.values())[0]

                # get action and mask
                action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
                mask = np.squeeze(mask, axis=0) if model.has_interaction(action) else None

                # debug
                if args.debug:
                    print("Pred: ", action)

                # update prev action
                prev_action = str(action)

                if action not in cls.TERMINAL_TOKENS:
                    # use predicted action and mask (if provided) to interact with the env
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                    if not t_success:
                        fails += 1
                        if fails >= args.max_fails:
                            print("Interact API failed %d times" % (fails) + "; latest error '%s'" % err)
                            break

                # next time-step
                t_reward, t_done = env.get_transition_reward()
                reward += t_reward

                # update subgoals
                curr_subgoal_idx = env.get_subgoal_idx()
                if curr_subgoal_idx == eval_idx:
                    subgoal_success = True
                    break

                # terminal tokens predicted
                if action in cls.TERMINAL_TOKENS:
                    print("predicted %s" % action)
                    break

            # increment time index
            t += 1

        # metrics
        pl = float(t - len(expert_init_actions)) + 1 # +1 for last action
        expert_pl = len([ll for ll in traj_data['plan']['low_actions'] if ll['high_idx'] == eval_idx])

        s_spl = (1 if subgoal_success else 0) * min(1., expert_pl / (pl + sys.float_info.epsilon))
        plw_s_spl = s_spl * expert_pl

        # log success/fails
        lock.acquire()

        # results
        for sg in cls.ALL_SUBGOALS:
            results[sg] = {
                    'sr': 0.,
                    'successes': 0.,
                    'evals': 0.,
                    'sr_plw': 0.
            }

        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'subgoal_idx': int(eval_idx),
                     'subgoal_type': subgoal_action,
                     'subgoal_instr': subgoal_instr,
                     'subgoal_success_spl': float(s_spl),
                     'subgoal_path_len_weighted_success_spl': float(plw_s_spl),
                     'subgoal_path_len_weight': float(expert_pl),
                     'reward': float(reward)}
        if subgoal_success:
            sg_successes = successes[subgoal_action]
            sg_successes.append(log_entry)
            successes[subgoal_action] = sg_successes
        else:
            sg_failures = failures[subgoal_action]
            sg_failures.append(log_entry)
            failures[subgoal_action] = sg_failures

        # save results
        print("-------------")
        subgoals_to_evaluate = list(successes.keys())
        subgoals_to_evaluate.sort()
        for sg in subgoals_to_evaluate:
            num_successes, num_failures = len(successes[sg]), len(failures[sg])
            num_evals = len(successes[sg]) + len(failures[sg])
            if num_evals > 0:
                sr = float(num_successes) / num_evals
                total_path_len_weight = sum([entry['subgoal_path_len_weight'] for entry in successes[sg]]) + \
                                        sum([entry['subgoal_path_len_weight'] for entry in failures[sg]])
                sr_plw = float(sum([entry['subgoal_path_len_weighted_success_spl'] for entry in successes[sg]]) +
                                    sum([entry['subgoal_path_len_weighted_success_spl'] for entry in failures[sg]])) / total_path_len_weight

                results[sg] = {
                    'sr': sr,
                    'successes': num_successes,
                    'evals': num_evals,
                    'sr_plw': sr_plw
                }

                print("%s ==========" % sg)
                print("SR: %d/%d = %.3f" % (num_successes, num_evals, sr))
                print("PLW SR: %.3f" % (sr_plw))
        print("------------")

        lock.release()

    def create_stats(self):
        '''
        storage for success, failure, and results info
        '''
        self.successes, self.failures = self.manager.dict(), self.manager.dict()
        self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': dict(self.successes),
                   'failures': dict(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'subgoal_results_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)