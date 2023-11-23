import os
import sys

import json
import glob
import os
import alfworld.gen.constants as constants
import cv2
import shutil
import numpy as np
import argparse
import threading
import time
import copy
import random
import alfworld.gen
import alfworld.gen.goal_library as glib
from alfworld.gen.utils.video_util import VideoSaver
from alfworld.gen.utils.py_util import walklevel
from alfworld.env.thor_env import ThorEnv
from alfworld.gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge

from alfworld.gen.utils import game_util, py_util
from alfworld.gen.graph import graph_obj


TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
HIGH_RES_IMAGES_FOLDER = "high_res_images"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

video_saver = VideoSaver()


class TextWorldTaskGameStateFullKnowledge(TaskGameStateFullKnowledge):
    def __init__(self, env, seed=None, action_space=None):
        super(TextWorldTaskGameStateFullKnowledge, self).__init__(env, seed, action_space)

    def state_to_pddl(self, traj_data):
        object_dict = game_util.get_object_dict(self.env.last_event.metadata)
        domain = 'alfred'
        problem_id = traj_data['task_id']
        pddl_params = traj_data['pddl_params']

        points_source = os.path.join(alfworld.gen.__path__[0], 'layouts/%s-openable.json' % traj_data['scene']['floor_plan'])
        with open(points_source, 'r') as f:
            openable_object_to_point = json.load(f)
        self.openable_object_to_point = openable_object_to_point


        receptacle_types = copy.deepcopy(constants.RECEPTACLES) - set(constants.MOVABLE_RECEPTACLES)
        objects = copy.deepcopy(constants.OBJECTS_SET) - receptacle_types
        object_str = '\n        '.join([obj + ' # object' for obj in objects])

        self.knife_obj = {'ButterKnife', 'Knife'} if pddl_params['object_sliced'] else {}

        otype_str = '\n        '.join([obj + 'Type # otype' for obj in objects])
        rtype_str = '\n        '.join([obj + 'Type # rtype' for obj in receptacle_types])


        pddl_goal = get_goal_pddl(traj_data)

        pddl_start = '''
(define (problem plan_%s)
(:domain %s)
(:objects
agent1 # agent
%s
%s
%s
''' % (
            problem_id,
            domain,

            object_str,
            otype_str,
            rtype_str,
        )

        pddl_init = '''
(:init
'''

        pddl_start = fix_pddl_str_chars(pddl_start)
        pddl_init = fix_pddl_str_chars(pddl_init)
        pddl_goal = fix_pddl_str_chars(pddl_goal)

        # pddl_mid section
        pose = game_util.get_pose(self.env.last_event)
        agent_location = 'loc|%d|%d|%d|%d' % (pose[0], pose[1], pose[2], pose[3])
        #agent_location = "Middle of the room."

        agent_location_str = '\n        (atLocation agent1 %s)' % agent_location
        opened_receptacle_str = '\n        '.join(['(opened %s)' % obj
                                                   for obj in list()])

        movable_recep_cls_with_knife = []
        in_receptacle_strs = []
        was_in_receptacle_strs = []
        for key, val in self.in_receptacle_ids.items():
            if len(val) == 0:
                continue
            key_cls = object_dict[key]['objectType']
            if key_cls in constants.MOVABLE_RECEPTACLES_SET:
                recep_str = 'inReceptacleObject'
            else:
                recep_str = 'inReceptacle'
            for vv in val:
                vv_cls = object_dict[vv]['objectType']
                if (vv_cls == pddl_params['object_target'] or
                        (pddl_params['mrecep_target'] is not None and vv_cls == pddl_params['mrecep_target']) or
                        (len(self.knife_obj) > 0 and vv_cls in self.knife_obj)):

                    # if knife is inside a movable receptacle, make sure to add it to the object list
                    if recep_str == 'inReceptacleObject':
                        movable_recep_cls_with_knife.append(key_cls)

                in_receptacle_strs.append('(%s %s %s)' % (
                    recep_str,
                    vv,
                    key)
                                              )
                # if key_cls not in constants.MOVABLE_RECEPTACLES_SET and vv_cls == pddl_params['object_target']:
                #     was_in_receptacle_strs.append('(wasInReceptacle  %s %s)' % (vv, key))

        in_receptacle_str = '\n        '.join(in_receptacle_strs)
        was_in_receptacle_str = '\n        '.join(was_in_receptacle_strs)

        # Note which openable receptacles we can safely open (precomputed).
        openable_objects = self.openable_object_to_point.keys()

        metadata_objects = self.env.last_event.metadata['objects']
        receptacles = set({obj['objectId'] for obj in metadata_objects
                           if obj['objectType'] in constants.RECEPTACLES and obj['objectType'] not in constants.MOVABLE_RECEPTACLES_SET})

        objects = set({obj['objectId'] for obj in metadata_objects if
                       (obj['objectType'] == pddl_params['object_target']
                        or obj['objectType'] in constants.MOVABLE_RECEPTACLES_SET
                        or (pddl_params['mrecep_target'] is not None and obj['objectType'] == pddl_params['mrecep_target'])
                        or ((pddl_params['toggle_target'] is not None and obj['objectType'] == pddl_params['toggle_target'])
                            or ((len(self.knife_obj) > 0 and
                                 (obj['objectType'] in self.knife_obj or
                                  obj['objectType'] in movable_recep_cls_with_knife)))))})

        objects = set(obj['objectId'] for obj in metadata_objects if obj['objectId'] not in receptacles)
        #from ipdb import set_trace; set_trace()

        if len(self.inventory_ids) > 0:
            objects = objects | self.inventory_ids
        if len(self.placed_items) > 0:
            objects = objects | self.placed_items

        receptacle_str = '\n        '.join(sorted([receptacle + ' # receptacle'
                                                   for receptacle in receptacles]))

        object_str = '\n        '.join(sorted([obj + ' # object' for obj in objects]))

        locations = set()
        for key, val in self.receptacle_to_point.items():
            key_cls = object_dict[key]['objectType']
            if key_cls not in constants.MOVABLE_RECEPTACLES_SET:
                locations.add(tuple(val.tolist()))
                # locations.add(key + "|loc")
        for obj, loc in self.object_to_point.items():
            obj_cls = object_dict[obj]['objectType']
            # if (obj_cls == pddl_params['object_target'] or
            #         (pddl_params['toggle_target'] is not None and obj_cls == pddl_params['toggle_target']) or
            #         (len(self.knife_obj) > 0 and obj_cls in self.knife_obj) or
            #         (obj_cls in constants.MOVABLE_RECEPTACLES_SET)):
            if obj_cls in constants.OBJECTS:
                locations.add(tuple(loc))

        location_str = ('\n        '.join(['loc|%d|%d|%d|%d # location' % (*loc,)
                                          for loc in locations]) +
                       '\n        %s # location' % agent_location)
        #location_str = ('\n        '.join(['loc|%d|%d|%d|%d # location' % (*loc,)
        # location_str = ('\n        '.join(['{loc} # location'.format(loc=loc) for loc in locations]) +
        #                 '\n        %s # location' % agent_location)
        # location_str = '\n        %s # location' % agent_location
        if constants.PRUNE_UNREACHABLE_POINTS:
            # don't flag problematic receptacleTypes for the planner.
            receptacle_type_str = '\n        '.join(['(receptacleType %s %sType)' % (
                receptacle, object_dict[receptacle]['objectType']) for receptacle in receptacles
                                                     if object_dict[receptacle]['objectType'] not in constants.OPENABLE_CLASS_SET or
                                                     receptacle in openable_objects])
        else:
            receptacle_type_str = '\n        '.join(['(receptacleType %s %sType)' % (
                receptacle, object_dict[receptacle]['objectType']) for receptacle in receptacles])

        object_type_str = '\n        '.join(['(objectType %s %sType)' % (
            obj, object_dict[obj]['objectType']) for obj in objects if object_dict[obj]['objectType'] in constants.OBJECTS])

        receptacle_objects_str = '\n        '.join(['(isReceptacleObject %s)' % (
            obj) for obj in objects if object_dict[obj]['objectType'] in constants.MOVABLE_RECEPTACLES_SET])

        if constants.PRUNE_UNREACHABLE_POINTS:
            openable_str = '\n        '.join(['(openable %s)' % receptacle for receptacle in receptacles
                                              if object_dict[receptacle]['objectType'] in constants.OPENABLE_CLASS_SET])
        else:
            # don't flag problematic open objects as openable for the planner.
            openable_str = '\n        '.join(['(openable %s)' % receptacle for receptacle in receptacles
                                              if object_dict[receptacle]['objectType'] in constants.OPENABLE_CLASS_SET and
                                              receptacle in openable_objects])

        # dists = []
        # dist_points = list(locations | {(pose[0], pose[1], pose[2], pose[3])})
        # for dd, l_start in enumerate(dist_points[:-1]):
        #     for l_end in dist_points[dd + 1:]:
        #         actions, path = self.gt_graph.get_shortest_path_unweighted(l_start, l_end)
        #         # Should cost one more for the trouble of going there at all. Discourages waypoints.
        #         dist = len(actions) + 1
        #         dists.append('(= (distance loc|%d|%d|%d|%d loc|%d|%d|%d|%d) %d)' % (
        #             l_start[0], l_start[1], l_start[2], l_start[3],
        #             l_end[0], l_end[1], l_end[2], l_end[3], dist))
        #         dists.append('(= (distance loc|%d|%d|%d|%d loc|%d|%d|%d|%d) %d)' % (
        #             l_end[0], l_end[1], l_end[2], l_end[3],
        #             l_start[0], l_start[1], l_start[2], l_start[3], dist))
        # location_distance_str = '\n        '.join(dists)
        location_distance_str = ""



        # pickupable objects
        pickupable_str =  '\n        '.join(['(pickupable %s)' % obj_id for obj_id in objects if object_dict[obj_id]["pickupable"]])

        # clean objects
        cleanable_str = '\n        '.join(['(cleanable %s)' % obj for obj in objects
                                           if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Cleanable']])

        is_clean_str = '\n        '.join((['(isClean %s)' % obj
                                           for obj in self.cleaned_object_ids if object_dict[obj]['objectType'] == pddl_params['object_target']]))

        # heat objects
        heatable_str = '\n        '.join(['(heatable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Heatable']])

        is_hot_str = '\n        '.join((['(isHot %s)' % obj
                                         for obj in self.hot_object_ids if object_dict[obj]['objectType'] == pddl_params['object_target']]))

        # cool objects
        coolable_str = '\n        '.join(['(coolable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Coolable']])

        # toggleable objects
        toggleable_str = '\n        '.join(['(toggleable %s)' % obj for obj in objects
                                            if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Toggleable']])

        is_on_str = '\n        '.join(['(isOn %s)' % obj
                                       for obj in self.on_object_ids
                                       if (pddl_params['toggle_target'] is not None
                                           and object_dict[obj]['objectType'] == pddl_params['toggle_target'])])

        # sliceable objects
        sliceable_str = '\n        '.join(['(sliceable %s)' % obj for obj in objects
                                           if (object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Sliceable'])])

        # sliced objects
        # TODO cleanup: sliced_object_ids is never added to. Does that matter?
        is_sliced_str = '\n        '.join((['(isSliced %s)' % obj
                                            for obj in self.sliced_object_ids
                                            if object_dict[obj]['objectType'] == pddl_params['object_target']]))

        # look for objects that are already cool
        for (key, val) in self.was_in_receptacle_ids.items():
            if 'Fridge' in key:
                for vv in val:
                    self.cool_object_ids.add(vv)

        is_cool_str = '\n        '.join((['(isCool %s)' % obj
                                          for obj in self.cool_object_ids if object_dict[obj]['objectType'] == pddl_params['object_target']]))

        # Receptacle Objects
        recep_obj_str = '\n        '.join(['(isReceptacleObject %s)' % obj for obj in receptacles
                                           if (object_dict[obj]['objectType'] in constants.MOVABLE_RECEPTACLES_SET and
                                               (pddl_params['mrecep_target'] is not None and object_dict[obj]['objectType'] == pddl_params['mrecep_target']))])

        receptacle_nearest_point_strs = sorted(
            # ['(receptacleAtLocation {obj} {loc})'.format(obj=obj_id, loc=obj_id + "|loc")
            ['(receptacleAtLocation %s loc|%d|%d|%d|%d)' % (obj_id, *point)
             for obj_id, point in self.receptacle_to_point.items()
             if (object_dict[obj_id]['objectType'] in constants.RECEPTACLES and
                 object_dict[obj_id]['objectType'] not in constants.MOVABLE_RECEPTACLES_SET)
             ])
        receptacle_at_location_str = '\n        '.join(receptacle_nearest_point_strs)

        receptacle_can_holds = []
        obj_types_in_scene = [obj['objectType'] for obj in object_dict.values()]
        for recep in receptacles:
            recep_type = object_dict[recep]['objectType']
            for obj in constants.VAL_RECEPTACLE_OBJECTS[recep_type]:
                if obj in obj_types_in_scene:
                    can_hold_str = '(canContain %sType %sType)' % (recep_type, obj)
                    receptacle_can_holds.append(can_hold_str)
        receptacle_can_holds_str = '\n        '.join(receptacle_can_holds)

        extra_facts = self.get_extra_facts()


        # salient_mat_str = '\n        '.join(['(salientMaterials %s %s)' % (obj, ','.join(object_dict[obj]['salientMaterials']))
        #                                      for obj in objects if object_dict[obj]['salientMaterials'] is not None])

        # mass_props_str = '\n        '.join(['(mass %s %s)' % (obj, str(object_dict[obj]['mass']))
        #                                     for obj in objects])

        # temp_props_str = '\n        '.join(['(temp %s %s)' % (obj, object_dict[obj]['ObjectTemperature'])
        #                                     for obj in objects])


        pddl_mid_start = '''
        %s
        %s
        %s
        )
    ''' % (
            object_str,
            receptacle_str,
            location_str,
        )
        pddl_mid_init = '''
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        )
    ''' % (
            receptacle_type_str,
            object_type_str,
            receptacle_can_holds_str,
            # salient_mat_str,
            # mass_props_str,
            # temp_props_str,
            pickupable_str,
            receptacle_objects_str,
            openable_str,
            agent_location_str,
            opened_receptacle_str,
            cleanable_str,
            is_clean_str,
            heatable_str,
            coolable_str,
            is_hot_str,
            is_cool_str,
            toggleable_str,
            is_on_str,
            recep_obj_str,
            sliceable_str,
            is_sliced_str,
            in_receptacle_str,
            was_in_receptacle_str,
            location_distance_str,
            receptacle_at_location_str,
            extra_facts,
        )

        pddl_mid_start = fix_pddl_str_chars(pddl_mid_start)
        pddl_mid_init = fix_pddl_str_chars(pddl_mid_init)

        pddl_str = (pddl_start + '\n' +
                    pddl_mid_start + '\n' +
                    pddl_init + '\n' +
                    pddl_mid_init + '\n' +
                    pddl_goal)

        return pddl_str

def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)



def fix_pddl_str_chars(input_str):
    return py_util.multireplace(input_str,
                                {'-': '_minus_',
                                 '#': '-',
                                 '|': '_bar_',
                                 '+': '_plus_',
                                 '.': '_dot_',
                                 ',': '_comma_'})

action_space = [
    {'action': 'Explore'},
    {'action': 'Scan'},
    {'action': 'Plan'},
    {'action': 'End'},
]



def get_goal_pddl(traj_data):
    goal_type = traj_data['task_type']
    pddl_params = traj_data['pddl_params']
    if pddl_params['object_sliced']:
        goal_type += "_slice"
    goal_str = glib.gdict[goal_type]['pddl']
    goal_str = goal_str.format(obj=pddl_params['object_target'],
                               recep=pddl_params['parent_target'],
                               toggle=pddl_params['toggle_target'],
                               mrecep=pddl_params['mrecep_target'])
    return goal_str



def augment_traj(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # make directories
    root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")

    # fresh images list
    traj_data['images'] = list()

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    print(scene_name)

    env.step(dict(traj_data['scene']['init_action']))
    print("Task: %s" % (traj_data['turk_annotations']['anns'][0]['task_desc']))

    # setup task
    env.set_task(traj_data, args, reward_type='dense')
    game_state = TextWorldTaskGameStateFullKnowledge(env)

    # reset
    game_state.receptacle_to_point = None
    game_state.task_target = None
    game_state.success = False

    # load nav graph
    game_state.gt_graph = graph_obj.Graph(use_gt=True, construct_graph=True, scene_id=scene_num)
    game_state.gt_graph.clear()

    game_state.agent_height = env.last_event.metadata['agent']['position']['y']
    game_state.camera_height = game_state.agent_height + constants.CAMERA_HEIGHT_OFFSET

    points_source = os.path.join(alfworld.gen.__path__[0], 'layouts/%s-openable.json' % scene_name)
    with open(points_source, 'r') as f:
        openable_object_to_point = json.load(f)
    game_state.openable_object_to_point = openable_object_to_point

    pddl_params = traj_data['pddl_params']
    game_state.object_target = constants.OBJECTS.index(pddl_params['object_target']) if pddl_params['object_target'] else None
    game_state.parent_target = constants.OBJECTS.index(pddl_params['parent_target']) if pddl_params['parent_target'] else None
    game_state.mrecep_target = constants.OBJECTS.index(pddl_params['mrecep_target']) if pddl_params['mrecep_target'] else None
    game_state.toggle_target = constants.OBJECTS.index(pddl_params['toggle_target']) if pddl_params['toggle_target'] else None
    game_state.task_target = (game_state.object_target, game_state.parent_target, game_state.toggle_target, game_state.mrecep_target)

    game_state.update_receptacle_nearest_points()
    pddl_str = game_state.state_to_pddl(traj_data)

    pddl_file = os.path.join(os.path.dirname(json_file), 'initial_state.pddl')
    with open(pddl_file, 'w') as f:
        f.write(pddl_str)
    print("Wrote to %s" % pddl_file)

    game_state.planner.process_pool.terminate()


def run():
    '''
    replay loop
    '''
    # start THOR env
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,
                  player_screen_height=IMAGE_HEIGHT)

    skipped_files = []

    while len(traj_list) > 0:
        lock.acquire()
        json_file = traj_list.pop()
        lock.release()

        print ("Augmenting PDDL: " + json_file)
        try:
            augment_traj(env, json_file)
        except ValueError as e:
            import traceback
            traceback.print_exc()
            print ("Error: " + repr(e))
            print ("Skipping " + json_file)
            skipped_files.append(json_file)

    env.stop()
    print("Finished.")

    # skipped files
    if len(skipped_files) > 0:
        print("Skipped Files:")
        print(skipped_files)


traj_list = []
lock = threading.Lock()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="$ALFWORLD_DATA/json_2.1.1")
parser.add_argument('--shuffle', dest='shuffle', action='store_true')
parser.add_argument('--reward_config', type=str, default='alfworld/agents/config/rewards.json')
parser.add_argument('--interactive', action="store_true",
                    help='enable simplistic interactive navigation within the environment')
args = parser.parse_args()

# make a list of all the traj_data json files
data_path = os.path.expandvars(args.data_path)
for dir_name, _, file_list in os.walk(data_path, topdown=False):
    if "trial_" in dir_name and "test" not in dir_name:
        json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
        if not os.path.isfile(json_file):
            continue
        traj_list.append(json_file)


# random shuffle
if args.shuffle:
    random.shuffle(traj_list)

run()
