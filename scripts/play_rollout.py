#!/usr/bin/env python

import io
import os
import copy
import glob
import json
import math
import shutil
import random
import argparse
from os.path import join as pjoin


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from ai2thor.controller import Controller
import alfworld.gen.constants as constants

import alfworld.agents
from alfworld.info import ALFWORLD_DATA
from alfworld.env.thor_env import ThorEnv
from alfworld.agents.detector.mrcnn import load_pretrained_model
from alfworld.agents.controller import OracleAgent, OracleAStarAgent, MaskRCNNAgent, MaskRCNNAStarAgent


class ThorPositionTo2DFrameTranslator(object):
   def __init__(self, frame_shape, cam_position, orth_size):
      self.frame_shape = frame_shape
      self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
      self.span = 2 * orth_size

   def __call__(self, position):
      if len(position) == 3:
            x, _, z = position
      else:
            x, z = position

      camera_position = (np.array((x, z)) - self.lower_left) / self.span
      return np.array(
            (
               round(self.frame_shape[0] * (1.0 - camera_position[1])),
               round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
      )


def position_to_tuple(position):
   return (position["x"], position["y"], position["z"])


def get_agent_map_data(c: Controller):
   c.step({"action": "ToggleMapView"})
   cam_position = c.last_event.metadata["cameraPosition"]
   cam_orth_size = c.last_event.metadata["cameraOrthSize"]
   pos_translator = ThorPositionTo2DFrameTranslator(
      c.last_event.frame.shape, position_to_tuple(cam_position), cam_orth_size
   )
   to_return = {
      "frame": c.last_event.frame,
      "cam_position": cam_position,
      "cam_orth_size": cam_orth_size,
      "pos_translator": pos_translator,
   }
   c.step({"action": "ToggleMapView"})
   return to_return


def add_agent_view_triangle(
      position, rotation, frame, pos_translator, scale=1.0, opacity=0.7
   ):
      p0 = np.array((position[0], position[2]))
      p1 = copy.copy(p0)
      p2 = copy.copy(p0)

      theta = -2 * math.pi * (rotation / 360.0)
      rotation_mat = np.array(
         [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
      )
      offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
      offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

      p1 += np.matmul(rotation_mat, offset1)
      p2 += np.matmul(rotation_mat, offset2)

      img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
      img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

      opacity = int(round(255 * opacity))  # Define transparency for the triangle.
      points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
      draw = ImageDraw.Draw(img2)
      draw.polygon(points, fill=(255, 255, 255, opacity))

      img = Image.alpha_composite(img1, img2)
      return np.array(img.convert("RGB"))


def setup_scene(env, traj_data, r_idx, args, reward_type='dense'):
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name, make_agents_visible=args.view)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty, make_agents_visible=args.view)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))

    # print goal instr
    print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    # setup task for reward
    env.set_task(traj_data, args, reward_type=reward_type)


def main(args):
    #args.problem = "/home/macote/.cache/alfworld/json_2.1.1/valid_unseen/look_at_obj_in_light-AlarmClock-None-DeskLamp-308/trial_T20190908_222917_366542"
    print(f"Playing '{args.problem}'.")

    name = f"{args.problem.split('json_2.1.1/')[-1].replace('/', '__')}"
    print(f"Trained agent: /data/src/alfworld/LFM_trajectories/alfworld/intent/default_0.5/inferred_feedback/{args.problem.split('json_2.1.1/')[-1].replace('/', '__')}.json")
    with open(f"/data/src/alfworld/LFM_trajectories/alfworld/intent/default_0.5/inferred_feedback/{args.problem.split('json_2.1.1/')[-1].replace('/', '__')}.json") as f:
        actions_to_follow = json.load(f)

    # start THOR
    env = ThorEnv()

    # load traj_data
    root = args.problem
    json_file = os.path.join(root, 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)

    # setup scene
    setup_scene(env, traj_data, 0, args)

    if not args.ground_truth:
        # choose controller
        if args.controller == "oracle":
            AgentModule = OracleAgent
            agent = AgentModule(env, traj_data, traj_root=root, load_receps=args.load_receps, debug=args.debug)
        elif args.controller == "oracle_astar":
            AgentModule = OracleAStarAgent
            agent = AgentModule(env, traj_data, traj_root=root, load_receps=args.load_receps, debug=args.debug)
        elif args.controller == "mrcnn":
            AgentModule = MaskRCNNAgent
            mask_rcnn = load_pretrained_model(pjoin(ALFWORLD_DATA, "detectors", "mrcnn.pth"))
            agent = AgentModule(env, traj_data, traj_root=root,
                                pretrained_model=mask_rcnn,
                                load_receps=args.load_receps, debug=args.debug)
        elif args.controller == "mrcnn_astar":
            AgentModule = MaskRCNNAStarAgent
            mask_rcnn = load_pretrained_model(pjoin(ALFWORLD_DATA, "detectors", "mrcnn.pth"))
            agent = AgentModule(env, traj_data, traj_root=root,
                                pretrained_model=mask_rcnn,
                                load_receps=args.load_receps, debug=args.debug)
        else:
            raise NotImplementedError()

        print(agent.feedback)

    savename = "playthrough"
    if args.ground_truth:
        actions_to_follow = traj_data["plan"]["low_actions"]
        savename += "_oracle"

    shutil.rmtree(savename)
    os.makedirs(savename, exist_ok=True)
    os.makedirs(savename + "_pov", exist_ok=True)


    import plotly.graph_objects as go

    # Create a list to store the frames
    frames = []
    traces = []
    old_position = position_to_tuple(env.last_event.metadata["agent"]["position"])
    for i, step in enumerate(actions_to_follow):

        if args.ground_truth:
            cmd = step["api_action"]["action"]
            env.step(step["api_action"])
        else:
            cmd = step["action"]
            agent.step(cmd)

        if args.view:
            t = get_agent_map_data(env)
            new_position = position_to_tuple(env.last_event.metadata["agent"]["position"])
            new_frame = add_agent_view_triangle(
                new_position,
                env.last_event.metadata["agent"]["rotation"]["y"],
                t["frame"],
                t["pos_translator"],
            )

            # Remove ticks and labels
            fig,ax = plt.subplots(1)
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax.axis('off')

            # Add a gray transluice band at the bottom of the image with the action taken to the frame
            new_frame[-40:, :] = 0.5 * new_frame[-40:, :] + 0.5 * 255
            ax.text(new_frame.shape[1] // 2, new_frame.shape[0] - 10, cmd, fontsize=18, color="black", ha='center', wrap=True)
            ax.imshow(new_frame)

            # Keep the lines and add them for subsequent images.
            traces.append([t["pos_translator"](old_position), t["pos_translator"](new_position)])

            for i, trace in enumerate(traces):
                color = (i / len(traces), 0, 1 - i / len(traces))  # Gradient coloring based on index
                ax.plot([trace[0][1], trace[1][1]], [trace[0][0], trace[1][0]], color=color, linewidth=3)

            #plt.show(block=False)
            # Save the figure
            fig.savefig(f"./{savename}/step_{i}.png", transparent=True, bbox_inches='tight', pad_inches=0)

            # POV
            fig,ax = plt.subplots(1)
            fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax.axis('off')

            # Add a gray transluice band at the bottom of the image with the action taken to the frame
            frame = env.last_event.frame.copy()
            frame[-40:, :] = 0.5 * frame[-40:, :] + 0.5 * 255
            ax.text(frame.shape[1] // 2, frame.shape[0] - 10, cmd, fontsize=14, color="black", ha='center', wrap=True)

            frame[:40, :] = 0.5 * frame[:40, :] + 0.5 * 255
            color = "green" if step["llm_pred"].lower().startswith("yes") else "red"
            ax.text(frame.shape[1] // 2, 10, step["llm_pred"], fontsize=14, color=color, ha='center', va='top', wrap=True)

            ax.imshow(frame)
            fig.savefig(f"./{savename}_pov/step_{i}.png")#, bbox_inches='tight', pad_inches=0)

            old_position = new_position

            # # POV

            # # Add a gray translucent band at the bottom of the image with the action taken to the frame
            # frame = env.last_event.frame.copy()
            # frame[-40:, :] = 0.5 * frame[-40:, :] + 0.5 * 255
            # fig = go.Figure(go.Image(z=frame))
            # #fig.add(), xref="x", yref="y", x=0, y=0, sizex=frame.shape[1], sizey=frame.shape[0], sizing="stretch", opacity=1, layer="below"))

            # fig.add_annotation( x=frame.shape[1] // 2, y=frame.shape[0] - 10, text=cmd, font=dict(size=14, color="black"), showarrow=False, xanchor="center", yanchor="middle")

            # fig.add_annotation( x=frame.shape[1] // 2, y=10, text=step["llm_pred"], font=dict(size=14, color="green" if step["llm_pred"].lower().startswith("yes") else "red"), showarrow=False, xanchor="center", yanchor="bottom")

            # # Add the frame to the list of frames
            # img_bytes = fig.to_image(format="png", width=frame.shape[1], height=frame.shape[0], scale=1)
            # img = Image.open(io.BytesIO(img_bytes))
            # img.save(f"./{savename}_pov2/step_{i}.png")

            # frames.append(fig)

        done = env.get_goal_satisfied()
        if done:
            print("You won!")
            break

    if args.view:
        # Make a video from the images
        os.system(f"ffmpeg -r 1 -i ./{savename}/step_%d.png -vcodec mpeg4 -y ./{name}_top.mp4")
        os.system(f"ffmpeg -r 1 -i ./{savename}_pov/step_%d.png -vcodec mpeg4 -y ./{name}.mp4")
        # os.system(f"ffmpeg -r 1 -i ./{savename}_pov2/step_%d.png -vcodec mpeg4 -y ./{savename}_pov2.mp4")

        # # Create the animation
        # animation = go.Figure(frames=[go.Frame(data=frame) for frame in frames])

        # # Save the animation as HTML
        # animation.write_html(f"./{savename}_pov/animation.html")


if __name__ == "__main__":
    description = "Play the abstract text version of an ALFRED environment."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("problem", nargs="?", default=None,
                        help="Path to a folder containing PDDL and traj_data files."
                             f"Default: pick one at random found in {ALFWORLD_DATA}")
    parser.add_argument("--controller", default="oracle", choices=["oracle", "oracle_astar", "mrcnn", "mrcnn_astar"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--load_receps', action="store_true")
    parser.add_argument('--reward_config', type=str, default=pjoin(alfworld.agents.__path__[0], 'config', 'rewards.json'))
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--ground-truth", action="store_true")
    args = parser.parse_args()

    if args.problem is None:
        problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
        args.problem = os.path.dirname(random.choice(problems))

    main(args)
