# Frequently Asked Questions


### Are the step-by-step instructions aligned with subgoals?

Yes, each step-by-step instruction has a corresponding subgoal in the training and validation trajectories. If you use this alignment during training, please see the [submission guidelines](https://leaderboard.allenai.org/alfred/submissions/get-started) for leaderboard submissions. 

### Getting 100% success rate with ground-truth trajectories

You should be able to achieve **>99%** success rate on training and validation tasks with the ground-truth actions and masks from the dataset. Occasionally, some [non-determistic behavior](https://github.com/askforalfred/alfred/issues/19) in THOR can lead to failures, but they are extremely rare. 

### Can you train an agent without mask prediction?

Mask prediction is an important part of the ALFRED challenge. Unlike non-interactive environments (e.g vision-language navigation), here it's necessary for the agent to specify *what* exactly it wants to interact with.

### Why do `feat_conv.pt` in [Full Dataset](https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z) have 10 more frames than the number of images?

The last 10 frames are copies of the features from the last image frame. 