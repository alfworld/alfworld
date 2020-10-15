# ALFWorld

[<b>Aligning Text and Embodied Environments for Interactive Learning</b>](https://arxiv.org/abs/1912.01734)  
[Mohit Shridhar](https://mohitshridhar.com/), [Xingdi (Eric) Yuan](https://xingdi-eric-yuan.github.io/), [Marc-Alexandre Côté](https://www.microsoft.com/en-us/research/people/macote/),   
[Yonatan Bisk](https://yonatanbisk.com/), [Adam Trischler](https://www.microsoft.com/en-us/research/people/adtrisch/), [Matthew Hausknecht](https://mhauskn.github.io/)

**ALFWorld** contains interactive TextWorld environments (Côté et. al) that parallel embodied worlds in the ALFRED dataset (Shridhar et. al). The aligned environments allow agents to reason and learn high-level policies in an abstract space before solving embodied tasks through low-level actuation.  

For the latest updates, see: [**alfworld.github.io**](https://askforalfred.com)

<img width="300" alt="portfolio_view" src="media/alfworld_teaser.png">


## Quickstart

Clone repo:
```bash
$ git clone https://github.com/askforalfred/alfred.git alfred
$ export ALFRED_ROOT=$(pwd)/alfred
```

Install requirements:
```bash
$ virtualenv -p $(which python3) --system-site-packages alfred_env # or whichever package manager you prefer
$ source alfred_env/bin/activate

$ cd $ALFRED_ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Download Trajectory JSONs and Resnet feats (~17GB):
```bash
$ cd $ALFRED_ROOT/data
$ sh download_data.sh json_feat
```

Train models:
```bash
$ cd $ALFRED_ROOT
$ python models/train/train_seq2seq.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```

## More Info 

- [**Dataset**](data/): Downloading full dataset, Folder structure, JSON structure.
- [**Models**](models/): Training and Evaluation, File structure, Pre-trained models.
- [**Data Generation**](gen/): Generation, Replay Checks, Data Augmentation (high-res, depth, segementation masks etc).
- [**FAQ**](doc/FAQ.md): Frequently Asked Questions. 

## Prerequisites

- Python 3
- PyTorch 1.1.0
- Torchvision 0.3.0
- AI2THOR 2.1.0

See [requirements.txt](requirements.txt) for all prerequisites

## Hardware 

Tested on:
- **GPU** - GTX 1080 Ti (12GB)
- **CPU** - Intel Xeon (Quad Core)
- **RAM** - 16GB
- **OS** - Ubuntu 16.04


## Leaderboard

Run your model on test seen and unseen sets, and create an action-sequence dump of your agent:

```bash
$ cd $ALFRED_ROOT
$ python models/eval/leaderboard.py --model_path <model_path>/model.pth --model models.model.seq2seq_im_mask --data data/json_feat_2.1.0 --gpu --num_threads 5
```

This will create a JSON file, e.g. `task_results_20191218_081448_662435.json`, inside the `<model_path>` folder. Submit this JSON here: [AI2 ALFRED Leaderboard](https://leaderboard.allenai.org/alfred/submissions/public). For rules and restrictions, see the [getting started page](https://leaderboard.allenai.org/alfred/submissions/get-started).

## Docker Setup

Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster). 

Modify [docker_build.py](scripts/docker_build.py) and [docker_run.py](scripts/docker_run.py) to your needs.

#### Build 

Build the image:

```bash
$ python scripts/docker_build.py 
```

#### Run (Local)

For local machines:

```bash
$ python scripts/docker_run.py
 
  source ~/alfred_env/bin/activate
  cd $ALFRED_ROOT
```

#### Run (Headless)

For headless VMs and Cloud-Instances:

```bash
$ python scripts/docker_run.py --headless 

  # inside docker
  tmux new -s startx  # start a new tmux session

  # start nvidia-xconfig (might have to run this twice)
  sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
  sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

  # start X server on DISPLAY 0
  sudo python ~/alfred/scripts/startx.py 0  # if this throws errors e.g "(EE) Server terminated with error (1)" or "(EE) already running ..." try a display > 0

  # detach from tmux shell
  # Ctrl+b then d

  # source env
  source ~/alfred_env/bin/activate
  
  # set DISPLAY variable to match X server
  export DISPLAY=:0

  # check THOR
  cd $ALFRED_ROOT
  python scripts/check_thor.py

  ###############
  ## (300, 300, 3)
  ## Everything works!!!
```

You might have to modify `X_DISPLAY` in [gen/constants.py](gen/constants.py) depending on which display you use.

## Cloud Instance

ALFRED can be setup on headless machines like AWS or GoogleCloud instances. 
The main requirement is that you have access to a GPU machine that supports OpenGL rendering. Run the [startx.py](scripts/startx.py) script
to examine the GPU devices on the host, generate a xorg.conf file, and then start X. You should be able to run AI2THOR normally for evaluation purposes. 
By default, the `:0.0` display will be used, but if you are running on a machine with more than one GPU, you can address 
these by modifying the screen component of the display. So `:0.0` refers to the first device, `:0.1` the second and so on.

Also, checkout this guide: [Setting up THOR on Google Cloud](https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)

## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

## License

MIT License

## Change Log

07/04/2020:
- Updated download links. Switched from Google Cloud to AWS. Old download links will be deactivated.


28/03/2020:
- Updated the mask-interaction API to use IoU scores instead of max pixel count for selecting objects.
- Results table in the paper will be updated with new numbers.

## Contact

Questions or issues? Contact [askforalfred@googlegroups.com](askforalfred@googlegroups.com)