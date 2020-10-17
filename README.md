# ALFWorld

[<b>Aligning Text and Embodied Environments for Interactive Learning</b>](https://arxiv.org/abs/2010.03768)  
[Mohit Shridhar](https://mohitshridhar.com/), [Xingdi (Eric) Yuan](https://xingdi-eric-yuan.github.io/), [Marc-Alexandre Côté](https://www.microsoft.com/en-us/research/people/macote/),   
[Yonatan Bisk](https://yonatanbisk.com/), [Adam Trischler](https://www.microsoft.com/en-us/research/people/adtrisch/), [Matthew Hausknecht](https://mhauskn.github.io/)

**ALFWorld** contains interactive TextWorld environments (Côté et. al) that parallel embodied worlds in the ALFRED dataset (Shridhar et. al). The aligned environments allow agents to reason and learn high-level policies in an abstract space before solving embodied tasks through low-level actuation.  

:exclamation: : **Work in progress**.

For the latest updates, see: [**alfworld.github.io**](https://alfworld.github.io)

## Quickstart

Clone repo:
```bash
$ git clone https://github.com/alfworld/alfworld.git alfworld
$ export ALFRED_ROOT=$(pwd)/alfworld
```

Install requirements:
```bash
# Note: Requires python 3.6 or higher 
$ virtualenv -p $(which python3.6) --system-site-packages alfworld_env # or whichever package manager you prefer
$ source alfworld_env/bin/activate

$ cd $ALFRED_ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Download PDDL & Game Files and pre-trained MaskRCNN detector:
```bash
$ sh $ALFRED_ROOT/data/download_data.sh
```

Train models:
```bash
$ cd $ALFRED_ROOT/agents
$ python dagger/train_dagger.py config/base_config.yaml
```

Play around with [TextWorld and THOR demos](scripts/).

## More Info 

- [**Data**](data/): PDDL and Game Files of ALFRED Tasks. Generating PDDL states and detection training images.
- [**Agents**](agents/): Training and evaluating TextDAgger, TextDQN, VisionDAgger agents.
- [**Explore**](scripts/): Play around with ALFWorld TextWorld and THOR environments.

## Prerequisites

- Python 3.6
- PyTorch 1.2.0
- Torchvision 0.4.0
- AI2THOR 2.1.0

See [requirements.txt](requirements.txt) for all prerequisites

## Hardware 

Tested on:
- **GPU** - GTX 1080 Ti (12GB)
- **CPU** - Intel Xeon (Quad Core)
- **RAM** - 16GB
- **OS** - Ubuntu 16.04


## Docker Setup

Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster). 

Modify [docker_build.py](scripts/docker_build.py) and [docker_run.py](scripts/docker_run.py) to your needs.

#### Build 

Build the image:

```bash
$ python docker/docker_build.py 
```

#### Run (Local)

For local machines:

```bash
$ python docker/docker_run.py
 
  source ~/alfworld_env/bin/activate
  cd $ALFRED_ROOT
```

#### Run (Headless)

For headless VMs and Cloud-Instances:

```bash
$ python docker/docker_run.py --headless 

  # inside docker
  tmux new -s startx  # start a new tmux session

  # start nvidia-xconfig (might have to run this twice)
  sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
  sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

  # start X server on DISPLAY 0
  sudo python ~/alfworld/docker/startx.py 0  # if this throws errors e.g "(EE) Server terminated with error (1)" or "(EE) already running ..." try a display > 0

  # detach from tmux shell
  # Ctrl+b then d

  # source env
  source ~/alfworld_env/bin/activate
  
  # set DISPLAY variable to match X server
  export DISPLAY=:0

  # check THOR
  cd $ALFRED_ROOT
  python docker/check_thor.py

  ###############
  ## (300, 300, 3)
  ## Everything works!!!
```

You might have to modify `X_DISPLAY` in [gen/constants.py](gen/constants.py) depending on which display you use.

## Cloud Instance

ALFWorld can be setup on headless machines like AWS or GoogleCloud instances. 
The main requirement is that you have access to a GPU machine that supports OpenGL rendering. Run [startx.py](scripts/startx.py) in a tmux shell:
```bash
# start tmux session
$ tmux new -s startx 

# start X server on DISPLAY 0
$ sudo python $ALFRED_ROOT/scripts/startx.py 0  # if this throws errors e.g "(EE) Server terminated with error (1)" or "(EE) already running ..." try a display > 0

# detach from tmux shell
# Ctrl+b then d

# set DISPLAY variable to match X server
$ export DISPLAY=:0

# check THOR
$ cd $ALFRED_ROOT
$ python docker/check_thor.py

###############
## (300, 300, 3)
## Everything works!!!
```

You might have to modify `X_DISPLAY` in [gen/constants.py](gen/constants.py) depending on which display you use.

Also, checkout this guide: [Setting up THOR on Google Cloud](https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)

## Citations

**ALFWorld**
```
@inproceedings{ALFWorld20,
  title ={{ALFWorld: Aligning Text and Embodied
           Environments for Interactive Learning}},
  author={Mohit Shridhar and Xingdi Yuan and
          Marc-Alexandre C\^ot\'e and Yonatan Bisk and
          Adam Trischler and Matthew Hausknecht},
  booktitle = {arXiv},
  year = {2020},
  url = {https://arxiv.org/abs/2010.03768}
}
```  

**ALFRED**
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

**TextWorld**
```
@inproceedings{cote2018textworld,
  title={Textworld: A learning environment for text-based games},
  author={C{\^o}t{\'e}, Marc-Alexandre and K{\'a}d{\'a}r, {\'A}kos and Yuan, Xingdi and Kybartas, Ben and Barnes, Tavian and Fine, Emery and Moore, James and Hausknecht, Matthew and El Asri, Layla and Adada, Mahmoud and others},
  booktitle={Workshop on Computer Games},
  pages={41--75},
  year={2018},
  organization={Springer}
}
```

## License

MIT License

## Contact

Questions or issues? File an issue or contact [Mohit Shridhar](https://mohitshridhar.com)