#!/usr/bin/env python
from __future__ import print_function

#########
# Credit: https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/docker/docker_run.py
#########

import argparse
import os
import socket
import getpass
import yaml

if __name__=="__main__":
    user_name = getpass.getuser()
    default_image_name = user_name + '-alfworld'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str,
                        help="(required) name of the image that this container is derived from", default=default_image_name.lower())

    parser.add_argument("-c", "--container", type=str, default="alfworld", help="(optional) name of the container")

    parser.add_argument("-d", "--data", type=str, default="data/", help="(optional) external data directory")

    parser.add_argument("-hl", "--headless",  action='store_true', help="(optional) run in headless mode")

    parser.add_argument("-g", "--gpus", type=str, default="all", help="(optional) gpus for nvidia docker")

    parser.add_argument("-dr", "--dry_run", action='store_true', help="(optional) perform a dry_run, print the command that would have been executed but don't execute it.")

    parser.add_argument("-p", "--passthrough", type=str, default="", help="(optional) extra string that will be tacked onto the docker run command, allows you to pass extra options. Make sure to put this in quotes and leave a space before the first character")

    args = parser.parse_args()
    print("running docker container derived from image %s" %args.image)
    source_dir = os.getcwd()

    image_name = args.image
    home_directory = '/home/' + user_name

    cmd = "xhost +local:root \n" if not args.headless else ""
    cmd += "docker run "
    if args.container:
        cmd += " --name %(container_name)s " % {'container_name': args.container}

    # gpus
    cmd += " --gpus %s" % (args.gpus)

    # display
    if args.headless:
        cmd += " -v /usr/bin/nvidia-xconfig:/usr/bin/nvidia-xconfig "
    else:
        cmd += " -e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw "    # enable graphics

    # bindings
    cmd += " -v %(source_dir)s:%(home_directory)s/alfworld " \
           % {'source_dir': source_dir, 'home_directory': home_directory}                  # mount source
    cmd += " -v ~/.ssh:%(home_directory)s/.ssh " % {'home_directory': home_directory}      # mount ssh keys
    cmd += " -v ~/.torch:%(home_directory)s/.torch " % {'home_directory': home_directory}  # mount torch folder

    cmd += " --user %s " % user_name                                                       # login as current user

    # custom data path
    cmd += " -v %s:/data " %(os.path.join(source_dir, args.data))

    cmd += "-v /usr/bin/nvidia-xconfig:/usr/bin/nvidia-xconfig "
    cmd += "--device=/dev/tty0:/dev/tty0 "

    # expose UDP ports
    cmd += " -p 8888:8888 "
    cmd += " --ipc=host "

    # share host machine network
    cmd += " --network=host "

    cmd += " " + args.passthrough + " "

    cmd += " --privileged"

    cmd += " --rm " # remove the image when you exit

    cmd += "-it "
    cmd += args.image
    cmd_endxhost = "xhost -local:root"

    print("command = \n \n", cmd, "\n", cmd_endxhost)
    print("")

    # build the docker image
    if not args.dry_run:
        print("executing shell command")
        code = os.system(cmd)
        print("Executed with code ", code)
        if not args.headless:
            os.system(cmd_endxhost)
        # Squash return code to 0/1, as
        # Docker's very large return codes
        # were tricking Jenkins' failure
        # detection
        exit(code != 0)
    else:
        print("dry run, not executing command")
        exit(0)
