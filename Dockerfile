FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

RUN apt-get update
RUN apt install sudo
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

# work directory
WORKDIR /home/$USER_NAME

# install system dependencies
COPY ./scripts/install_deps.sh /tmp/install_deps.sh
RUN yes "Y" | /tmp/install_deps.sh

COPY ./scripts/install_nvidia.sh /tmp/install_nvidia.sh
RUN yes "Y" | /tmp/install_nvidia.sh

# setup python environment
RUN cd $WORKDIR
ENV VIRTUAL_ENV=/home/$USER_NAME/alfred_env
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python requirements
RUN pip install --upgrade pip==19.3.1
RUN pip install -U setuptools
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# alfred-tw extras
RUN pip install -U spacy
RUN python -m spacy download en
RUN pip install https://github.com/MarcCote/downward/archive/faster_replan.zip
RUN pip install https://github.com/MohitShridhar/TextWorld/archive/handcoded_expert_integration.zip

# download thor
#RUN python3 -c "import ai2thor.controller; ai2thor.controller.Controller(download_only=True)"

# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils && \
   rm -rf /var/lib/apt/lists/*


###### COMMENT OUT THIS BLOCK IF USING NVIDIA-DOCKER1 ###########
# SOURCE: https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/master/docker/pytorch-dense-correspondence.dockerfile
# needed to get OpenGL running inside the docker
# https://github.com/machinekoder/nvidia-opengl-docker

# optional, if the default user is not "root", you might need to switch to root here and at the end of the script to the original user again.
# e.g.
# USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev && \
    rm -rf /var/lib/apt/lists/*

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
# e.g. nvidia/opengl:1.1-glvnd-runtime-ubuntu16.04)
COPY --from=machinekoder/nvidia-opengl-docker:1.1-glvnd-runtime-stretch \
  /usr/local/lib/x86_64-linux-gnu \
  /usr/local/lib/x86_64-linux-gnu

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
# e.g. nvidia/opengl:1.1-glvnd-runtime-ubuntu16.04
COPY --from=machinekoder/nvidia-opengl-docker:1.1-glvnd-runtime-stretch \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
    ldconfig && \
    echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
    echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# USER original_user
###### END BLOCK ###########



# change ownership of everything to our user
RUN mkdir /home/$USER_NAME/alfred_tw
RUN cd ${USER_HOME_DIR} && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .

ENTRYPOINT bash -c "export ALFRED_ROOT=~/alfred_tw && /bin/bash"