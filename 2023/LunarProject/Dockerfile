#=============================================================================
## PROJECT CHRONO - http://projectchrono.org
##
## Copyright (c) 2023 projectchrono.org
## All rights reserved.
##
## Use of this source code is governed by a BSD-style license that can be found
## in the LICENSE file at the top level of the distribution and at
## http://projectchrono.org/license-chrono.txt.
##
## =============================================================================
## Authors: Thomas Liang
## =============================================================================
FROM uwsbel/packages:ubuntu

#####################################################
# Evironmental variables
#####################################################
ENV DISPLAY=:1 \
    VNC_PORT=5901 \
    NO_VNC_PORT=6901 \
    HOME=/sbel \
    TERM=xterm \
    STARTUPDIR=/dockerstartup \
    NO_VNC_HOME=/sbel/noVNC \
    DEBIAN_FRONTEND=noninteractive \
    VNC_COL_DEPTH=24 \
    VNC_RESOLUTION=1600x900 \
    VNC_PW=sbel
EXPOSE $VNC_PORT $NO_VNC_PORT

RUN apt update && apt install -y locales && locale-gen "en_US.UTF-8"
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# #####################################################
# # ROS Install
# #####################################################
# RUN apt update && apt install -y software-properties-common && add-apt-repository universe -y && apt install -y curl \
#     && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
#     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
#     && apt update && apt upgrade -y && apt install -y ros-humble-desktop python3-rosdep python3-colcon-common-extensions 

# #####################################################
# # ROS Tests
# #####################################################
# RUN mkdir -p $HOME/ros-src && cd $HOME/ros-src && git clone https://github.com/ros/ros_tutorials.git -b humble-devel \
#     && cd $HOME && rosdep init && rosdep update && rosdep install -i --from-path $HOME/ros-src --rosdistro humble -y \
#     && cd $HOME && . /opt/ros/humble/setup.sh && colcon build
    
#####################################################
# Chrono Dependencies
#####################################################
RUN export LIB_DIR="lib" && export IOMP5_DIR="" \
    && apt-get update && apt-get -y install unzip wget python3 python3-pip \
    git cmake ninja-build doxygen libvulkan-dev pkg-config libirrlicht-dev \
    freeglut3-dev mpich libasio-dev libboost-dev libglfw3-dev libglm-dev openmpi-common libopenmpi-dev \
    libglew-dev libtinyxml2-dev swig python3-dev libhdf5-dev libnvidia-gl-530 libxxf86vm-dev \
    && ldconfig && apt-get autoclean -y && apt-get autoremove -y

#####################################################
# Build Chrono and Install
#####################################################
ADD buildChrono.sh /
ADD chrono-internal $HOME/Desktop/chrono
# RUN chmod +x /buildChrono.sh && bash /buildChrono.sh

#####################################################
# Visualization and GUI
#####################################################
RUN apt-get update && apt-get install -y net-tools bzip2 procps python3-numpy
    # TigerVNC
RUN apt-get update && apt-get install -y tigervnc-standalone-server \
    && printf '\n# sbel-docker:\n$localhost = "no";\n1;\n' >>/etc/tigervnc/vncserver-config-defaults \
    # noVNC
    && mkdir -p $NO_VNC_HOME/utils/websockify \
    && wget -qO- https://github.com/novnc/noVNC/archive/refs/tags/v1.3.0.tar.gz | tar xz --strip 1 -C $NO_VNC_HOME \
    && wget -qO- https://github.com/novnc/websockify/archive/refs/tags/v0.10.0.tar.gz | tar xz --strip 1 -C $NO_VNC_HOME/utils/websockify \ 
    && ln -s $NO_VNC_HOME/vnc_lite.html $NO_VNC_HOME/index.html \
    # XFCE
    && apt-get install -y supervisor xfce4 xfce4-terminal xterm dbus-x11 libdbus-glib-1-2 \
    && apt-get autoclean -y && apt-get autoremove -y \
    # Ensure $STARTUPDIR exists
    && mkdir $STARTUPDIR

#####################################################
# Startup and Cleanup
#####################################################
ADD ./src/ $HOME/src/
ADD ./desktop/ $HOME/Desktop/
# ADD bashrc $HOME/.bashrc
# RUN mkdir $HOME/Desktop/chrono/chrono_sensor_ros_node
# ADD ./chrono_sensor_ros_node/ $HOME/Desktop/chrono/chrono_sensor_ros_node
RUN chmod a+x $HOME/src/vnc_startup.sh $HOME/src/wm_startup.sh \
    && rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1
    # && mkdir $HOME/ros-src/image_subscriber/
# ADD streamer.py $HOME/ros-src/image_subscriber/
WORKDIR /sbel
ENTRYPOINT ["/sbel/src/vnc_startup.sh"]
