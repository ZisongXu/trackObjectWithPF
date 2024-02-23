#!/bin/bash
source /opt/ros/noetic/setup.bash

if [ ! -d "$HOME/catkin_ws/devel" ]; then
    cd $HOME/catkin_ws/src
    git clone https://github.com/roboticsleeds/moveit.git; cd moveit; git checkout noetic-devel; cd ..
    git clone https://github.com/roboticsleeds/moveit_msgs.git
    git clone https://github.com/roboticsleeds/panda_moveit_config.git
    git clone https://github.com/roboticsleeds/franka_ros.git
    cd $HOME/catkin_ws/
    catkin_make
    pip install pyquaternion
    pip3 install pybullet
    pip3 install pandas==2.0.0
    pip3 install scikit-surgerycore
    pip3 install seaborn==0.12.2
    pip3 install matplotlib==3.7.1
    pip install --upgrade pip
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

source $HOME/catkin_ws/devel/setup.bash
source $HOME/.bashrc
alias xterm="xterm -fa 'Monospace' -fs 14 &"

