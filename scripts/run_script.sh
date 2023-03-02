#!/bin/bash
source /opt/ros/noetic/setup.bash

if [ ! -d "$HOME/catkin_ws/devel" ]; then
    cd $HOME/catkin_ws/src
    git clone git@github.com:roboticsleeds/moveit.git; cd moveit; git checkout noetic-devel; cd ..
    git clone git@github.com:roboticsleeds/moveit_msgs
    git clone git@github.com:roboticsleeds/panda_moveit_config.git
    git clone git@github.com:roboticsleeds/franka_ros.git
    cd $HOME/catkin_ws/
    catkin_make
    pip install pyquaternion
    pip3 install pybullet
    pip3 install pandas
    pip3 install scikit-surgerycore
    pip3 install seaborn
fi

source $HOME/catkin_ws/devel/setup.bash
source $HOME/.bashrc
alias xterm="xterm -fa 'Monospace' -fs 14 &"

