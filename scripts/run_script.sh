#!/bin/bash
source /opt/ros/noetic/setup.bash

if [ ! -d "$HOME/catkin_ws/devel" ]; then
    cd $HOME/catkin_ws/src
    git clone https://github.com/ros-planning/moveit.git
    git clone https://github.com/ros-planning/panda_moveit_config.git
    git clone https://github.com/andreasBihlmaier/pysdf.git
    git clone https://github.com/andreasBihlmaier/gazebo2rviz.git
    cd panda_moveit_config
    git checkout noetic-devel
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

