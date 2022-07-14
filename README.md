# trackObjectWithPF
Ubuntu20 ROS:noetic Pybullet
  1. Go to this file
  2. Run ./build.sh in the terminal
  3. Run ./run.sh in the terminal

How to run the panda robot in the real world?
  1. Go into the container
  2. [PandaPlanning] xterm
  3. [PandaPlanning xterm1] roslaunch panda_moveit moveit_with_rviz.launch
  4. [PandaPlanning] xterm
  5. [PandaPlanning xterm2] cd ~/catkin_ws/src/panda_moveit/scripts
  6. [PandaPlanning xterm2] python3 demo_talker_real_robot.py
