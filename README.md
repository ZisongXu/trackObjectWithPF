# trackObjectWithPF
Ubuntu20 ROS:noetic Pybullet
  1. Go to this file
  2. Run ./build.sh in the terminal
  3. Run ./run.sh in the terminal

How to run the panda robot in the real world?
  1. Go into the container
  2. [TrackObjectWithPF] xterm
  3. [TrackObjectWithPF xterm1] roslaunch panda_moveit moveit_with_rviz.launch
  4. [TrackObjectWithPF] xterm
  5. [TrackObjectWithPF xterm2] cd ~/catkin_ws/src/panda_moveit/scripts
  6. [TrackObjectWithPF xterm2] python3 demo_talker_real_robot.py
  How to run the PF code
  7. [TrackObjectWithPF] xterm
  8. [TrackObjectWithPF xterm3] cd ~/phd_project/code
  9. [TrackObjectWithPF xterm3] python3 franka_robot_realrobot_test.py
