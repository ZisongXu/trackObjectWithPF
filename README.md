# trackObjectWithPF
Ubuntu20 ROS:noetic Pybullet
  1. Go to this file
  2. Run ./build.sh in the terminal
  3. Run ./run.sh in the terminal

How to run the panda robot in the real world:
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
  
How to run the DOPE:
  1. Enter into the container
  2. [realsense] roslaunch panda_calibration with_ar_markers.launch 
  OR [DOPE] roslaunch realsense2_camera rs_camera.launch filters:=pointcloud enable_infra1:=false enable_infra2:=false
  or [realsense] roslaunch panda_camera_launchers realsense.launch
  3. [DOPE] roslaunch dope dope.launch
  4. [DOPE] rosrun rviz rviz
  5. [DOPE] rosrun dope convert_dope_2_tf.py
  6. In the RViz:
    1. [Global Options]
      6.1.1. [Fixed Frame] panda_link0
    6.2. ADD [Camera]
      6.2.1. [Image Topic] /dope/rgb_points
    6.3. ADD [TF]
      6.3.1. [Frames]
  
Some commands:
  rosbag:
  rosbag record -a
  rosbag play "file_name".bag
  
  rosparam:
  rosparam list
  rosparam get /DOPE_object_names
