# trackObjectWithPF
Ubuntu20; ROS:noetic; Pybullet
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

How to run the OptiTrack:
  1. In the VM16: rosrun natnet_ros ...
  2. In the server [DOPE]: rosrun dope_utilities read_dope_objects_to_param_server.py
  3. In the VM16: roslaunch dope_utilities dope_convertions.launch


How to run the DOPE:
  1. Enter into the container
  2. [realsense] roslaunch panda_calibration with_ar_markers.launch 
  OR [DOPE] roslaunch realsense2_camera rs_camera.launch filters:=pointcloud enable_infra1:=false enable_infra2:=false
  or [realsense] roslaunch panda_camera_launchers realsense.launch
  3. [DOPE] roslaunch dope dope.launch
  4. [DOPE] rosrun rviz rviz
  5. [DOPE] rosrun dope convert_dope_2_tf.py
  6. In the RViz:
    
    [Global Options]
      [Fixed Frame] panda_link0
    ADD [Camera]
      [Image Topic] /dope/rgb_points
    ADD [TF]
      [Frames]
  
Some commands:
  
  rosbag:
  1. rosbag record -a
  2. rosbag play "file_name".bag
  
  rosparam:
  1. rosparam list
  2. rosparam get /DOPE_object_names
