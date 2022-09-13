# trackObjectWithPF
Ubuntu20; ROS:noetic; Pybullet
  1. Go to this file
  2. Run ./build.sh in the terminal
  3. Run ./run.sh in the terminal


The whole process:
  Control real robot
  1. (server) Desk 
  2. (server) ssh to WORKSTATION: roscore 
  3. (server) ssh to WORKSTATION: roslaunch leeds_panda_launchers position_joint_trajectory_controller.launch 
  3. (server) [TrackObjectWithPF]: roslaunch panda_moveit moveit.launch 
  4. (server) [TrackObjectWithPF]: rosrun panda_moveit demo_talker_real_robot.py 
  DOPE
  5. (server) [realsense]: roslaunch panda_camera_launchers realsense.launch 
  6. (server) [DOPE]: roslaunch dope dope.launch 
  7. (server) [DOPE]: rosrun dope_utilities read_dope_objects_to_param_server.py
  8. (Ubuntu16VM) rosrun natnet_ros client _server:=192.168.131.103
  9. (Ubuntu16VM) roslaunch dope_utilities dope_convertion.launch
  (maybe not)
  10. (server) [DOPE]: rosrun rviz rviz 
  RViz:
    [Global Options]: pandaRobot
    [Image]: /dope/rgb_points
    [TF]: cracker/cheezit
  TrackObjectWithPF Code
  11. (server) [TrackObjectWithPF]: cd /home/phd_code/python3 code.py



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
  2. rosbag record /tf /mocap/rigid_bodies/RealSense/pose /mocap/rigid_bodies/baseofcheezit/pose /mocap/rigid_bodies/cheezit/pose /mocap/rigid_bodies/pandaRobot/pose /joint_states -o test.bag
  3. rosbag record /tf /mocap/rigid_bodies/RealSense/pose /mocap/rigid_bodies/baseofcheezit/pose /mocap/rigid_bodies/cheezit/pose /mocap/rigid_bodies/pandaRobot/pose /joint_states /camera/color/image_raw /DOPE_pose /Opti_pose /PFPE_pose -o test.bag
  4. rosbag record /tf /joint_states /camera/color/image_raw -o test.bag
  5. rosbag record /tf /joint_states /camera/color/image_raw /DOPE_pose /PFPE_pose -o test.bag
  6. rosbag play "file_name".bag
  
  rosparam:
  1. rosparam list
  2. rosparam get /DOPE_object_names
  
  kill:
  1. kill -9 %1
