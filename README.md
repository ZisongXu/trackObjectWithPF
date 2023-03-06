# Physics_Based_Particle_Filtering 

This is the official implementation of our paper "Real-Time Physics-Based Object Pose Tracking during Non-Prehensile Manipulation" submitted to IEEE/RSJ International Conference on Intelligent Robots and System (IROS) 2023.

**Abstract:** We propose a method to track the 6D pose of an object over time, while the object is under non-prehensile manipulation by a robot. At any given time during the manipulation of the object, we assume access to the robot joint controls and an image from a camera. We use the robot joint controls to perform a physics-based prediction of how the object might be moving. We then combine this prediction with the observation coming from the camera, to estimate the object pose as accurately as possible. We use a particle filtering approach to combine the control information with the visual information. We compare the proposed method with two baselines: (i) using only an image-based pose estimation system at each time-step, and (ii) a particle filter which does not perform the computationally expensive physics predictions, but assumes the object moves with constant velocity. Our results show that making physics-based predictions is worth the computational cost, resulting in more accurate tracking, and estimating object pose even when the object is not clearly visible to the camera.


# Supplementary Video:

Click to watch

[![Watch the video](https://i.ytimg.com/vi/EMBFYzkno64/maxresdefault.jpg)](https://www.youtube.com/watch?v=EMBFYzkno64)


# Brief Description:

We propose a method to track the pose of an object over time, by using the image from the camera, and the particles in the physical engine. Although sometimes the camera cannot see the object clearly, our method can still track the pose of the object.


# Quick Setup:
1. **Build Container** (This project uses singularity container to support all the code)

	Please enter into the main folder and run ```./build.sh``` in Ubuntu20 terminal to build the container.

2. **Download Rosbags** (For running demos only)
	
	Download [the rosbags](https://drive.google.com/drive/folders/13EbCuu231izDbmrcIeyjeQlJSPJL1qWW?usp=sharing) and save them to the ```rosbag``` folder, i.e., ```~/rosbag/```.


# Running Code
1. **Start Container**

	In the terminal, enter into the main file and run ```./run.sh```, and then you can see ```[TrackObjectWithPF] Singularity> ~ $```

2. **Start ROS Master**
	
	```$ roscore```
	
3. **Using Simulation Time** (For running demos only)

	```$ rosparam set use_sim_time true```
	
4. **Edit Config Information** (if desired) in ```~/catkin_ws/src/PBPF/config/parameter_info.yaml```

	- ```err_file```: 
	- ```gazebo_flag```: 
	- ```object_name_list```:
	- ```object_num```: 
	- ```other_obj_num```: 
	- ```oto_name_list```:
	- ```otob_name_list```:
	- ```particle_num```: 
	- ```pick_particle_rate```: 
	- ```ray_point_num```: 
	- ```robot_num```: 
	- ```run_alg_flag```: 
	- ```task_flag```:
	- ```update_style_flag```: 
	- ```version```: 
	
In the container:

```[TrackObjectWithPF] Singularity> ~/project/code $ python3 Physics_Based_Particle_Filtering.py```

# Rosbag Link
[Rosbags for each scene of different objects](https://drive.google.com/drive/folders/13EbCuu231izDbmrcIeyjeQlJSPJL1qWW?usp=sharing)


