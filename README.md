# Physics_Based_Particle_Filtering 

This is the official implementation of our paper "Real-Time Physics-Based Object Pose Tracking during Non-Prehensile Manipulation" submitted to IEEE International Conference on Robotics and Automation (ICRA) 2023.

**Abstract:** We propose a method to track the 6D pose of an object over time, while the object is under non-prehensile manipulation by a robot. At any given time during the manipulation of the object, we assume access to the robot joint controls and an image from a camera looking at the scene. We use the robot joint controls to perform a physics-based prediction of how the object might be moving. We then combine this prediction with the observation coming from the camera, to estimate the object pose as accurately as possible. We use a particle filtering approach to combine the control information with the visual information. We compare the proposed method with two baselines: (i) using only an image-based pose estimation system at each time-step, and (ii) a particle filter which does not perform the computationally expensive physics predictions, but assumes the object moves with constant velocity. Our results show that making physics-based predictions is worth the computational cost, resulting in more accurate tracking, and estimating object pose even when the object is not clearly visible to the camera.


# Supplementary Video:

Click to watch

[![Watch the video](https://img.youtube.com/vi/srZZM_CKum4/hqdefault.jpg)](https://youtu.be/srZZM_CKum4)


# Brief Description:

We propose a method to track the pose of an object over time, by using the image from the camera, and the particles in the physical engine. Although sometimes the camera cannot see the object clearly, our method can still track the pose of the object.


# Quick Setup:
This project uses singularity container to support all the code:

```./build.sh``` 

```./run.sh```

