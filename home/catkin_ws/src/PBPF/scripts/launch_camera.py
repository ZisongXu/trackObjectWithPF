# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:18:07 2022
@author: 12106
"""
#test about changing the parameters of the target onject
import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
from numpy.linalg import inv
from pyquaternion import Quaternion
import math
import random
import copy
import os
import yaml
import matplotlib  
import matplotlib.pyplot as plt  
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tf
import tf.transformations as transformations
import rospy

class LaunchCamera():
    def __init__(self, width, height, fov_v):
        self.a = 0
        self.pixelWidth = width
        self.pixelHeight = height
        self.fov_v = fov_v

        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.OPTITRACK_FLAG = self.parameter_info['optitrack_flag']
        self.GAZEBO_FLAG = self.parameter_info['gazebo_flag']
        self.LOCATE_CAMERA_FLAG = self.parameter_info['locate_camera_flag']
        self.NEARVAL = self.parameter_info['nearVal']
        self.FARVAL = self.parameter_info['farVal']
        self.pw_T_camD_tf_4_4 = 0
        self.compute_cam_pose_flag = 0
        
    def setCameraPicAndGetPic(self, p_world=0, tf_listener=0, pw_T_rob_sim_4_4=0):
        
        # width = 1
        # height = 1
        # rgbImg = 1
        # depthImg = 1
        # segImg = 1

        pw_T_camD_tf_4_4 = self.getCameraInPybulletWorldPose44(tf_listener, pw_T_rob_sim_4_4)
        camera_eye_position = pw_T_camD_tf_4_4[:3, 3]
        camera_orientation = pw_T_camD_tf_4_4[:3, :3]
        camera_eye_position = [pw_T_camD_tf_4_4[0][3], pw_T_camD_tf_4_4[1][3], pw_T_camD_tf_4_4[2][3]] # pw_T_camD_tf_4_4[2][3]+0.05
        
        vector_x = camera_orientation @ np.array([1, 0, 0])
        vector_y = camera_orientation @ np.array([0, 1, 0])
        vector_z = camera_orientation @ np.array([0, 0, 1])
        # camera_eye_position = camera_eye_position + 0.022*vector_z
        
        # Calculate camera target position based on its orientation
        camera_target_position = camera_eye_position + vector_z # Assuming Z-axis is the forward direction

        # Calculate up vector for the camera (assuming Y-axis is down)
        camera_up_vector = camera_orientation @ np.array([0, -1, 0])
        
        viewMatrix = p_world.computeViewMatrix(
            cameraEyePosition = camera_eye_position,
            cameraTargetPosition = camera_target_position,
            cameraUpVector = camera_up_vector
        )

        projectionMatrix = p_world.computeProjectionMatrixFOV(
            fov = self.fov_v,               # field of view
            aspect = self.pixelWidth/self.pixelHeight,          # width / height
            nearVal = self.NEARVAL,            # len lower limit
            farVal = self.FARVAL                # len upper limit
        )
        width, height, rgbImg, depthImg, segImg = p_world.getCameraImage(
            width = self.pixelWidth, 
            height = self.pixelHeight,
            viewMatrix = viewMatrix,
            projectionMatrix = projectionMatrix
            # flags = p.ER_NO_SEGMENTATION_MASK
        )
        return width, height, rgbImg, depthImg, segImg, self.NEARVAL, self.FARVAL


    def setCameraPicAndGetPic2(self, p_world=0):
        
        # viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
        camTargetPos = [-0.3, -0, 0.0]
        upAxisIndex = 2
        camDistance = 1
        
        pixelWidth = self.pixelWidth
        pixelHeight = self.pixelHeight
        pixelWidth = 320
        pixelHeight = 220
        camTargetPos = [0, 0, 0]
        camDistance = 4
        yaw = 0.0
        pitch = -10
        roll = 0
        upAxisIndex = 2
        viewMatrix = p_world.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
                                                        upAxisIndex)
        projectionMatrix = [
            1.0825318098068237, 0.0, 0.0, 0.0,
            0.0, 1.732050895690918, 0.0, 0.0,
            0.0, 0.0, -1.0002000331878662, -1.0,
            0.0, 0.0, -0.020002000033855438, 0.0
        ]
        # projectionMatrix = [908.8558959960938, 0.0, 626.7174072265625, 0.0,
        #                     0.0, 906.6900634765625, 383.2095947265625, 0.0,
        #                     0.0, 0.0, 1.0, 0.0,
        #                     0.0, 0.0, -0.020002000033855438, 0.0]
        print(viewMatrix)
        print(projectionMatrix)
        width, height, rgbImg, depthImg, segImg = p_world.getCameraImage(pixelWidth,
                                                                         pixelWidth,
                                                                         viewMatrix=viewMatrix,
                                                                         projectionMatrix=projectionMatrix,
                                                                         shadow=1,
                                                                         lightDirection=[1, 1, 1]
                                                                         )
        return width, height, rgbImg, depthImg, segImg

    def getCameraInPybulletWorldPose44(self, tf_listener, pw_T_rob_sim_4_4):
        if self.compute_cam_pose_flag == 0:
            if self.OPTITRACK_FLAG == True and self.LOCATE_CAMERA_FLAG == "opti":
                realsense_tf = '/RealSense' # (use Optitrack)
            else:
                realsense_tf = '/ar_tracking_camera_frame' # (do not use Optitrack)
            if self.GAZEBO_FLAG == True:
                realsense_tf = '/realsense_camera'
            # mark
            while_time = 0
            while True:
                while_time = while_time + 1
                if while_time > 1000:
                    # print("In launch_camera.py: Can not find the pose of the camera!!!! You need to wait a while or try to debug")
                    a = 1
                try:
                    # (trans_camera, rot_camera) = tf_listener.lookupTransform('/zisong_robot', realsense_tf, rospy.Time(0))
                    (trans_camera, rot_camera) = tf_listener.lookupTransform('/panda_link0', realsense_tf, rospy.Time(0))
                    # (trans_camera_link0, rot_camera_link0) = tf_listener.lookupTransform('/panda_link0', realsense_tf, rospy.Time(0))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            camRGB_T_camD_tf_pos = [0.015, 0.0, 0.0]
            camRGB_T_camD_tf_ori = [0.0, 0.0, -0.008, 1] # x, y, z, w
            # camRGB_T_camD_tf_ori = [0.001, 0.001, -0.009, 1] # x, y, z, w


            # camRGB_T_camD_tf_pos = [0.0, 0.0, 0.0]
            # camRGB_T_camD_tf_ori = [0.0, 0.0, -0.00, 1] # x, y, z, w

            camRGB_T_camD_tf_3_3 = np.array(p.getMatrixFromQuaternion(camRGB_T_camD_tf_ori)).reshape(3, 3)
            camRGB_T_camD_tf_3_4 = np.c_[camRGB_T_camD_tf_3_3, camRGB_T_camD_tf_pos]  # Add position to create 3x4 matrix
            camRGB_T_camD_tf_4_4 = np.r_[camRGB_T_camD_tf_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix

            rob_T_camRGB_tf_pos = list(trans_camera)
            rob_T_camRGB_tf_ori = list(rot_camera)
            rob_T_camRGB_tf_3_3 = np.array(p.getMatrixFromQuaternion(rob_T_camRGB_tf_ori)).reshape(3, 3)
            rob_T_camRGB_tf_3_4 = np.c_[rob_T_camRGB_tf_3_3, rob_T_camRGB_tf_pos]  # Add position to create 3x4 matrix
            rob_T_camRGB_tf_4_4 = np.r_[rob_T_camRGB_tf_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix

            rob_T_camD_tf_4_4 = np.dot(rob_T_camRGB_tf_4_4, camRGB_T_camD_tf_4_4)
            self.pw_T_camD_tf_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_camD_tf_4_4)
            # self.pw_T_camD_tf_4_4[0][3] = self.pw_T_camD_tf_4_4[0][3] - 0.02
            # self.pw_T_camD_tf_4_4[1][3] = self.pw_T_camD_tf_4_4[1][3] - 0.02
            self.compute_cam_pose_flag = 1
        else:
            return self.pw_T_camD_tf_4_4

        return self.pw_T_camD_tf_4_4 # pw_T_camD_tf_4_4

    def getFocalLength(self):
        F_x = 651.248474121094
        F_y = 651.248474121094
        PPX = 636.278076171875
        PPY = 354.887420654297
        if self.pixelWidth == 1280 and self.pixelHeight == 720:
            F_x = 651.248474121094
            F_y = 651.248474121094
            PPX = 636.278076171875
            PPY = 354.887420654297
        focal_length = F_x * PPX / self.pixelWidth
        return focal_length






def quaternion_correction(quaternion): # x,y,z,w
    new_quat = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]) # w,x,y,z
    cos_theta_over_2 = new_quat.w
    sin_theta_over_2 = math.sqrt(new_quat.x ** 2 + new_quat.y ** 2 + new_quat.z ** 2)
    theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
    theta = theta_over_2 * 2.0
    while theta >= math.pi:
        theta = theta - 2.0*math.pi
    while theta <= -math.pi:
        theta = theta + 2.0*math.pi
    new_quaternion = [math.sin(theta/2.0)*(new_quat.x/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.y/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.z/sin_theta_over_2), math.cos(theta/2.0)]
    #if theta >= math.pi or theta <= -math.pi:
    #    new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
    #    return new_quaternion
    #return quaternion # x,y,z,w
    return new_quaternion



              