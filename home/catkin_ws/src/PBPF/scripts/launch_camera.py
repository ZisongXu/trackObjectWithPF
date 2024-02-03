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
    def __init__(self, width, height):
        self.a = 0
        self.pixelWidth = width
        self.pixelHeight = height

        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.optitrack_flag = self.parameter_info['optitrack_flag']
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        self.pw_T_cam_tf_4_4 = 0
        self.compute_cam_pose_flag = 0
        self.nearVal = 0.01
        self.farVal = 10.0
        
    def setCameraPicAndGetPic(self, p_world=0, tf_listener=0, pw_T_rob_sim_4_4=0):
        # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        # width = 1
        # height = 1
        # rgbImg = 1
        # depthImg = 1
        # segImg = 1

        pw_T_cam_tf_4_4 = self.getCameraInPybulletWorldPose44(tf_listener, pw_T_rob_sim_4_4)
        
        camera_eye_position = [pw_T_cam_tf_4_4[0][3], pw_T_cam_tf_4_4[1][3], pw_T_cam_tf_4_4[2][3]]
        tz_vec_x = [pw_T_cam_tf_4_4[0][0], pw_T_cam_tf_4_4[1][0], pw_T_cam_tf_4_4[2][0]]
        tz_vec_y = [pw_T_cam_tf_4_4[0][1], pw_T_cam_tf_4_4[1][1], pw_T_cam_tf_4_4[2][1]]
        tz_vec_z = [pw_T_cam_tf_4_4[0][2], pw_T_cam_tf_4_4[1][2], pw_T_cam_tf_4_4[2][2]]
        tz_vec_y_array = np.array(tz_vec_y)
        tz_vec_y_array_new = tz_vec_y_array * (-1)
        tz_vec_y_new = list(tz_vec_y_array_new)
        
        focal_length = self.getFocalLength()
        camera_target_position = self.getCameraTargetPosition(camera_eye_position, focal_length, tz_vec_z)
        
        viewMatrix = p_world.computeViewMatrix(
            cameraEyePosition = camera_eye_position,
            cameraTargetPosition = camera_target_position,
            cameraUpVector = tz_vec_y_new
        )

        projectionMatrix = p_world.computeProjectionMatrixFOV(
            # fov=86.0,               # 摄像头的视线夹角
            # aspect=16.0/9,          # width / height
            # nearVal=0.3,            # 摄像头焦距下限
            # farVal=3                # 摄像头能看上限
            fov = 57.86,               # 摄像头的视线夹角
            aspect = self.pixelWidth/self.pixelHeight,          # width / height
            nearVal = self.nearVal,            # 摄像头焦距下限
            farVal = self.farVal                # 摄像头能看上限
        )
        width, height, rgbImg, depthImg, segImg = p_world.getCameraImage(
            width=self.pixelWidth, height=self.pixelHeight,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        return width, height, rgbImg, depthImg, segImg, self.nearVal, self.farVal


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
            if self.optitrack_flag == True:
                realsense_tf = '/RealSense' # (use Optitrack)
            else:
                realsense_tf = '/ar_tracking_camera_frame' # (do not use Optitrack)
            if self.gazebo_flag == True:
                realsense_tf = '/realsense_camera'
            
            try:
                (trans_camera, rot_camera) = tf_listener.lookupTransform('/panda_link0', realsense_tf, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("Can not find the pose of the camera!!!!")

            rob_T_cam_tf_pos = list(trans_camera)
            rob_T_cam_tf_ori = list(rot_camera)
            rob_T_cam_tf_3_3 = transformations.quaternion_matrix(rob_T_cam_tf_ori)
            rob_T_cam_tf_4_4 = rotation_4_4_to_transformation_4_4(rob_T_cam_tf_3_3, rob_T_cam_tf_pos)
            self.pw_T_cam_tf_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_cam_tf_4_4)
            self.compute_cam_pose_flag = 1
        else:
            return self.pw_T_cam_tf_4_4
        return self.pw_T_cam_tf_4_4

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

    def getCameraTargetPosition(self, camera_eye_position, focal_length, tz_vec_z):
        camera_target_position_x =  camera_eye_position[0] + focal_length * tz_vec_z[0]
        camera_target_position_y =  camera_eye_position[1] + focal_length * tz_vec_z[1]
        camera_target_position_z =  camera_eye_position[2] + focal_length * tz_vec_z[2]
        camera_target_position = [camera_target_position_x, camera_target_position_y, camera_target_position_z]
        return camera_target_position


def rotation_4_4_to_transformation_4_4(rotation_4_4, pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4


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



              