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
import matplotlib  
import matplotlib.pyplot as plt  
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tf
import tf.transformations as transformations

class LaunchCamera():
    def __init__(self, width, height):
        self.a = 0
        self.pixelWidth = width
        self.pixelHeight = height
    def setCameraPicAndGetPic(self, p_world=0):
        # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        
        pw_T_cam_tf = [[-0.01873203,  0.57812041, -0.81573642,  1.14657295],
                       [ 0.99966816, -0.00360018, -0.0255072 ,  0.13681107],
                       [-0.01768303, -0.81594352, -0.57786113,  0.41750256],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]
        pw_T_cam_tf_pos = [pw_T_cam_tf[0][3], pw_T_cam_tf[1][3], pw_T_cam_tf[2][3]]
        pw_T_cam_tf_ori = transformations.quaternion_from_matrix(pw_T_cam_tf)
        
        pw_T_cam_tf_ori = quaternion_correction(pw_T_cam_tf_ori) # x, y, z, w
        # getEulerFromQuaternion returns a list of 3 floating point values, a vec3. 
        # The rotation order is first roll around X, then pitch around Y and finally yaw around Z, as in the ROS URDF rpy convention.

        pw_T_cam_tf_radian = p_world.getEulerFromQuaternion(pw_T_cam_tf_ori)
        x_rad = pw_T_cam_tf_radian[0]
        y_rad = pw_T_cam_tf_radian[1]
        z_rad = pw_T_cam_tf_radian[2]
        RADIAN_TO_ANGLE_VALUE = 180.0 / math.pi
        x_ang = RADIAN_TO_ANGLE_VALUE * x_rad
        y_ang = RADIAN_TO_ANGLE_VALUE * y_rad
        z_ang = RADIAN_TO_ANGLE_VALUE * z_rad
        tz_vec = [0.0, 0.0, 1]
        cameraPos = [-0.4, -0.4, 0.4]
        cameraPos = pw_T_cam_tf_pos
        targetPos = [0.0, 0.0, 0.0]
        viewMatrix = p_world.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tz_vec
        )
        
        # Facing y
        camTargetPos = [0., 0., 0.3]
        camDistance = 1
        yaw = 45 # z
        pitch = -90 # x
        roll = 45 # y
        yaw = z_ang # z  z: 91.07349685292553
        pitch = y_ang # x  y: 1.0132157040776695
        roll = -x_ang # y  x: -125.3065901872438
        upAxisIndex = 1
        viewMatrix = p_world.computeViewMatrixFromYawPitchRoll(camTargetPos, 
                                                               camDistance, 
                                                               yaw, pitch, roll,
                                                               upAxisIndex)

        width = self.pixelWidth
        height = self.pixelHeight
        projectionMatrix = p_world.computeProjectionMatrixFOV(
            # fov=86.0,               # 摄像头的视线夹角
            # aspect=16.0/9,          # width / height
            # nearVal=0.3,            # 摄像头焦距下限
            # farVal=3                # 摄像头能看上限
            fov = 69.40,               # 摄像头的视线夹角
            aspect = width/height,          # width / height
            nearVal = 0.01,            # 摄像头焦距下限
            farVal = 10.0                # 摄像头能看上限
        )
        width, height, rgbImg, depthImg, segImg = p_world.getCameraImage(
            width=width, height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix
            # flags=p.ER_NO_SEGMENTATION_MASK
        )
        return width, height, rgbImg, depthImg, segImg
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



              