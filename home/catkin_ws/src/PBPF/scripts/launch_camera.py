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
class ExpWorld():
    def __init__(self):
        self.pos_x_sigma = 0.05
        self.pos_y_sigma = 0.05
        self.pos_z_sigma = 0
        self.ang_x_sigma = 0
        self.ang_y_sigma = 0
        self.ang_z_sigma = 0
    def real_world_scene(self):
        p_real_world = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER
        # physicsClient = p.connect(p.direct)
        p_real_world.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_real_world.setGravity(0,0,-9.81)
        p_real_world.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])#转变视角
        plane_id = p_real_world.loadURDF("plane.urdf")
        #load and set real robot
        real_world_cracker_pos = [0, 0, 0.081]
        real_world_cracker_ori = p_real_world.getQuaternionFromEuler([0, math.pi/2.0, 0])
        real_world_cracker_id = p_real_world.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_par_no_visual_hor.urdf"),
                                                    real_world_cracker_pos,
                                                    real_world_cracker_ori)
        return p_real_world, real_world_cracker_pos, real_world_cracker_ori

class LaunchCamera():
    def __init__(self, width, height):
        self.a = 0
        self.pixelWidth = width
        self.pixelHeight = height
    def setCameraPicAndGetPic(self, p_world=0):
        # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        tz_vec = [0.0, 0.0, 1.0]
        cameraPos = [-0.4, -0.4, 0.4]
        targetPos = [0.0, 0.0, 0.0]
        viewMatrix = p_world.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tz_vec
        )

        camTargetPos = [0., 0., 0.]
        camDistance = 1
        yaw = 0.0
        pitch = -90
        roll = 0
        upAxisIndex = 2
        viewMatrix = p_world.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
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
        pw_T_cam_tf = [[-0.01873203,  0.57812041, -0.81573642,  1.14657295],
                       [ 0.99966816, -0.00360018, -0.0255072 ,  0.13681107],
                       [-0.01768303, -0.81594352, -0.57786113,  0.41750256],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]

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
if __name__ == '__main__':
    run_camera_flag = True
    loop_flag = 0
    exp_world = ExpWorld()
    p_real_world, real_world_cracker_pos, real_world_cracker_ori = exp_world.real_world_scene()
    width = 1280
    height = 720
    launch_camera = LaunchCamera(width, height)
    width, height, rgbImg, depthImg, segImg = launch_camera.setCameraPicAndGetPic(p_real_world)
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(depthImg, cmap="gray")
    axs[0].set_title('Depth image')
    axs[1].imshow(rgbImg, cmap="gray")
    axs[1].set_title('Rgb image')
    axs[2].imshow(segImg, cmap="gray")
    axs[2].set_title('Segmentation image')
    # plt.imshow(depthImg, cmap="gray")
    plt.plot(label='ax2')
    # plt.legend()
    plt.show()
    while True:
        width, height, rgbImg, depthImg, segImg = launch_camera.setCameraPicAndGetPic(p_real_world)
        # width, height, rgbImg, depthImg, segImg = launch_camera.setCameraPicAndGetPic2(p_real_world)
        p_real_world.stepSimulation()
        # print(depthImg)
        time.sleep(1/240)        
    # while True:
    #     loop_flag = loop_flag + 1
    #     if loop_flag == 11:
    #         break;
    #     p_real_world, real_world_cracker_pos, real_world_cracker_ori = exp_world.real_world_scene()
    #     p_world_list = []
    #     world_cracker_pos_list = []
    #     world_cracker_ori_lsit = []
    #     p_world_list.append(p_real_world)
    #     world_cracker_pos_list.append(real_world_cracker_pos)
    #     world_cracker_ori_lsit.append(real_world_cracker_ori)
    #     random_world_num = 100
    #     for i in range(random_world_num):
    #         p_particle_world, particle_world_cracker_pos, particle_world_cracker_ori = exp_world.particle_world_scene()
    #         p_world_list.append(p_particle_world)
    #         world_cracker_pos_list.append(particle_world_cracker_pos)
    #         world_cracker_ori_lsit.append(particle_world_cracker_ori)
    #     t1 = time.time()
    #     t2 = time.time()
    #     p1 = distance_list  # 数据点
    #     p2 = depth_value_difference_sum_list_normalize
    #     #创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
    #     plt.figure('Draw')
    #     plt.scatter(p1, p2, s=5)  # scatter绘制散点图
    #     plt.xlabel("Distance Error")
    #     plt.ylabel("Depth Img Error")
    #     plt.draw()
    #     print("time_consuming:", t2 - t1)
    #     # print(depthImg)
    #     # # print(depthImg[224][112])
    #     # print(len(depthImg))
    #     # print(len(depthImg[0]))
    #     p_real_world.stepSimulation()
    #     time.sleep(1/240)