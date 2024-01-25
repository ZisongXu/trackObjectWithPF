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
 
 
import math
import random
import copy
 
import matplotlib  
import matplotlib.pyplot as plt  
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
 
class EXP_world():
    def __init__(self):
        self.pos_x_sigma = 0.05
        self.pos_y_sigma = 0.05
        self.pos_z_sigma = 0
        self.ang_x_sigma = 0
        self.ang_y_sigma = 0
        self.ang_z_sigma = 0

 
    def real_world_scene(self):
        p_real_world = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT, GUI_SERVER
        # physicsClient = p.connect(p.direct)
        p_real_world.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_real_world.setGravity(0,0,-9.81) 
        p_real_world.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])#转变视角
        plane_id = p_real_world.loadURDF("plane.urdf")
        #load and set real robot
        real_world_cracker_pos = [0, 0, 0.081]
        real_world_cracker_ori = p_real_world.getQuaternionFromEuler([0, math.pi/2.0, 0])
        real_world_cracker_id = p_real_world.loadURDF("D:/Study/University_of_Leeds/PhD/PhD_project/code/cracker/cracker_par_no_visual_hor.urdf",
                                                    real_world_cracker_pos,
                                                    real_world_cracker_ori)
        return p_real_world, real_world_cracker_pos, real_world_cracker_ori
    def particle_world_scene(self):
        p_particle_world = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT, GUI_SERVER
        # physicsClient = p.connect(p.direct)
        p_particle_world.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_particle_world.setGravity(0,0,-9.81) 
        p_particle_world.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])#转变视角
        plane_id = p_particle_world.loadURDF("plane.urdf")
        # load and set real robot
        real_world_cracker_pos = [0, 0, 0.081]
        real_world_cracker_ang = [0, math.pi/2.0, 0]
        real_world_cracker_ori = p_particle_world.getQuaternionFromEuler([0, math.pi/2.0, 0])
        # add noise to pos
        pos_x = add_noise_2_pos(real_world_cracker_pos[0], self.pos_x_sigma)
        pos_y = add_noise_2_pos(real_world_cracker_pos[1], self.pos_y_sigma)
        pos_z = add_noise_2_pos(real_world_cracker_pos[2], self.pos_z_sigma)
        ang_x = add_noise_2_ang(real_world_cracker_ang[0], self.ang_x_sigma)
        ang_y = add_noise_2_ang(real_world_cracker_ang[1], self.ang_y_sigma)
        ang_z = add_noise_2_ang(real_world_cracker_ang[2], self.ang_z_sigma)
        particle_world_cracker_pos = [pos_x, pos_y, pos_z]
        particle_world_cracker_ang = [ang_x, ang_y, ang_z]
        particle_world_cracker_ori = p_particle_world.getQuaternionFromEuler(particle_world_cracker_ang)
        particle_world_cracker_id = p_particle_world.loadURDF("D:/Study/University_of_Leeds/PhD/PhD_project/code/cracker/cracker_par_no_visual_hor.urdf",
                                                    particle_world_cracker_pos,
                                                    particle_world_cracker_ori)
        return p_particle_world, particle_world_cracker_pos, particle_world_cracker_ori
 
 
def take_easy_gaussian_value(mean,sigma):
    normal = random.normalvariate(mean, sigma)
    return normal
 
def add_noise_2_pos(current_pos=0, sigma=0):
    mean = current_pos
    sigma = sigma
    new_pos_is_added_noise = take_easy_gaussian_value(mean, sigma)
    return new_pos_is_added_noise
 
def add_noise_2_ang(current_angle=0, sigma=0):
    mean = current_angle
    sigma = sigma
    new_angle_is_added_noise = take_easy_gaussian_value(mean, sigma)
    return new_angle_is_added_noise
# compute posibility
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)
 
def compute_distance_bt_2_points(pos1, pos2):
    x_d = pos1[0]-pos2[0]
    y_d = pos1[1]-pos2[1]
    z_d = pos1[2]-pos2[2]
    distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
    return distance
 
def compute_difference_bt_2_depthImg(depthImg1, depthImg2):
    depth_value_diff_container = depthImg1-depthImg2
    return depth_value_diff_container
def depth_value_difference_abs_sum(depth_value_diff_container):
    dim_number = depth_value_diff_container.ndim
    depth_value_diff_container_abs = np.abs(depth_value_diff_container)
    interim_container = copy.deepcopy(depth_value_diff_container_abs)
    for dim_n in range(dim_number):
        interim_container = sum(interim_container)
    return interim_container
 
def depth_value_difference_square_root(depth_value_diff_container):
    dim_number = depth_value_diff_container.ndim
    depth_value_diff_container_square = depth_value_diff_container ** 2
    interim_container = copy.deepcopy(depth_value_diff_container_square)
    for dim_n in range(dim_number):
        interim_container = sum(interim_container)
    test_list_array_square_sum_root = math.sqrt(interim_container)
    return test_list_array_square_sum_root
def normalization_method(given_list):
    given_list_normalize = []
    given_list_sum = sum(given_list)
    for index, value in enumerate(given_list):
        value = value / given_list_sum
        given_list_normalize.append(value)
    return given_list_normalize
 
def setCameraPicAndGetPic(p=0, width=640, height=480):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    tz_vec = [0.0, 0.0, 1.0]
    # print("baseOrientation:", baseOrientation)
    # print("matrix:", matrix)
    # print("tx_vec:", tx_vec)
    # print("tz_vec:", tz_vec)
    # 摄像头的位置
    # BASE_RADIUS 为 0.5，是机器人底盘的半径。BASE_THICKNESS 为 0.2 是机器人底盘的厚度。
    # 别问我为啥不写成全局参数，因为我忘了我当时为什么这么写的。
    cameraPos = [-0.4, -0.4, 0.4]
    targetPos = [0.0, 0.0, 0.0]
 
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec
        # physicsClientId=physicsClientId
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=86.0,               # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.3,            # 摄像头焦距下限
        farVal=3                # 摄像头能看上限
        # physicsClientId=physicsClientId
    )
 
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix
        # physicsClientId=physicsClientId,
        # flags=p.ER_NO_SEGMENTATION_MASK 
    )
 
    return width, height, rgbImg, depthImg, segImg
 
 
if __name__ == '__main__':
    run_camera_flag = True
    loop_flag = 0
    exp_world = EXP_world()

    while True:
        loop_flag = loop_flag + 1
        if loop_flag == 11:
            break;
        p_real_world, real_world_cracker_pos, real_world_cracker_ori = exp_world.real_world_scene()
        p_world_list = []
        world_cracker_pos_list = []
        world_cracker_ori_lsit = []
        p_world_list.append(p_real_world)
        world_cracker_pos_list.append(real_world_cracker_pos)
        world_cracker_ori_lsit.append(real_world_cracker_ori)
        random_world_num = 100
        for i in range(random_world_num):
            p_particle_world, particle_world_cracker_pos, particle_world_cracker_ori = exp_world.particle_world_scene()
            p_world_list.append(p_particle_world)
            world_cracker_pos_list.append(particle_world_cracker_pos)
            world_cracker_ori_lsit.append(particle_world_cracker_ori)

        t1 = time.time()
        if run_camera_flag == True:
            depthImg_list = []
            segImg_list = []
            distance_list = []
            depth_value_difference_sum_list = []
            distance_test_list = []
            distance_test_list.append(0)
            for index, value in enumerate(p_world_list):
                p_world_id = value
                width, height, rgbImg, depthImg, segImg = setCameraPicAndGetPic(p=p_world_id)
                depthImg_list.append(depthImg)
                segImg_list.append(segImg)
                # fig, axs = plt.subplots(1,3)
                # axs[0].imshow(depthImg, cmap="gray")
                # axs[0].set_title('Depth image')
                # axs[1].imshow(rgbImg, cmap="gray")
                # axs[1].set_title('Rgb image')
                # axs[2].imshow(segImg, cmap="gray")
                # axs[2].set_title('Segmentation image')
                # # plt.imshow(depthImg, cmap="gray")
                # plt.plot(label='ax2')
                # # plt.legend()
                # plt.show()
                np.set_printoptions(threshold=np.inf)
                # print(segImg)
            # compute distance
            for index, value in enumerate(world_cracker_pos_list):
                if index+1 < len(p_world_list):
                    pos1 = copy.deepcopy(world_cracker_pos_list[0])
                    pos2 = copy.deepcopy(world_cracker_pos_list[index+1])
                    distance = compute_distance_bt_2_points(pos1, pos2)
                    distance_list.append(distance)
                    depthImg1 = copy.deepcopy(depthImg_list[0])
                    depthImg2 = copy.deepcopy(depthImg_list[index+1])
                    depth_value_difference = compute_difference_bt_2_depthImg(depthImg1, depthImg2)
                    depth_value_difference_sum = depth_value_difference_abs_sum(depth_value_difference)
                    depth_value_difference_sum = depth_value_difference_square_root(depth_value_difference)
                    depth_value_difference_sum_list.append(depth_value_difference_sum)
            # normalization
            depth_value_difference_sum_list_normalize = normalization_method(depth_value_difference_sum_list)        
                # depth_value_difference_sum_list_normalize_single = []
                # depth_value_difference_sum_list_normalize_single.append(value)
                # distance_list_single = []
                # distance_list_single.append(distance_list[index])
                # p1 = distance_list_single  # 数据点
                # p2 = depth_value_difference_sum_list_normalize_single
                # plt.figure('Draw')
                # plt.scatter(p1, p2, s=5)  # scatter绘制散点图
                # plt.xlabel("Distance Error")
                # plt.ylabel("Depth Img Error")
                # plt.draw()
                # print("here")
        # print(depth_value_difference_sum_list_normalize)     
            # print(distance_list)
            # print(world_cracker_pos_list[1:len(world_cracker_pos_list)])
            # print("distance_list:", len(distance_list))    
            # print(len(depthImg_list))
        t2 = time.time()
        p1 = distance_list  # 数据点
        p2 = depth_value_difference_sum_list_normalize
        #创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
        plt.figure('Draw')
        plt.scatter(p1, p2, s=5)  # scatter绘制散点图
        plt.xlabel("Distance Error")
        plt.ylabel("Depth Img Error")
        plt.draw()
        print("time_consuming:", t2 - t1)
        # print(depthImg)
        # # print(depthImg[224][112])
        # print(len(depthImg))
        # print(len(depthImg[0]))

        p_real_world.stepSimulation()
        time.sleep(1/240)
        for index, value in enumerate(p_world_list):
            value.disconnect()