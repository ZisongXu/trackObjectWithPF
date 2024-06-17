import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
import os
import matplotlib  
from matplotlib import pyplot as plt
import threading
import multiprocessing
from multiprocessing import Process
import math
import jax
import random

def set_real_robot_JointPosition(pybullet_env, robot):
    a1 = random.uniform(-1, 1)
    a2 = random.uniform(-1, 1)
    a3 = random.uniform(-1, 1)
    a4 = -2.1335976389165348
    a5 = random.uniform(-1, 1)
    a6 = random.uniform(-1, 1)
    
    a7 = random.uniform(-1, 1)
    a8 = 0.00026791999698616564
    a9 = 0.00026791999698616564
    joint_states = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
    # print("======================")
    # print(a1,a2,a3,a4,a5,a6,a7,a8,a9)
    # print("======================")
    position = joint_states
    num_joints = 9
    for joint_index in range(num_joints):
        if joint_index == 7 or joint_index == 8:
            pybullet_env.setJointMotorControl2(robot,
                                               joint_index+2,
                                               pybullet_env.POSITION_CONTROL,
                                               targetPosition=position[joint_index])
        else:
            pybullet_env.setJointMotorControl2(robot,
                                               joint_index,
                                               pybullet_env.POSITION_CONTROL,
                                               targetPosition=position[joint_index])
    for i in range(240):
        pybullet_env.stepSimulation()                            
  
def create_pybullet_world():
    # pybullet.connect(pybullet.GUI) # GUI/DIRECT
    p_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
    p_env.setAdditionalSearchPath(pybullet_data.getDataPath())
    p_env.setGravity(0,0,-9.81) 
    p_env.resetDebugVisualizerCamera(cameraDistance=1., cameraYaw=90, cameraPitch=-50, cameraTargetPosition=[0.1,0.15,0.35])    
    # p_env.setTimeStep(1.0/100)

    plane_id = p_env.loadURDF("plane.urdf")
    pos, ori = p_env.getBasePositionAndOrientation(plane_id)
    print(pos)
    table_pos_1 = [0.46, -0.01, 0.710]
    table_ori_1 = p_env.getQuaternionFromEuler([0,0,0])
    table_id_1 = p_env.loadURDF("table.urdf", table_pos_1, table_ori_1, useFixedBase = 1)

    barry_pos_1 = [-0.694, 0.443, 0.895]
    barry_ori_1 = p_env.getQuaternionFromEuler([0,math.pi/2,0])
    barry_id_1 = p_env.loadURDF("barrier.urdf", barry_pos_1, barry_ori_1, useFixedBase = 1)
    
    barry_pos_2 = [-0.694, -0.607, 0.895]
    barry_ori_2 = p_env.getQuaternionFromEuler([0,math.pi/2,0])
    barry_id_2 = p_env.loadURDF("barrier.urdf", barry_pos_2, barry_ori_2, useFixedBase = 1)
    
    barry_pos_3 = [0.459, -0.972, 0.895]
    barry_ori_3 = p_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
    barry_id_3 = p_env.loadURDF("barrier.urdf", barry_pos_3, barry_ori_3, useFixedBase = 1)

    barry_pos_4 = [-0.549, 0.61, 0.895]
    barry_ori_4 = p_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
    # barry_id_4 = p_env.loadURDF("barrier.urdf", barry_pos_4, barry_ori_4, useFixedBase = 1)
    
    barry_pos_5 = [0.499, 0.61, 0.895]
    barry_ori_5 = p_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
    # barry_id_5 = p_env.loadURDF("barrier.urdf", barry_pos_5, barry_ori_5, useFixedBase = 1)
    
    board_pos_1 = [0.274, 0.581, 0.87575]
    board_ori_1 = p_env.getQuaternionFromEuler([math.pi/2,math.pi/2,0])
    board_id_1 = p_env.loadURDF("board.urdf", board_pos_1, board_ori_1, useFixedBase = 1)
    
    ketchup_pos_1 = [0.274, 0.381, 0.085+table_pos_1[2]]
    ketchup_ori_1 = p_env.getQuaternionFromEuler([math.pi/2,0,0])
    ketchup_id_1 = p_env.loadURDF(os.path.expanduser("YcbTomatoSoupCan/model.urdf"), ketchup_pos_1, ketchup_ori_1)
    
    # ketchup_pos_1 = [0.274, 0.381, 0.0743+table_pos_1[2]]
    # ketchup_ori_1 = p_env.getQuaternionFromEuler([math.pi/2,0,0])
    # ketchup_id_1 = p_env.loadURDF(os.path.expanduser("tomato_ketchup/tomato_ketchup.urdf"), ketchup_pos_1, ketchup_ori_1)
    
    # ketchup_pos_1 = [0.274, 0.381, 0.11+table_pos_1[2]]
    # ketchup_ori_1 = p_env.getQuaternionFromEuler([math.pi/2,0,0])
    # ketchup_id_1 = p_env.loadURDF(os.path.expanduser("orange_juice/orange_juice.urdf"), ketchup_pos_1, ketchup_ori_1)
    
    panda_robot_start_pos = [0, 0, 0.02+table_pos_1[2]]
    panda_robot_start_ori = [0, 0, 0, 1]
    panda_robot_id = p_env.loadURDF(os.path.expanduser("franka_panda/panda.urdf"), panda_robot_start_pos, panda_robot_start_ori, useFixedBase = 1)
    


    # joint_states = [-0.41631911936413146, 0.825290742999629, -0.07070329878754492, -2.1335976389165348, 1.0842230853806765, 1.497383708821378, 0.9379512580567759, 0.00026791999698616564, 0.00026791999698616564]
    
    
    # set_real_robot_JointPosition(p_env, panda_robot_id, joint_states)
    # for i in range(240):
    #     p_env.stepSimulation()
    #     time.sleep(1/240)
    # num = p_env.getNumJoints(panda_robot_id)
    # for i in range(num):
    #     joint_state = p_env.getJointState(panda_robot_id, i)
    #     print("joint_state:", joint_state)
    # print(num)
    return p_env, panda_robot_id


def for_loop(pybullet_env_id_list, robot_id_list):
    for env_index in range(env_num):   
        set_real_robot_JointPosition(pybullet_env_id_list[env_index], robot_id_list[env_index])

def cpu_thread(pybullet_env_id_list, robot_id_list):
    threads_obs = []
    for index, pybullet_env in enumerate(pybullet_env_id_list):
        thread_obs = threading.Thread(target=set_real_robot_JointPosition, args=(pybullet_env, robot_id_list[index]))
        thread_obs.start()
        threads_obs.append(thread_obs)
    for thread_obs in threads_obs:
        thread_obs.join()

def cpu_thread_env(pybullet_env_id_list, robot_id_list):
    threads_obs = []
    for index, pybullet_env in enumerate(pybullet_env_id_list):
        thread_obs = threading.Thread(target=set_real_robot_JointPosition, args=(pybullet_env, robot_id_list[index]))
        thread_obs.start()
        threads_obs.append(thread_obs)
    for thread_obs in threads_obs:
        thread_obs.join()

def cpu_multi(pybullet_env_id_list, robot_id_list):
    processes = []
    parallel = multiprocessing.Process
    for index, pybullet_env in enumerate(pybullet_env_id_list):
        process = parallel(target=set_real_robot_JointPosition, args=(pybullet_env, robot_id_list[index]))
        processes.append(process)
    for i in range(len(pybullet_env_id_list)):
        processes[i].start()
    for process in processes:
        process.join()

def cpu_multi_env(pybullet_env_id_list, robot_id_list):
    processes = []
    parallel = multiprocessing.Process
    for index, pybullet_env in enumerate(pybullet_env_id_list):
        process = parallel(target=set_real_robot_JointPosition, args=(pybullet_env, robot_id_list[index]))
        processes.append(process)
    for i in range(len(pybullet_env_id_list)):
        processes[i].start()
    for process in processes:
        process.join()

        
env_num = 100
pybullet_env_id_list = [0] * env_num
robot_id_list = [0] * env_num
CPU_Paralle_flag = "multiprocess" # thread/multiprocess/normal

time_before_create = time.time()
for env_index in range(env_num):   
    p_env, robot_id = create_pybullet_world()
    pybullet_env_id_list[env_index] = p_env
    robot_id_list[env_index] = robot_id
    
time_after_create = time.time()
print("")
print("======================")
print(time_after_create - time_before_create)
print("======================")

time_before = time.time()
if CPU_Paralle_flag == "normal":
    for_loop(pybullet_env_id_list, robot_id_list)
elif CPU_Paralle_flag == "thread":
    cpu_thread(pybullet_env_id_list, robot_id_list)
elif CPU_Paralle_flag == "multiprocess":
    cpu_multi(pybullet_env_id_list, robot_id_list)
time_after = time.time()
print("")
print("======================")
print(time_after - time_before)
print("======================")


# joint_states = [-0.41631911936413146, 0.825290742999629, -0.07070329878754492, -2.1335976389165348, 1.0842230853806765, 1.497383708821378, 0.9379512580567759, 0.00026791999698616564, 0.00026791999698616564]
# set_real_robot_JointPosition(pybullet, panda_robot_id, joint_states)
# for i in range(70):
#     pybullet.stepSimulation()
#     time.sleep(1/240)



# while True:
#     p_env.stepSimulation()
#     time.sleep(1/240)
