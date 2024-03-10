import pybullet 
import time
import pybullet_data
from pybullet_utils import bullet_client
import numpy as np
import os
import matplotlib  
from matplotlib import pyplot as plt
from simulated_camera import SimulatedCamera
import math
import jax
def set_real_robot_JointPosition(pybullet_env, robot, position):
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
                                               
def plot(depth_image):
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(depth_image, cmap="gray")
    axs[0].set_title('Depth image')
    plt.plot(label='ax2')
    plt.show()
    
def create_pybullet_world():
    client_id = pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setGravity(0,0,-9.81) 
    pybullet.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])#转变视角

    plane_id = pybullet.loadURDF("plane.urdf")
    pos, ori = pybullet.getBasePositionAndOrientation(plane_id, physicsClientId=client_id)
    print(pos)
    table_pos_1 = [0.46, -0.01, 0.710]
    table_ori_1 = pybullet.getQuaternionFromEuler([0,0,0])
    table_id_1 = pybullet.loadURDF("table.urdf", table_pos_1, table_ori_1, useFixedBase = 1)

    barry_pos_1 = [-0.694, 0.443, 0.895]
    barry_ori_1 = pybullet.getQuaternionFromEuler([0,math.pi/2,0])
    barry_id_1 = pybullet.loadURDF("barrier.urdf", barry_pos_1, barry_ori_1, useFixedBase = 1)
    
    barry_pos_2 = [-0.694, -0.607, 0.895]
    barry_ori_2 = pybullet.getQuaternionFromEuler([0,math.pi/2,0])
    barry_id_2 = pybullet.loadURDF("barrier.urdf", barry_pos_2, barry_ori_2, useFixedBase = 1)
    
    barry_pos_3 = [0.459, -0.972, 0.895]
    barry_ori_3 = pybullet.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
    barry_id_3 = pybullet.loadURDF("barrier.urdf", barry_pos_3, barry_ori_3, useFixedBase = 1)

    barry_pos_4 = [-0.549, 0.61, 0.895]
    barry_ori_4 = pybullet.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
    # barry_id_4 = pybullet.loadURDF("barrier.urdf", barry_pos_4, barry_ori_4, useFixedBase = 1)
    
    barry_pos_5 = [0.499, 0.61, 0.895]
    barry_ori_5 = pybullet.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
    # barry_id_5 = pybullet.loadURDF("barrier.urdf", barry_pos_5, barry_ori_5, useFixedBase = 1)
    
    board_pos_1 = [0.274, 0.581, 0.87575]
    board_ori_1 = pybullet.getQuaternionFromEuler([math.pi/2,math.pi/2,0])
    board_id_1 = pybullet.loadURDF("board.urdf", board_pos_1, board_ori_1, useFixedBase = 1)
    
    ketchup_pos_1 = [0.274, 0.381, 0.085+table_pos_1[2]]
    ketchup_ori_1 = pybullet.getQuaternionFromEuler([math.pi/2,0,0])
    ketchup_id_1 = pybullet.loadURDF(os.path.expanduser("YcbTomatoSoupCan/model.urdf"), ketchup_pos_1, ketchup_ori_1)
    
    # ketchup_pos_1 = [0.274, 0.381, 0.0743+table_pos_1[2]]
    # ketchup_ori_1 = pybullet.getQuaternionFromEuler([math.pi/2,0,0])
    # ketchup_id_1 = pybullet.loadURDF(os.path.expanduser("tomato_ketchup/tomato_ketchup.urdf"), ketchup_pos_1, ketchup_ori_1)
    
    # ketchup_pos_1 = [0.274, 0.381, 0.11+table_pos_1[2]]
    # ketchup_ori_1 = pybullet.getQuaternionFromEuler([math.pi/2,0,0])
    # ketchup_id_1 = pybullet.loadURDF(os.path.expanduser("orange_juice/orange_juice.urdf"), ketchup_pos_1, ketchup_ori_1)
    
    panda_robot_start_pos = [0, 0, 0.02+table_pos_1[2]]
    panda_robot_start_ori = [0, 0, 0, 1]
    panda_robot_id = pybullet.loadURDF(os.path.expanduser("franka_panda/panda.urdf"), panda_robot_start_pos, panda_robot_start_ori, useFixedBase = 1)

    # joint_states = [-0.41631911936413146, 0.825290742999629, -0.07070329878754492, -2.1335976389165348, 1.0842230853806765, 1.497383708821378, 0.9379512580567759, 0.00026791999698616564, 0.00026791999698616564]
    # set_real_robot_JointPosition(pybullet, panda_robot_id, joint_states)
    # for i in range(70):
    #     pybullet.stepSimulation()
    #     time.sleep(1/240)
    return client_id, panda_robot_id

client_id, robot_id = create_pybullet_world()

width = 848
height = 480
near_val = 0.01
far_val = 100
robot_to_camera_pos = [0.372, -0.037, -0.013]
robot_to_camera_ori = [-0.457, -0.477, 0.535, 0.527]
robot_to_camera_3_3 = np.array(pybullet.getMatrixFromQuaternion(robot_to_camera_ori)).reshape(3, 3)
robot_to_camera_3_4 = np.c_[robot_to_camera_3_3, robot_to_camera_pos]  # Add position to create 3x4 matrix
robot_to_camera_4_4 = np.r_[robot_to_camera_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
# Calculate camera's world transformation matrix
robot_to_camera_transform = [
        [-0.07599739,  0.48345446, -0.87206432,  1.15092245],
        [ 0.9966351,   0.00989689, -0.08136667,  0.12411099],
        [-0.03070636, -0.87531356, -0.48257982,  0.39669871],
        [ 0.,          0.,          0.,          1.        ]]
robot_to_camera_transform = robot_to_camera_4_4
simulated_camera = SimulatedCamera(client_id, robot_to_camera_transform, width, height, near_val, far_val)

depth_image = simulated_camera.generate_depth_image(robot_id)
plot(depth_image)
while True:
    print("wgat")
    pybullet.stepSimulation()
    time.sleep(1/240)
