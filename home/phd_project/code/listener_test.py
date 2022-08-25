# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
#ROS
import itertools
import os.path

import rospy
import rospkg
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Int8
from std_msgs.msg import ColorRGBA, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point,PointStamped,PoseStamped,Quaternion,TransformStamped, Vector3
import tf
import tf.transformations as transformations
from visualization_msgs.msg import Marker

#pybullet
import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
import math
import random
import copy
import os
'''
def callback(data):
    #pos
    x_pos = data.pose.position.x
    y_pos = data.pose.position.y
    z_pos = data.pose.position.z
    #ori
    x_ori = data.pose.orientation.x
    y_ori = data.pose.orientation.y
    z_ori = data.pose.orientation.z
    w_ori = data.pose.orientation.w
    
def listener():
    rospy.init_node('listener',anonymous=True)
    rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose',PoseStamped,callback)
    rospy.spin()
    
if __name__ == '__main__':
    listener()
'''

#visualisation_model
p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER)
p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
#p_visualisation.setGravity(0,0,-9.81)
p_visualisation.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])
plane_id = p_visualisation.loadURDF("plane.urdf")


class Target_object():
    def __init__(self,target_object_id):
        self.target_object_id = target_object_id
    def target_object_move(self,pos,ori):
        p_visualisation.resetBasePositionAndOrientation(self.target_object_id,pos,ori)


class InitialRealworldModel():
    def __init__(self,joint_pos=0):
        self.flag = 0
        self.joint_pos = joint_pos
    def initial_robot(self,robot_pos,robot_orientation = [0,0,0,1]):
        #robot_orientation = p_visualisation.getQuaternionFromEuler(robot_euler)
        real_robot_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                 robot_pos,
                                                 robot_orientation,
                                                 useFixedBase=1)
        '''
        self.set_real_robot_JointPosition(p_visualisation,real_robot_id,self.joint_pos)
        for i in range(240):
            p_visualisation.stepSimulation()
            time.sleep(1./240.)
        '''
        return real_robot_id
    def initial_target_object(self,object_pos,object_orientation = [0,0,0,1]):
        #object_orientation = p_visualisation.getQuaternionFromEuler(object_euler)
        real_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_object_small_test.urdf"),
                                                  object_pos,
                                                  object_orientation)
        return real_object_id
    def set_real_robot_JointPosition(self,pybullet_simulation_env,robot, position):
        #num_joints = 7
        num_joints = 9
        for joint_index in range(num_joints): 
            if joint_index == 7 or joint_index == 8:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index+2,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])                
            else:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])           
class Ros_listener():
    def __init__(self):
        self.joint_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)
        self.robot_pose = rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose',PoseStamped, self.robot_pose_callback)
        self.object_pose = rospy.Subscriber('/mocap/rigid_bodies/zisongObject/pose',PoseStamped, self.object_pose_callback)
        self.current_joint_values = [-1.57,0.0,0.0,-2.8,1.7,1.57,1.1]
        #self.robot_pos = [ 0.139080286026,
        #                  -0.581342339516,
        #                   0.0238141193986]
        #x,y,z,w
        #self.robot_ori = [ 0.707254290581,
        #                   0.0115503482521,
        #                  -0.0140119809657,
        #                  -0.706726074219]
                          
        #self.object_pos = [ 0.504023790359,
        #                   -0.214561194181,
        #                    0.0601389780641]
        #x,y,z,w
        #self.object_ori = [-0.51964700222,
        #                   -0.476704657078,
        #                    0.490200251342,
        #                    0.512272834778]
        
        #self.object_ori = [0,0,0,1]        
        
    def joint_values_callback(self, msg):
        self.current_joint_values = list(msg.position)
    def robot_pose_callback(self, data):
        #print("in robot pose callback")
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.robot_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.robot_ori = [x_ori,y_ori,z_ori,w_ori]
    def object_pose_callback(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori,y_ori,z_ori,w_ori]

def rotation_4_4_to_transformation_4_4(rotation_4_4,pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4
def robot_T_object_tran(robot_T_object):
    y = robot_T_object[1][3]
    robot_T_object[1][3] = robot_T_object[2][3]
    robot_T_object[2][3] = -y
    return robot_T_object
def ori_matrix_change(robot_T_object):
    x,y,z=transformations.euler_from_matrix(robot_T_object)
    mid = y
    y = z
    z = -mid
    robot_T_object = transformations.euler_matrix(x,y,z) 
    return robot_T_object
def compute_distance_between_2_points_3D(pos1,pos2):
    x1=pos1[0]
    y1=pos1[1]
    z1=pos1[2]
    x2=pos2[0]
    y2=pos2[1]
    z2=pos2[2]
    x_d = x1-x2
    y_d = y1-y2
    z_d = z1-z2
    distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
    return distance  
            
if __name__ == '__main__':
    rospy.init_node('listen_joint_info')
    #build an object of class "Ros_listener"
    ros_listener = Ros_listener()
    time.sleep(0.5)
    current_joint_values = ros_listener.current_joint_values
    print(current_joint_values)
    
    #give some time to update the data
    time.sleep(0.5)
    init_robot_pos = ros_listener.robot_pos
    init_robot_ori = ros_listener.robot_ori 
    init_object_pos = ros_listener.object_pos
    init_object_ori = ros_listener.object_ori
    
    #compute transformation matrix
    input('Press [ENTER] to compute transformation matrix')
    robot_transformation_matrix = transformations.quaternion_matrix(init_robot_ori)
    print("robot_transformation_matrix:")
    print(robot_transformation_matrix)
    ow_T_robot = rotation_4_4_to_transformation_4_4(robot_transformation_matrix,init_robot_pos)
    print("ow_T_robot:")
    print(ow_T_robot)
    object_transformation_matrix = transformations.quaternion_matrix(init_object_ori)
    print("object_transformation_matrix:")
    print(object_transformation_matrix)
    ow_T_object = rotation_4_4_to_transformation_4_4(object_transformation_matrix,init_object_pos)
    print("ow_T_object:")
    print(ow_T_object)
    robot_T_ow = np.linalg.inv(ow_T_robot)
    robot_T_object = np.dot(robot_T_ow,ow_T_object)
    print("robot_T_object:")
    print(robot_T_object)
    '''
    pybullet_robot_pos = [ 0.139080286026,
                          -0.581342339516,
                           0.0238141193986]
    #x,y,z,w
    pybullet_robot_ori = [ 0.707254290581,
                           0.0115503482521,
                          -0.0140119809657,
                          -0.706726074219]
    '''
    
    pybullet_robot_pos = [0,0,0]
    pybullet_robot_ori = [0,0,0,1]
    #pybullet_robot_ori = [0.06146124,0.234234,-0.1341234,0.99810947]
    #pybullet_robot_ori = [0.05933772,0.22614109,-0.12948937,0.96362428]
    
    input('Press [ENTER] to compute the pose of object in the pybullet world')
    #init_object_pos = [0.567, -0.3642, 0.057]
    pybullet_robot_transformation_matrix = transformations.quaternion_matrix(pybullet_robot_ori)
    pw_T_robot = rotation_4_4_to_transformation_4_4(pybullet_robot_transformation_matrix,pybullet_robot_pos)
    pw_T_object = np.dot(pw_T_robot,robot_T_object)
    print("pw_T_object:")
    print(pw_T_object)
    pw_T_object_pos = [pw_T_object[0][3],
                       pw_T_object[1][3],
                       pw_T_object[2][3]]       

    pw_T_object_ori = transformations.quaternion_from_matrix(pw_T_object) 
    ow_distance = compute_distance_between_2_points_3D(init_robot_pos,init_object_pos)
    print("ow_distance:",ow_distance)
    pw_distance = compute_distance_between_2_points_3D(pybullet_robot_pos,pw_T_object_pos)
    print("pw_distance:",pw_distance)
                               
    input('Press [ENTER] to initial real world model')
    #build an object of class "InitialRealworldModel"
    real_world_object = InitialRealworldModel(ros_listener.current_joint_values)
    #initialize the real robot in the pybullet
    real_robot_id = real_world_object.initial_robot(robot_pos = pybullet_robot_pos,robot_orientation = pybullet_robot_ori)
    #initialize the real object in the pybullet
    real_object_id = real_world_object.initial_target_object(object_pos = pw_T_object_pos,object_orientation = pw_T_object_ori)    
    target_object = Target_object(real_object_id)
    flag = 0
    while True:
    
        #compute transformation matrix
        
        
        
        robot_transformation_matrix = transformations.quaternion_matrix(ros_listener.robot_ori )
        print("robot_transformation_matrix:")
        print(robot_transformation_matrix)
        object_transformation_matrix = transformations.quaternion_matrix(ros_listener.object_ori)
        print("object_transformation_matrix:")
        print(object_transformation_matrix)
                
        ow_T_robot = rotation_4_4_to_transformation_4_4(robot_transformation_matrix,ros_listener.robot_pos)
        print("ow_T_robot:")
        print(ow_T_robot)
        ow_T_object = rotation_4_4_to_transformation_4_4(object_transformation_matrix,ros_listener.object_pos)
        print("ow_T_object:")
        print(ow_T_object)
        robot_T_ow = np.linalg.inv(ow_T_robot)
        robot_T_object = np.dot(robot_T_ow,ow_T_object)
        #robot_T_object = ori_matrix_change(robot_T_object)
        #robot_T_object = rotation_4_4_to_transformation_4_4(robot_T_object,ros_listener.object_pos)
        print("robot_T_object_before:")
        print(robot_T_object)
        #robot_T_object = robot_T_object_tran(robot_T_object)  
 
        print("robot_T_object_after:")
        print(robot_T_object)
         
        
        
        
        
        
        
        
        pybullet_robot_transformation_matrix = transformations.quaternion_matrix(pybullet_robot_ori)
        pw_T_robot = rotation_4_4_to_transformation_4_4(pybullet_robot_transformation_matrix,pybullet_robot_pos)
        pw_T_object = np.dot(pw_T_robot,robot_T_object)
        #print("pw_T_object:")
        #print(pw_T_object)
        pw_T_object_pos = [pw_T_object[0][3],
                           pw_T_object[1][3],
                           pw_T_object[2][3]]    
                            
        x_rotation_matrix = [[1,                     0,                      0,0],
                             [0, math.cos(-math.pi/2.0),math.sin(-math.pi/2.0),0],
                             [0,-math.sin(-math.pi/2.0),math.cos(-math.pi/2.0),0],
                             [0,                      0,                     0,1]]
        x_rotation_matrix = np.array(x_rotation_matrix)
        pw_T_object_rot = np.dot(pw_T_object,x_rotation_matrix)
        #print("pw_T_object_rot:")
        #print(pw_T_object_rot)   
        
        #pw_T_object_ori = transformations.quaternion_from_matrix(pw_T_object) 
        ow_distance = compute_distance_between_2_points_3D(init_robot_pos,init_object_pos)
        #print("ow_distance:",ow_distance)
        pw_distance = compute_distance_between_2_points_3D(pybullet_robot_pos,pw_T_object_pos)
        #print("pw_distance:",pw_distance)
        
        #pw_T_object = ori_matrix_change(pw_T_object)
        pw_T_object_ori = transformations.quaternion_from_matrix(pw_T_object) 
        
        target_object.target_object_move(pw_T_object_pos,pw_T_object_ori)
        p_visualisation.stepSimulation()
        #time.sleep(1./240.)
        #continue
        #input('Press [ENTER] to compute the pose of object in the pybullet world')
        
        
        
        
        
