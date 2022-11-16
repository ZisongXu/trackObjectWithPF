#!/usr/bin/python3
#Class of franka robot move
class Franka_robot():
    def __init__(self,franka_robot_id, p_visualisation):
        self.franka_robot_id = franka_robot_id
        self.p_visualisation = p_visualisation
        
    def fanka_robot_move(self,targetPositionsJoints):
        self.setJoinVisual(self.franka_robot_id,targetPositionsJoints)
        
    def setJoinVisual(self,robot, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                self.p_visualisation.resetJointState(robot,
                                                     joint_index+2,
                                                     targetValue=position[joint_index])
            else:
                self.p_visualisation.resetJointState(robot,
                                                     joint_index,
                                                     targetValue=position[joint_index])
                
    def setJointPosition(self,robot, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                self.p_visualisation.setJointMotorControl2(robot,
                                                           joint_index+2,
                                                           self.p_visualisation.POSITION_CONTROL,
                                                           targetPosition=position[joint_index])
            else:
                self.p_visualisation.setJointMotorControl2(robot,
                                                           joint_index,
                                                           self.p_visualisation.POSITION_CONTROL,
                                                           targetPosition=position[joint_index])
