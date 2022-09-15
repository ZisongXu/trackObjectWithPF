import sys
import time
import rospy
from sensor_msgs.msg import JointState
import geometry_msgs.msg
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal
import moveit_commander
import moveit_msgs.msg
from numpy.linalg import inv
from numpy.linalg import pinv
import numpy as np
import math
from pyquaternion import Quaternion
import copy

class Panda:
    def __init__(self):
        self.gripper_speed = 0.1
        self.subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)
        
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
                                                            
        self.gripper_move_client = actionlib.SimpleActionClient("franka_gripper/move", MoveAction)
        self.gripper_move_client.wait_for_server(timeout=rospy.Duration(secs=5))
        self.moveit_robot_commander = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander('panda_arm')
        
        self.move_step = 0.01
        
        
        time.sleep(1)

    def move_to(self, x, y, z):
        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.w = 1.0
        pose_target.position.x = x
        pose_target.position.y = y
        pose_target.position.z = z

        

        
        joint_name = self.moveit_group.get_joints()
        #print("joint_name:",joint_name)
        
        self.moveit_group.set_pose_target(pose_target)
        self.moveit_group.go(wait=True)
        self.moveit_group.clear_pose_targets()
        
        #current_pose = self.moveit_group.get_current_pose()
        #print("current_pose:",current_pose)
        
        #current_joint_values2 = self.moveit_group.get_current_joint_values()
        #print("current_joint_values2:",current_joint_values2)
        
        
        
        
    def move_to_target_joints(self,target_joint):
        self.moveit_group.set_joint_value_target(target_joint)
        self.moveit_group.go(wait=True)
        self.moveit_group.clear_pose_targets()
        current_pose = self.moveit_group.get_current_pose()
        #print("current_pose:",current_pose)
        current_joint_values3 = self.moveit_group.get_current_joint_values()
        #print("current_joint_values3:",current_joint_values3)
        
    def move_straight_line_jac(self):
        org_current_pose = self.moveit_group.get_current_pose()
        org_end_effector_ori_x = org_current_pose.pose.orientation.x
        org_end_effector_ori_y = org_current_pose.pose.orientation.y
        org_end_effector_ori_z = org_current_pose.pose.orientation.z
        org_end_effector_ori_w = org_current_pose.pose.orientation.w
        org_end_effector_ori_list = [org_end_effector_ori_w,
                                     org_end_effector_ori_x,
                                     org_end_effector_ori_y,
                                     org_end_effector_ori_z]
        step_flag = 0
        
        targetPositionsJoints = self.moveit_group.get_current_joint_values()
        
        for i in range(60):	
            step_flag = step_flag + 1
            current_pose = self.moveit_group.get_current_pose()
            #print("current_pose:",current_pose)
            #position
            end_effector_x = current_pose.pose.position.x
            end_effector_y = current_pose.pose.position.y
            end_effector_z = current_pose.pose.position.z
            end_effector_pos_list = [end_effector_x,end_effector_y,end_effector_z]
            #orientation
            end_effector_ori_x = current_pose.pose.orientation.x
            end_effector_ori_y = current_pose.pose.orientation.y
            end_effector_ori_z = current_pose.pose.orientation.z
            end_effector_ori_w = current_pose.pose.orientation.w
            end_effector_ori_list = [end_effector_ori_w,end_effector_ori_x,end_effector_ori_y,end_effector_ori_z]
        
            #rotation
            end_effector_ori_list_qu = Quaternion(end_effector_ori_list)
            end_effector_ori_list_qu_inv = end_effector_ori_list_qu.inverse
            org_end_effector_ori_list_qu = Quaternion(org_end_effector_ori_list)
            quaternion_multiple = org_end_effector_ori_list_qu * end_effector_ori_list_qu_inv
            print("quaternion_multiple:",quaternion_multiple)
            w_cos_theta_over_2 = quaternion_multiple.w
            sin_theta_over_2 = math.sqrt(quaternion_multiple.x ** 2 + quaternion_multiple.y ** 2 + quaternion_multiple.z ** 2)
            theta_over_2 = math.atan2(sin_theta_over_2,w_cos_theta_over_2)
            theta =  theta_over_2 * 2
            
            #motion
            jacobian_matrix = self.moveit_group.get_jacobian_matrix(targetPositionsJoints,end_effector_pos_list)
            #print("jacobian_matrix:",jacobian_matrix)
            jac_t = [jacobian_matrix[0],jacobian_matrix[1],jacobian_matrix[2]]
            jac_r = [jacobian_matrix[3],jacobian_matrix[4],jacobian_matrix[5]]
            #print("jac_t:",jac_t)
            #print("jac_r:",jac_r)
            jac_t_pi = pinv(jac_t)
            #print("jac_t_pi:",jac_t_pi)
            movement_vector = [0.0,0.0,-self.move_step]
            expected_delta_q_dot_1 = list(np.dot(jac_t_pi, movement_vector))
            print("expected_delta_q_dot_1:",expected_delta_q_dot_1)
            targetPositionsJoints = list(np.sum([expected_delta_q_dot_1, targetPositionsJoints], axis = 0))
            
            if sin_theta_over_2 != 0:
                ax = quaternion_multiple.x / sin_theta_over_2
                bx = quaternion_multiple.y / sin_theta_over_2
                cx = quaternion_multiple.z / sin_theta_over_2
                axis = [ax*theta,bx*theta,cx*theta]
                expected_rotation = axis
                jac_r_pi = pinv(jac_r)
                expected_delta_q_dot_2 = list(np.dot(jac_r_pi, expected_rotation))
                
            else:
                expected_delta_q_dot_2 = [0,0,0,0,0,0,0]
            print("expected_delta_q_dot_2:",expected_delta_q_dot_2)
            targetPositionsJoints = list(np.sum([expected_delta_q_dot_2, targetPositionsJoints], axis = 0))
            print("targetPositionsJoints:",targetPositionsJoints)
            self.move_to_target_joints(targetPositionsJoints)
            
    def move_straight_line_ccp(self):
        waypoints = []
        
        scale = 1.0
        wpose = self.moveit_group.get_current_pose().pose
        for i in range(5):
            
            wpose.position.z = wpose.position.z - scale * 0.03
            waypoints.append(copy.deepcopy(wpose))
        
        '''
        scale = 1.0
        wpose = self.moveit_group.get_current_pose().pose
        wpose.position.z = wpose.position.z - scale * 0.1
        wpose.position.y = wpose.position.y + scale * 0.2
        waypoints.append(copy.deepcopy(wpose))
        
        wpose.position.x = wpose.position.x + scale * 0.1
        waypoints.append(copy.deepcopy(wpose))
        
        wpose.position.y = wpose.position.y - scale * 0.1
        waypoints.append(copy.deepcopy(wpose))
        '''
        
        (plan,fraction) = self.moveit_group.compute_cartesian_path(waypoints,0.01,0.0)
        self.moveit_group.execute(plan, wait=True)
        
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.moveit_group.get_current_state()
        display_trajectory.trajectory.append(plan)
        print("display_trajectory:",display_trajectory)
        self.display_trajectory_publisher.publish(display_trajectory)
        
        self.moveit_group.stop()

    
    def joint_values_callback(self, msg):
        self.current_joint_values = msg.position

    def set_gripper_openning(self, value):
        move_goal = MoveGoal(width=value, speed=self.gripper_speed)
        self.gripper_move_client.send_goal(move_goal)

    def fully_open_gripper(self):
    	self.set_gripper_openning(0.08)

    def fully_close_gripper(self):
        self.set_gripper_openning(0.0)


if __name__ == '__main__':
    rospy.init_node('panda_demo')
    moveit_commander.roscpp_initialize(sys.argv)
    panda = Panda()

    input('Press [ENTER] to start')

    panda.fully_open_gripper()
    
    #panda.move_to(x=-0.2, y=0.0, z=1.9)
    #panda.move_straight_line_jac()
    targetPositionsJoints_test = [0.0,-0.08,0.0,-1.65,0.0,1.58,0.870]
    panda.move_to_target_joints(targetPositionsJoints_test)
    #panda.move_straight_line_ccp()
    
    panda.fully_close_gripper()
