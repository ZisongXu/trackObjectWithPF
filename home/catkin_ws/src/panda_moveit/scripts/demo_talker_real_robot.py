#!/usr/bin/python3
import os
import sys
import time
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Int8
import geometry_msgs.msg
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal
from franka_gripper.msg import MoveAction, MoveGoal
import moveit_commander
import moveit_msgs.msg
from numpy.linalg import inv
from numpy.linalg import pinv
import numpy
import math
from pyquaternion import Quaternion
import copy
import tf
from tf2_msgs.msg import TFMessage
from tf import transformations as t


class Panda:
    def __init__(self):
        '''
        self.listener = tf.TransformListener()
        
        #self.gripper_speed = 0.1
        #self.subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)
        #self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',moveit_msgs.msg.DisplayTrajectory,queue_size=20)
        #self.joint_publisher = rospy.Publisher('joint_chatter',JointState,queue_size=10)
                                                         
        #self.gripper_move_client = actionlib.SimpleActionClient("franka_gripper/move", MoveAction)
        #self.gripper_move_client.wait_for_server(timeout=rospy.Duration(secs=5))
        #self.moveit_robot_commander = moveit_commander.RobotCommander()
        #self.moveit_scene = moveit_commander.PlanningSceneInterface()
        #self.moveit_group = moveit_commander.MoveGroupCommander('panda_arm')
        
        
        #==========================================
        
        self.target_marker = 'ar_marker_3'
        self.robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander('panda_arm')
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

        self.subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)

        self.gripper_speed = 0.1
        self.gripper_grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.gripper_grasp_client.wait_for_server()

        self.gripper_move_client = actionlib.SimpleActionClient("franka_gripper/move", MoveAction)
        self.gripper_move_client.wait_for_server()
        
        #==========================================
        
        self.move_step = 0.01
        
        self.home_cartesian_pose = ((0.3,0.0,0.6),(0.5,0.5,0.5,0.5))

        self.home_joint_configuration = [0.06434442693934876, -0.25159467341141767, -0.09634342054034868, -1.9197004701225044, -0.042293509957305124, 1.7079073634412554, 0.8450151283072738]
        self.dropoff_joint_configuration = [-0.6246857446330409, 0.2644096889929886, -0.15063071233574576, -1.409807436081744, 0.001742226868867874, 1.7051944175031448, 0.8014592822532559]

        self.objects_added_to_planning_scene_not_grasped = []
        self.objects_added_to_planning_scene_grasped = []

        self.ee_link = 'panda_hand_tcp'
        self.moveit_group.set_end_effector_link("panda_hand_tcp")
        self.moveit_group.set_pose_reference_frame("panda_link0")

        # add table
        table_size = (2.0,2.0,0.04) 
        table_name = "panda_table"
        table_pose = geometry_msgs.msg.PoseStamped()
        table_pose.header.frame_id = "panda_link0"
        table_pose.pose.orientation.w = 1.0
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.04 # 2 cm due to table height, 2 cm due to clamp
        self.moveit_scene.add_box(table_name,table_pose,table_size)
        time.sleep(1)
        self.moveit_scene.attach_box('panda_link0', table_name, touch_links=['panda_link1']) # Setting touchlinks to empty, because assuming link0 collisions will be ignored anyways
        
        time.sleep(1)
        '''
        
        self.listener = tf.TransformListener()
        self.target_marker = 'ar_marker_3'
        self.robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander('panda_arm')
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

        self.subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)

        self.gripper_speed = 0.1
        self.gripper_grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.gripper_grasp_client.wait_for_server()

        self.gripper_move_client = actionlib.SimpleActionClient("franka_gripper/move", MoveAction)
        self.gripper_move_client.wait_for_server()
 
        # home cartesian pose is when the end-effector rotation is such that
        # the ee z-axis points along x-axis of robot (assumed world) frame
        # the ee x-axis points along y-axis of robot frame
        # the ee y-axis points along z-axis of robot frame
        # which gives the 0.5X4 quaternion below.
        self.home_cartesian_pose = ((0.3,0.0,0.6),(0.5,0.5,0.5,0.5))

        self.home_joint_configuration = [0.06434442693934876, -0.25159467341141767, -0.09634342054034868, -1.9197004701225044, -0.042293509957305124, 1.7079073634412554, 0.8450151283072738]
        self.dropoff_joint_configuration = [-0.6246857446330409, 0.2644096889929886, -0.15063071233574576, -1.409807436081744, 0.001742226868867874, 1.7051944175031448, 0.8014592822532559]

        self.objects_added_to_planning_scene_not_grasped = []
        self.objects_added_to_planning_scene_grasped = []

        self.ee_link = 'panda_hand_tcp'
        self.moveit_group.set_end_effector_link("panda_hand_tcp")
        self.moveit_group.set_pose_reference_frame("panda_link0")

        # add table
        table_size = (2.0,2.0,0.04) 
        table_name = "panda_table"
        table_pose = geometry_msgs.msg.PoseStamped()
        table_pose.header.frame_id = "panda_link0"
        table_pose.pose.orientation.w = 1.0
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.04 # 2 cm due to table height, 2 cm due to clamp
        self.moveit_scene.add_box(table_name,table_pose,table_size)
        time.sleep(1)
        self.moveit_scene.attach_box('panda_link0', table_name, touch_links=['panda_link1']) # Setting touchlinks to empty, because assuming link0 collisions will be ignored anyways
        time.sleep(1)
        '''
        # add bin 
        bin_size = (0.55, 0.40, 0.25) 
        bin_name = "panda_bin"
        bin_pose = geometry_msgs.msg.PoseStamped()
        bin_pose.header.frame_id = "panda_link0"
        bin_pose.pose.orientation.w = 1.0
        bin_pose.pose.position.x = 0.45
        bin_pose.pose.position.y = -0.45
        bin_pose.pose.position.z = 0.135
        self.moveit_scene.add_box(bin_name,bin_pose,bin_size)
        time.sleep(1)
        self.moveit_scene.attach_box('panda_link0', bin_name, touch_links=[]) # Setting touchlinks to empty, because assuming link0 collisions will be ignored anyways
        time.sleep(1)
        '''
        
        
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
        

        
        #current_joint_values2 = self.moveit_group.get_current_joint_values()
        #print("current_joint_values2:",current_joint_values2)
        
        
        
        
    def move_to_target_joints(self,target_joint):
        self.moveit_group.set_joint_value_target(target_joint)
        self.moveit_group.go(wait=True)
        self.moveit_group.clear_pose_targets()
        
        
    def move_straight_line_jac(self):
        end_effector_link = self.moveit_group.get_end_effector_link()
        print(end_effector_link)
        
        org_current_pose = self.moveit_group.get_current_pose()
        org_end_effector_ori_x = org_current_pose.pose.orientation.x
        org_end_effector_ori_y = org_current_pose.pose.orientation.y
        org_end_effector_ori_z = org_current_pose.pose.orientation.z
        org_end_effector_ori_w = org_current_pose.pose.orientation.w
        org_link_ori = [org_end_effector_ori_w,
                        org_end_effector_ori_x,
                        org_end_effector_ori_y,
                        org_end_effector_ori_z]
        step_flag = 0
        move_step = 0.001
        targetPositionsJoints = self.moveit_group.get_current_joint_values()
        print("targetPositionsJoints:",targetPositionsJoints)
        org_link_pos = [org_current_pose.pose.position.x,
                        org_current_pose.pose.position.y,
                        org_current_pose.pose.position.z]
        last_link_pos = org_link_pos
        print("org_link_pos:",org_link_pos)
        
        for i in range(10):	
            step_flag = step_flag + 1
            current_pose = self.moveit_group.get_current_pose()
            #position
            end_effector_x = current_pose.pose.position.x
            end_effector_y = current_pose.pose.position.y
            end_effector_z = current_pose.pose.position.z
            curr_link_pos = [end_effector_x,end_effector_y,end_effector_z]
            delta_x = org_link_pos[0] + step_flag * 0 - curr_link_pos[0]
            delta_y = org_link_pos[1] + step_flag * move_step - curr_link_pos[1]
            delta_z = org_link_pos[2] + step_flag * 0 - curr_link_pos[2]
            movement_vector = [delta_x,delta_y,delta_z]
            #orientation
            end_effector_ori_x = current_pose.pose.orientation.x
            end_effector_ori_y = current_pose.pose.orientation.y
            end_effector_ori_z = current_pose.pose.orientation.z
            end_effector_ori_w = current_pose.pose.orientation.w
            curr_link_ori = [end_effector_ori_w,end_effector_ori_x,end_effector_ori_y,end_effector_ori_z]
            #rotation
            curr_link_ori_qu = Quaternion(curr_link_ori)
            curr_link_ori_qu_inv = curr_link_ori_qu.inverse
            org_link_ori_qu = Quaternion(org_link_ori)
            quaternion_multiple = org_link_ori_qu * curr_link_ori_qu_inv
            w_cos_theta_over_2 = quaternion_multiple.w
            sin_theta_over_2 = math.sqrt(quaternion_multiple.x ** 2 + quaternion_multiple.y ** 2 + quaternion_multiple.z ** 2)
            theta_over_2 = math.atan2(sin_theta_over_2,w_cos_theta_over_2)
            theta =  theta_over_2 * 2
            
            #motion
            jacobian_matrix = self.moveit_group.get_jacobian_matrix(targetPositionsJoints,curr_link_pos)
            jac_t = [jacobian_matrix[0],jacobian_matrix[1],jacobian_matrix[2]]
            jac_r = [jacobian_matrix[3],jacobian_matrix[4],jacobian_matrix[5]]
            jac_t_pi = pinv(jac_t)
            expected_delta_q_dot_1 = list(numpy.dot(jac_t_pi, movement_vector))
            targetPositionsJoints = list(numpy.sum([expected_delta_q_dot_1, targetPositionsJoints], axis = 0))
            
            if sin_theta_over_2 != 0:
                ax = quaternion_multiple.x / sin_theta_over_2
                bx = quaternion_multiple.y / sin_theta_over_2
                cx = quaternion_multiple.z / sin_theta_over_2
                axis = [ax*theta,bx*theta,cx*theta]
                expected_rotation = axis
                jac_r_pi = pinv(jac_r)
                expected_delta_q_dot_2 = list(numpy.dot(jac_r_pi, expected_rotation))
                
            else:
                expected_delta_q_dot_2 = [0,0,0,0,0,0,0]
            targetPositionsJoints = list(numpy.sum([expected_delta_q_dot_2, targetPositionsJoints], axis = 0)) 
            self.move_to_target_joints(targetPositionsJoints)
        return
    
    def move_through_waypoints(self, end_effector_waypoints):
        plan, _ = self.moveit_group.compute_cartesian_path(end_effector_waypoints, 0.01, 0.0)
        trajectory = plan.joint_trajectory.points
        self.moveit_group.go(trajectory[-1].positions, wait=True)
        
    def move_straight_line_ccp(self):
        waypoints = []
        
        scale = 1.0
        wpose = self.moveit_group.get_current_pose().pose
        #be careful inertance
        for i in range(5):
            
            wpose.position.x = wpose.position.x + scale * 0.01
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
        joint = self.moveit_group.get_current_joint_values()
        joint_state_info = JointState()
        #joint_state_info.header = Header()
        #joint_state_info.header.stamp = rospy.Time.now()
        joint_state_info.position = joint
        #self.joint_publisher.publish(joint_state_info)
        self.moveit_group.stop()

    def move_back_straight_line_ccp(self):
        waypoints = []
        scale = 1.0
        wpose = self.moveit_group.get_current_pose().pose
        #be careful inertance
        for i in range(5):
            wpose.position.x = wpose.position.x + scale * (-0.01)
            waypoints.append(copy.deepcopy(wpose))
        
        (plan,fraction) = self.moveit_group.compute_cartesian_path(waypoints,0.01,0.0)
        self.moveit_group.execute(plan, wait=True)
        joint = self.moveit_group.get_current_joint_values()
        joint_state_info = JointState()
        #joint_state_info.header = Header()
        #joint_state_info.header.stamp = rospy.Time.now()
        joint_state_info.position = joint
        #self.joint_publisher.publish(joint_state_info)
        self.moveit_group.stop()

    def move_y_straight_line_ccp(self):
        waypoints = []
        
        scale = 1.0
        wpose = self.moveit_group.get_current_pose().pose
        print("wpose")
        print(wpose)
        # be careful inertance
        for i in range(1):
            wpose.position.y = wpose.position.y + scale * (-0.01)
            waypoints.append(copy.deepcopy(wpose))
                
        (plan,fraction) = self.moveit_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        self.moveit_group.execute(plan, wait=True)
        joint = self.moveit_group.get_current_joint_values()
        joint_state_info = JointState()
        #joint_state_info.header = Header()
        #joint_state_info.header.stamp = rospy.Time.now()
        joint_state_info.position = joint
        #self.joint_publisher.publish(joint_state_info)
        self.moveit_group.stop()
        
    def move_minus_y_straight_line_ccp(self):
        waypoints = []
        
        scale = 1.0
        wpose = self.moveit_group.get_current_pose().pose
        #be careful inertance
        for i in range(1):
            wpose.position.y = wpose.position.y + scale * (-0.01)
            waypoints.append(copy.deepcopy(wpose))
                
        (plan,fraction) = self.moveit_group.compute_cartesian_path(waypoints,0.01,0.0)
        self.moveit_group.execute(plan, wait=True)
        joint = self.moveit_group.get_current_joint_values()
        joint_state_info = JointState()
        #joint_state_info.header = Header()
        #joint_state_info.header.stamp = rospy.Time.now()
        joint_state_info.position = joint
        #self.joint_publisher.publish(joint_state_info)
        self.moveit_group.stop()
    
    def joint_values_callback(self, msg):
        self.current_joint_values = msg.position
        #print("self.current_joint_values:",self.current_joint_values)

    def set_gripper_openning(self, value):
        move_goal = MoveGoal(width=value, speed=self.gripper_speed)
        self.gripper_move_client.send_goal(move_goal)

    def fully_open_gripper(self):
        self.set_gripper_openning(0.08)

    def fully_close_gripper(self):
        self.set_gripper_openning(0.0)

print(0)
if __name__ == '__main__':
    rospy.init_node('panda_demo')
    moveit_commander.roscpp_initialize(sys.argv)
    panda = Panda()
    #input('Press [ENTER] to start')
    
    
    #be careful inertance

    
    
    #panda.fully_open_gripper()
    #panda.fully_close_gripper()
    
    #panda.move_to(x=-0.2, y=0.0, z=1.9)
    #panda.move_straight_line_jac()
    #for i in range(8):
    #    panda.move_y_straight_line_ccp()
    #    time.sleep(1)
    # targetPositionsJoints_test = [0.0,-0.08,0.0,-1.65,0.0,1.58,0.870]
    # panda.move_to_target_joints(targetPositionsJoints_test)
    #targetPositionsJoints_test = [-2.5,0.0,0.0,-1.0,1.7,1.57,0.6]
    #panda.move_to_target_joints(targetPositionsJoints_test)
    #cube
    #targetPositionsJoints_test = [-0.41429875365684077, 0.7531559819003992, -0.09432195875240348, -2.172390429781196, 1.080601253367148, 1.499608373509513, 0.9228163020693847]
    #panda.move_to_target_joints(targetPositionsJoints_test)
    #cheezit
    #targetPositionsJoints_test = [-0.37611809028263904, 1.0542567263506848, 0.0039623287396545115, -1.514141659285099, 1.2019493953534113, 1.4195338642218471, 1.2938332387612945]
    #panda.move_to_target_joints(targetPositionsJoints_test)
    #test
    #targetPositionsJoints_test = [-0.39268936579891744, 1.0749383543375984, -0.04246610496406864, -1.4827234936764365, 1.1450266403932545, 1.4720578482110835, 1.0963488744405498]
    #panda.move_to_target_joints(targetPositionsJoints_test)
    #panda.move_straight_line_ccp()
    #panda.move_back_straight_line_ccp()
    #panda.move_y_straight_line_ccp()
    #panda.move_minus_y_straight_line_ccp
    #panda.fully_close_gripper()
    #cylinder
    panda.fully_open_gripper()
    panda.fully_close_gripper()
    # sys.exit()
    # for index in range(5):
    #     wpose = panda.moveit_group.get_current_pose().pose
    #     waypoints = []
    #     wpose.position.y += 0.1
    #     # wpose.position.z += 0.08
    #     # #wpose.position.y += 0.15
    #     waypoints.append(copy.deepcopy(wpose))    
    #     panda.move_through_waypoints(waypoints)
    cheezit_flag = True
    soup_flag = False
    if cheezit_flag == True:
        time = 4
    if soup_flag == True:
        time = 4
        
        
#    for index in range(time):
#        wpose = panda.moveit_group.get_current_pose().pose
#        waypoints = []
#        wpose.position.y -= 0.1
#        # wpose.position.z += 0.08
#        # wpose.position.y -= 0.15
#        waypoints.append(copy.deepcopy(wpose))    
#        panda.move_through_waypoints(waypoints)
    #  *************
    #  *           *
    # ***          *
    # * *          *
    #              *
    #              *
    #              *
    #            *****
    #            *****
    targetPositionsJoints_test = [-0.021860349543687967, -0.43821428636082427, 0.13770668564553845, -2.1723260837131075, 0.044060199449459714, 1.729482148355908, 0.780151212690881]

    #          *
    #         **
    #        * *
    #       *  *
    #      *   *
    #     *    *
    #    *     *
    #  ***   *****
    #  * *   *****
    targetPositionsJoints_test = [-0.41639349778717333, 0.8254077830686731, -0.07092072120488697, -2.1336947324364215, 1.0840709206509551, 1.4970466512368048, 0.9383130510987506]
    # targetPositionsJoints_test = [0.0,-0.08,0.0,-1.65,0.0,1.58,0.870]

    #           *
    # ***      **
    #   ******* *
    # ***       *
    #           *
    #           *
    #           *
    #         *****
    #         *****
    # targetPositionsJoints_test = [0.9997282949225943, -1.4047088639736132, -1.4841185072148564, -2.007658022560122, -0.9748820327350661, 2.8517021448082396, 0.3551867792836711]
    # targetPositionsJoints_test = [-0.08887629778144394, -0.43511143521676976, -0.09249673150817637, -2.8784299948985614, -0.2534064017368684, 3.7472803575481732, 0.9084821005872942]
    
    
    panda.move_to_target_joints(targetPositionsJoints_test)
    panda.fully_close_gripper()
    # sys.exit()
    # input("hit enter to move forward")
    
    for index in range(5):
        wpose = panda.moveit_group.get_current_pose().pose
        waypoints = []
        wpose.position.y += 0.1
        # wpose.position.z -= 0.06
        # wpose.position.y += 0.15
        waypoints.append(copy.deepcopy(wpose))    
        panda.move_through_waypoints(waypoints)

#    panda.move_y_straight_line_ccp()
#    panda.move_straight_line_jac()
    
    

    
    
    
    # for index in range(2):
    #     wpose = panda.moveit_group.get_current_pose().pose
    #     waypoints = []
    #     wpose.position.z -= 0.08
    #     waypoints.append(copy.deepcopy(wpose))    
    #     panda.move_through_waypoints(waypoints)
    # targetPositionsJoints_test = [-0.41429875365684077, 0.7531559819003992, -0.09432195875240348, -2.172390429781196, 1.080601253367148, 1.469608373509513, 0.9228163020693847]
    # panda.move_to_target_joints(targetPositionsJoints_test)
    # time.sleep(2)
    # wpose = panda.moveit_group.get_current_pose().pose
    # waypoints = []
    # wpose.position.y += 0.15
    # waypoints.append(copy.deepcopy(wpose))    
    # panda.move_through_waypoints(waypoints)
