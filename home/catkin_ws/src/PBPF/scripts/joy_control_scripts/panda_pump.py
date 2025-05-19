#!/usr/bin/python3

import rospy
import time
import rospy
import geometry_msgs.msg
import actionlib
# from franka_gripper.msg import GraspAction
# from franka_gripper.msg import MoveAction, MoveGoal
import moveit_commander
import moveit_msgs.msg

class Panda:
    def __init__(self):
        self.setup_moveit()
        # self.setup_gripper_actionlibs()
        self.set_end_effector_link()
        self.add_static_obstacles_to_scene()

    @property
    def end_effector_pose(self):
        return self.moveit_group.get_current_pose().pose
    
    def move_to_configuration(self, config):
        self.moveit_group.set_joint_value_target(config)
        self.moveit_group.go(wait=True)
        self.moveit_group.clear_pose_targets()
        
    def setup_moveit(self):
        self.robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander('panda_arm')
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

    # def setup_gripper_actionlibs(self):
    #     self.gripper_speed = 0.1
    #     self.gripper_grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
    #     self.gripper_grasp_client.wait_for_server()

    #     self.gripper_move_client = actionlib.SimpleActionClient("franka_gripper/move", MoveAction)
    #     self.gripper_move_client.wait_for_server()

    def set_end_effector_link(self):
        self.ee_link = 'panda_pump_tcp'
        self.moveit_group.set_end_effector_link("panda_pump_tcp")
        self.moveit_group.set_pose_reference_frame("panda_link0")

    def add_static_obstacles_to_scene(self):
        # Add table to scene to avoid Panda smashing the table.
        table_size = (2.0, 2.0, 0.04) 
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

    def move_through_waypoints(self, end_effector_waypoints):
        plan, _ = self.moveit_group.compute_cartesian_path(end_effector_waypoints, 0.01, 0.0)
        trajectory = plan.joint_trajectory.points
        self.moveit_group.go(trajectory[-1].positions, wait=True)
    
    # def set_gripper_openning(self, value):
    #     move_goal = MoveGoal(width=value, speed=self.gripper_speed)
    #     self.gripper_move_client.send_goal(move_goal)

    # def fully_open_gripper(self):
    #     self.set_gripper_openning(0.08)

    # def fully_close_gripper(self):
    #     self.set_gripper_openning(0.0)

