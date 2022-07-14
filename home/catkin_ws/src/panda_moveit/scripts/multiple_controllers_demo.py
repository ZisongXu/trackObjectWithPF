#!/usr/bin/python3

import sys
import copy
import time
import rospy
from sensor_msgs.msg import JointState
import geometry_msgs.msg
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal
import moveit_commander
import moveit_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import LoadController, LoadControllerRequest
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from controller_manager_msgs.srv import ListControllers, ListControllersRequest


class PandaMoveIt:
    def __init__(self):
        self.commander = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander('panda_arm')
        time.sleep(1)

    @property
    def end_effector_cartesian_pose(self):
        return self.group.get_current_pose().pose

    def move_to_end_effector_pose(self, end_effector_pose):
        self.group.set_pose_target(end_effector_pose)
        self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

    def move_to_joint_angles(self, joint_angles=None):
        self.group.go(joint_angles, wait=True)
        self.group.stop()

    def move_through_end_effector_waypoints(self, end_effector_waypoints):
        (plan, fraction) = self.group.compute_cartesian_path(end_effector_waypoints, 0.01, 0.0)
        self.group.execute(plan, wait=True)
        self.group.stop()


class Panda:
    def __init__(self):
        self.gripper_speed = 0.1
        self.subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)
        self.gripper_move_client = actionlib.SimpleActionClient("franka_gripper/move", MoveAction)
        self.gripper_move_client.wait_for_server(timeout=rospy.Duration(secs=5))
        self.moveit = PandaMoveIt()
        self.load_controller = rospy.ServiceProxy('controller_manager/load_controller', LoadController)
        self.load_controller.wait_for_service()

        self.switch_controller = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)
        self.switch_controller.wait_for_service()
        self.active_controller = None

        self.list_controllers = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
        self.list_controllers.wait_for_service()

        self.positions_publisher = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        self.torques_publisher = rospy.Publisher('/effort_joint_trajectory_controller/command', JointTrajectory, queue_size=1)

    def joint_values_callback(self, msg):
        self.current_joint_values = msg.position

    @property
    def joint_names(self):
        return [f'panda_joint{i}' for i in range(1, 8)]

    def set_gripper_openning(self, value):
        move_goal = MoveGoal(width=value, speed=self.gripper_speed)
        self.gripper_move_client.send_goal(move_goal)

    def fully_open_gripper(self):
        self.set_gripper_openning(0.08)

    def fully_close_gripper(self):
        self.set_gripper_openning(0.0)

    @property
    def loaded_controllers(self):
        params = ListControllersRequest()
        result = self.list_controllers(params)
        loaded_controllers = [controller.name for controller in result.controller]
        return loaded_controllers

    def switch_to(self, controller_name):
        if controller_name not in self.loaded_controllers:
            params = LoadControllerRequest()
            params.name = controller_name
            if not self.load_controller(params):
                rospy.logerr(f'Couldn\'t load controller {controller_name!r}.')
                sys.exit(1)
            rospy.loginfo(f'Controller {controller_name!r} loaded.')

        params = SwitchControllerRequest()
        params.start_controllers = [controller_name]
        if self.active_controller:
            params.stop_controllers = [self.active_controller]
        else:
            params.stop_controllers = ['effort_joint_trajectory_controller']
        params.strictness = params.STRICT
        if not self.switch_controller(params):
            rospy.logerr('Couldn\'t switch from {self.active_controller!} to {controller_name!r}.')
            sys.exit(1)
        
        rospy.loginfo(f'Controller {controller_name!r} started and is now active.')
        self.active_controller = controller_name

    def publish_joint_position(self, joint_position_trajectory):
        self.switch_to('position_joint_trajectory_controller')
        self.positions_publisher.publish(joint_position_trajectory)


def moveit_end_effector_pose_demo():
    pose = geometry_msgs.msg.Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 2.0
    pose.orientation.w = 1.0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0

    print('Moving arm to desired end-effector cartesian pose')
    panda.moveit.move_to_end_effector_pose(pose)

def moveit_using_joint_angles_demo():
    new_joint_values = [0.0, -0.08, 0.0, -1.65, 0.0, 1.58, 0.870]
    print('Moving arm to desired joint angles')
    panda.moveit.move_to_joint_angles(new_joint_values)

def moveit_through_end_effector_waypoints_demo():
    waypoints = []
    scale = 2

    wpose = panda.moveit.end_effector_cartesian_pose
    wpose.position.z -= scale * 0.1  # First move up (z)
    wpose.position.y += scale * 0.2  # and sideways (y)
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.y -= scale * 0.1  # Third move sideways (y)
    waypoints.append(copy.deepcopy(wpose))

    print('Moving arm through end-effector cartesian waypoints.')
    panda.moveit.move_through_end_effector_waypoints(waypoints)

def move_arm_with_custom_joint_positions_trajectory_demo():
    time_diff = 2.0

    # time_from_start = Desired time from the trajectory start to arrive at this trajectory point.
    trajectory_points = [
        JointTrajectoryPoint(positions=[0.0, -0.1, 0.0, -1.65, 0.0, 1.58, 0.870], time_from_start=rospy.Duration(time_diff)),
        JointTrajectoryPoint(positions=[0.0, -0.1, 0.1, -1.65, 0.0, 1.58, 0.870], time_from_start=rospy.Duration(2 * time_diff)),
        JointTrajectoryPoint(positions=[0.1, -0.1, 0.1, -1.65, 0.0, 1.58, 0.870], time_from_start=rospy.Duration(3 * time_diff)),
        JointTrajectoryPoint(positions=[0.1, -0.1, 0.1, -1.65, 0.3, 1.58, 0.870], time_from_start=rospy.Duration(4 * time_diff)),
    ]

    trajectory = JointTrajectory(joint_names=panda.joint_names, points=trajectory_points)
    panda.publish_joint_position(trajectory)
    time.sleep(time_diff * len(trajectory_points))


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_demo')
    panda = Panda()
    panda.switch_to('position_joint_trajectory_controller')

    input('Press [ENTER] to start')

    panda.fully_open_gripper()

    moveit_end_effector_pose_demo()
    moveit_through_end_effector_waypoints_demo()
    moveit_using_joint_angles_demo()
    move_arm_with_custom_joint_positions_trajectory_demo()
    move_arm_with_custom_joint_torques_trajectory_demo()

    panda.fully_close_gripper()
