import sys
import time
import rospy
from sensor_msgs.msg import JointState
import geometry_msgs.msg
import actionlib
from franka_gripper.msg import MoveAction, MoveGoal
import moveit_commander
import moveit_msgs.msg


class Panda:
    def __init__(self):
        self.gripper_speed = 0.1
        self.subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)
        self.gripper_move_client = actionlib.SimpleActionClient("franka_gripper/move", MoveAction)
        self.gripper_move_client.wait_for_server(timeout=rospy.Duration(secs=5))
        self.moveit_robot_commander = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        self.moveit_group = moveit_commander.MoveGroupCommander('panda_arm')
        time.sleep(1)

    def move_to(self, x, y, z):
        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.w = 1.0
        pose_target.position.x = x
        pose_target.position.y = y
        pose_target.position.z = z
        self.moveit_group.set_pose_target(pose_target)
        self.moveit_group.go(wait=True)
        self.moveit_group.clear_pose_targets()

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
    panda.move_to(x=-0.2, y=0.0, z=1.9)
    panda.fully_close_gripper()
