#!/usr/bin/python3

from multiple_controllers_demo import Panda

class PandaTorque(Panda):
    def publish_joint_torque(self, joint_torque_trajectory):
        self.switch_to('effort_joint_trajectory_controller')
        self.torques_publisher.publish(joint_torque_trajectory)


def move_arm_with_custom_joint_torques_trajectory_demo():
    time_diff = 2.0

    # time_from_start = Desired time from the trajectory start to arrive at this trajectory point.
    trajectory_points = [
        JointTrajectoryPoint(effort=[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start=rospy.Duration(time_diff)),
        JointTrajectoryPoint(effort=[0.0, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0], time_from_start=rospy.Duration(2 * time_diff)),
        JointTrajectoryPoint(effort=[0.0, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0], time_from_start=rospy.Duration(3 * time_diff)),
        JointTrajectoryPoint(effort=[0.0, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0], time_from_start=rospy.Duration(4 * time_diff)),
    ]

    trajectory = JointTrajectory(joint_names=panda.joint_names, points=trajectory_points)
    panda.publish_joint_torque(trajectory)
    time.sleep(time_diff * len(trajectory_points))


if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_demo')
    panda = PandaTorque()

    input('Press [ENTER] to start')

    print(' *****WARNING****** THIS SCRIPT SHOULD ONLY BE USED IF YOU HAVE HAD A DISCUSSION WITH MEHMET AND RAFAEL FIRST.')
    print(' Please confirm below that you have spoken to Mehmet and Rafael before running this script..')

    correct_prompt = "I discussed with Mehmet and Rafael about the torque control demo"
    while True:
        prompt = input(f'Please type {correct_prompt!r}:')
        if prompt == correct_prompt:
            break
        print('Wrong prompt, please type the correct prompt to continue')

    panda.fully_open_gripper()
    move_arm_with_custom_joint_torques_trajectory_demo()
    panda.fully_close_gripper()
