import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point,PointStamped,PoseStamped,Quaternion,TransformStamped, Vector3

#Class of franka robot listen to info from ROS
class Ros_Listener():
    def __init__(self, optitrack_working_flag, object_flag):
        self.optitrack_working_flag = optitrack_working_flag
        self.object_flag = object_flag
        self.joint_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback, queue_size=1)
        if self.optitrack_working_flag == True:
            self.robot_pose = rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose', PoseStamped, self.robot_pose_callback, queue_size=10)
            if self.object_flag == "cracker":
                self.object_pose = rospy.Subscriber('/mocap/rigid_bodies/cheezit/pose', PoseStamped, self.object_pose_callback, queue_size=10)
            if self.object_flag == "soup":
                self.object_pose = rospy.Subscriber('/mocap/rigid_bodies/zisongsoup/pose', PoseStamped, self.object_pose_callback, queue_size=10)
            self.base_pose = rospy.Subscriber('/mocap/rigid_bodies/baseofcheezit/pose', PoseStamped, self.base_of_cheezit_callback, queue_size=10)
        elif self.optitrack_working_flag == False:
            self.fake_opti_pose = rospy.Subscriber('/Opti_pose', PoseStamped, self.fake_optipose_callback, queue_size=10)
        rospy.spin
        
    def joint_values_callback(self, msg):
        self.current_joint_values = list(msg.position)
        
    def robot_pose_callback(self, data):
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
        
    def base_of_cheezit_callback(self,data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.base_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.base_ori = [x_ori, y_ori, z_ori, w_ori]
        
    def fake_optipose_callback(self,data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.fake_opti_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.fake_opti_ori = [x_ori, y_ori, z_ori, w_ori]