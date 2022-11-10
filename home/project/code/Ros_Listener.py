import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3

#Class of franka robot listen to info from ROS
class Ros_Listener():
    def __init__(self, optitrack_working_flag, object_flag):
        self.optitrack_working_flag = optitrack_working_flag
        self.object_flag = object_flag
        rospy.Subscriber('/joint_states', JointState, self.joint_values_callback, queue_size=1)
        self.joint_subscriber = JointState()
        if self.optitrack_working_flag == True:
            rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose', PoseStamped, self.robot_pose_callback, queue_size=1)
            self.robot_pose = PoseStamped()
            rospy.Subscriber('/mocap/rigid_bodies/cheezit/pose', PoseStamped, self.object_pose_callback_cracker, queue_size=1)
            self.object_cracker_pose = PoseStamped()
            rospy.Subscriber('/mocap/rigid_bodies/zisongsoup/pose', PoseStamped, self.object_pose_callback_soup, queue_size=1)
            self.object_soup_pose = PoseStamped()
            rospy.Subscriber('/mocap/rigid_bodies/baseofcheezit/pose', PoseStamped, self.base_of_cheezit_callback, queue_size=1)
            self.base_pose = PoseStamped()
        elif self.optitrack_working_flag == False:
            rospy.Subscriber('/Opti_pose', PoseStamped, self.fake_optipose_callback, queue_size=10)
            self.fake_opti_pose = PoseStamped()
        rospy.spin
    
    
    def listen_2_object_pose(self, object_flag):
        if object_flag == "cracker":
            return self.object_cracker_pose
        elif object_flag == "soup":
            return self.object_soup_pose
        elif object_flag == "base":
            return self.base_pose
    
    def listen_2_robot_pose(self):
        return self.robot_pose
    
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
        self.robot_pose = [self.robot_pos, self.robot_ori]
        
    def object_pose_callback_cracker(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_cracker_pose = [self.object_pos, self.object_ori]
        
    def object_pose_callback_soup(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_soup_pose = [self.object_pos, self.object_ori]
        
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
        self.base_pose = [self.base_pos, self.base_ori]
        
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
        self.fake_opti_pose = [self.fake_opti_pos, self.fake_opti_ori]