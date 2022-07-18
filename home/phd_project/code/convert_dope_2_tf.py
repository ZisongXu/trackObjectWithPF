#!/usr/bin/env python3  
import rospy
from geometry_msgs.msg import PointStamped,PoseStamped
import tf


def convert_dope_2_tf(msg,object_frame_id):
    br = tf.TransformBroadcaster()
    br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                     (msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w),
                     rospy.Time.now(),
                     "cheezit",
                     "camera_color_optical_frame")
    rate.sleep()

if __name__ == '__main__':
    rospy.init_node('dope_2_tf')
    object_frame_id = "cheezit"
    rospy.Subscriber('/dope/pose_cracker',
                     PoseStamped, 
                     convert_dope_2_tf,
                     object_frame_id)
    rate = rospy.Rate(10)
    rospy.spin()
