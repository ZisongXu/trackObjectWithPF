#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import roslaunch
import time
import rospy
from sensor_msgs.msg import Joy
from key_mapper import KEYMAP
from panda_pump import Panda
import subprocess
import utils


def move_gripper(x=0.0, y=0.0, z=0.0):
    wpose = panda.end_effector_pose
    wpose.position.x += x
    wpose.position.y += y
    wpose.position.z += z
    panda.move_through_waypoints([wpose])
    
def move_home():
    home_config = [-2.2748631408944138, 0.974557520662193, 0.6354566556779961,
                   -2.3196569263832387, 1.2403121058379996, 2.060963524952071,
                   -2.2097670971912553]
    home_config = [-0.41639349778717333, 0.8254077830686731, -0.07092072120488697, -2.1336947324364215, 1.0840709206509551, 1.4970466512368048, 0.9383130510987506]
    
    # camera on the hand    
    home_config = [-0.11386161916169073, -0.3224095850701918, 0.7887302985776935, -1.685773666775809, 0.2324141370407867, 1.5458522827095453, 1.4731677771386262]
    # camera on the top
    home_config = [-0.5282810975828385, -0.4472975845964331, 0.44842371379955714, -1.8997312286483752, 0.22086196654372744, 1.6459452103244183, 0.5922973285106663]
    # high
    home_config = [0.01564718111305425, -0.14825592407882143, -0.0001504720379582373, -1.7706743231154325, 0.04046481457683775, 1.6776742815516352, 0.7991213814756937]
    # low
    home_config = [-0.0500577302505128, 0.032328707273344436, 0.04637714152020083, -2.0581338763117354, -0.02282543493807316, 2.103516532447603, 0.9540484014538868]
    
    panda.move_to_configuration(home_config)

def launch_particle_filtering():
    global particle_filter_running

    if particle_filter_running: # Killing previous process isn't finished yet; wait.
        print('Particle filtering already running')
        return
    
    particle_filter_running = True
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    path = os.path.expanduser('~/catkin_ws/src/PBPF')
    launch = roslaunch.parent.ROSLaunchParent(uuid, [f'{path}/launch/code_visuial.launch'])
    launch.start()
    time.sleep(3)
    manage_windows()

def manage_windows():
    commands = [
        'wmctrl -i -r $(wmctrl -l | grep "/dope/rgb_points_my_wireframe" | awk "{print $1}") -e "0,0,0,960,1080"',
        'wmctrl -i -r $(wmctrl -l | grep "Bullet" | sed -n 1p | awk "{print $1}") -e "0,1400,0,600,500"',
        'wmctrl -i -r $(wmctrl -l | grep "Bullet" | sed -n 2p | awk "{print $1}") -e "0,1400,510,600,500"',
        'wmctrl -i -a $(wmctrl -l | grep "/dope/rgb_points_my_wireframe" | awk "{print $1}")',
        'wmctrl -i -r $(wmctrl -l | grep "Bullet" | sed -n 3p | awk "{print $1}") -e "0,0,0,400,300"',
        'wmctrl -i -a $(wmctrl -l | grep "Bullet" | sed -n 3p | awk "{print $1}")',
    ]

    for command in commands:
        subprocess.Popen(command, shell=True)

def kill_particle_filtering():
    global particle_filter_running
    if particle_filter_running:
        print('Killing particle filter')
        os.system('pkill -9 Visualisation')
        os.system('pkill -9 Physics_Based_')
        os.system('pkill -9 image_view')
        particle_filter_running = False

def cp_multi_2_one():
    global particle_filter_running
    if not particle_filter_running:
        print('cp yaml file one')
        os.system('cp ~/catkin_ws/src/PBPF/config/cracker_only.yaml ~/catkin_ws/src/PBPF/config/parameter_info.yaml')
        
def cp_one_2_multi():
    global particle_filter_running
    if not particle_filter_running:
        print('cp yaml file multi')
        os.system('cp ~/catkin_ws/src/PBPF/config/cracker_and_soup.yaml ~/catkin_ws/src/PBPF/config/parameter_info.yaml')


def joystick_callback(data):
    global start_particle_filter, particle_filter_running

    if data.buttons[KEYMAP['start']]:
        print("Should run the alg, but have not done!")
        # if not particle_filter_running:
        #     start_particle_filter = True
    elif data.buttons[KEYMAP['select']]:
        print("Should stop the alg, but have not done!")
        # kill_particle_filtering()
    # elif data.buttons[KEYMAP['x']]:
    #     panda.fully_close_gripper()
    # elif data.buttons[KEYMAP['o']]:
    #     panda.fully_open_gripper()
    elif data.buttons[KEYMAP['up-arrow']]:
        move_gripper(x=0.02)
    elif data.buttons[KEYMAP['down-arrow']]:
        move_gripper(x=-0.02)
    elif data.buttons[KEYMAP['left-arrow']]:
        move_gripper(y=0.02)
    elif data.buttons[KEYMAP['right-arrow']]:
        move_gripper(y=-0.02)
    elif data.buttons[KEYMAP['l1']]:
        move_gripper(z=0.01)
    elif data.buttons[KEYMAP['r1']]:
        move_gripper(z=-0.01)
    elif data.buttons[KEYMAP['l2']]:
        # cp_multi_2_one()
        print("Have not done!")
    elif data.buttons[KEYMAP['r2']]:
        # cp_one_2_multi()
        print("Have not done!")
    elif data.buttons[KEYMAP['pskey']]:
        print("Ready to go to initial pose...")
        move_home()
        

if __name__ == '__main__':
    rospy.init_node('panda-ps3-controller')
    particle_filter_running = False
    start_particle_filter = False

    # cp_one_2_multi()

    panda = Panda()
    rospy.Subscriber('joy', Joy, joystick_callback, queue_size=1)
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if start_particle_filter:
            launch_particle_filtering()
            start_particle_filter = False
        rate.sleep()
