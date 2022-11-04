import os.path
import os

#Class of initialize the real world model
class InitialRealworldModel():
    def __init__(self, object_num=0, joint_pos=0, object_flag=False, p_visualisation=0):
        self.object_num = object_num
        self.flag = 0
        self.joint_pos = joint_pos
        self.object_flag = object_flag
        self.p_visualisation = p_visualisation
        
    def initial_robot(self, robot_pos, robot_orientation = [0,0,0,1]):
        # robot_orientation = p_visualisation.getQuaternionFromEuler(robot_euler)
        if self.joint_pos == 0:
            pw_T_rob_opti_pos = [0.4472889147344443, -0.08, 0.0821006075425945]
            pw_T_rob_opti_ori = [0,1,0,1]
            # pw_T_rob_opti_ori = [0.52338279, 0.47884367, 0.52129429, -0.47437481]
            real_robot_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_cheat_robot.urdf"),
                                                          pw_T_rob_opti_pos,
                                                          pw_T_rob_opti_ori)
            for i in range(240):
                self.p_visualisation.stepSimulation()
        else:
            real_robot_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                          robot_pos,
                                                          robot_orientation,
                                                          useFixedBase=1)
            self.set_real_robot_JointPosition(self.p_visualisation,real_robot_id,self.joint_pos)
            for i in range(240):
                self.p_visualisation.stepSimulation()
        return real_robot_id
    
    def initial_target_object(self, object_pos, object_orientation = [0,0,0,1]):
        # object_orientation = p_visualisation.getQuaternionFromEuler(object_euler)
        if self.object_flag == "cracker":
            real_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_contact_test_obj_hor.urdf"),
                                                           object_pos,
                                                           object_orientation)
        if self.object_flag == "soup":
            real_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/soup_contact_test_obj_hor.urdf"),
                                                           object_pos,
                                                           object_orientation)
        self.p_visualisation.changeDynamics(real_object_id,-1,mass=0.380,lateralFriction = 0.5)
        return real_object_id
    
    def initial_contact_object(self, objects_pose_list):
        # object_orientation = p_visualisation.getQuaternionFromEuler(object_euler)
        for i in range(self.object_num):
            object_name = objects_pose_list[i].opti_obj_name
            object_pos = objects_pose_list[i].pos
            object_ori = objects_pose_list[i].ori
            real_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+object_name+"/"+object_name+"_contact_test_obj_hor.urdf"),
                                                           object_pos,
                                                           object_ori)
            self.p_visualisation.changeDynamics(real_object_id,-1,mass=0.380,lateralFriction = 0.5)
            
            
#        if self.object_flag == "cracker":
#            real_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_contact_test_obj_hor.urdf"),
#                                                           object_pos,
#                                                           object_orientation)
#        if self.object_flag == "soup":
#            real_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/soup_contact_test_obj_hor.urdf"),
#                                                           object_pos,
#                                                           object_orientation)
#        self.p_visualisation.changeDynamics(real_object_id,-1,mass=0.380,lateralFriction = 0.5)
        return real_object_id
    
    def set_real_robot_JointPosition(self,pybullet_simulation_env,robot, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index+2,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])
            else:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])
