import os.path
import os

#Class of initialize the real world model
class InitialRealworldModel():
    def __init__(self,joint_pos=0, object_cracker_flag=False, object_soup_flag=False, p_visualisation=0):
        self.flag = 0
        self.joint_pos = joint_pos
        self.object_cracker_flag = object_cracker_flag
        self.object_soup_flag = object_soup_flag
        self.p_visualisation = p_visualisation
        
    def initial_robot(self,robot_pos,robot_orientation = [0,0,0,1]):
        #robot_orientation = p_visualisation.getQuaternionFromEuler(robot_euler)
        real_robot_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                      robot_pos,
                                                      robot_orientation,
                                                      useFixedBase=1)
        self.set_real_robot_JointPosition(self.p_visualisation,real_robot_id,self.joint_pos)
        for i in range(240):
            self.p_visualisation.stepSimulation()
            #time.sleep(1./240.)
        return real_robot_id
    
    def initial_target_object(self,object_pos,object_orientation = [0,0,0,1]):
        #object_orientation = p_visualisation.getQuaternionFromEuler(object_euler)
        if self.object_cracker_flag == True:
            real_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/cube/cheezit_obj_small_hor.urdf"),
                                                           object_pos,
                                                           object_orientation)
        if self.object_soup_flag == True:
            real_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_obj_small_hor.urdf"),
                                                           object_pos,
                                                           object_orientation)
        self.p_visualisation.changeDynamics(real_object_id,-1,mass=0.380,lateralFriction = 0.5)
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
