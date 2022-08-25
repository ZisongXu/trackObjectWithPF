# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""


import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
import math
import random
import copy
import os

'''
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
p.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])
planeId = p.loadURDF("plane.urdf")
'''


#visualisation_model
p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER)

p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
p_visualisation.setGravity(0,0,-9.81)
p_visualisation.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])

#load and set real robot
plane_id = p_visualisation.loadURDF("plane.urdf")

cylinder_real_robot_start_pos = [0, 0, 0.3]
cylinder_real_robot_start_orientation = p_visualisation.getQuaternionFromEuler([0,0,0])
real_robot_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_real_robot.urdf"),
                                         cylinder_real_robot_start_pos,
                                         cylinder_real_robot_start_orientation)


#load and set object
cylinder_real_object_start_pos = [1.5, 0, 0.2]
cylinder_real_object_start_orientation = p_visualisation.getQuaternionFromEuler([0,0,0])
real_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_object.urdf"),
                                          cylinder_real_object_start_pos,
                                          cylinder_real_object_start_orientation)



class Particle(object):
    def __init__(self,x=0.0,y=0.0,z=0.0,w=1.0,index = 0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.index = index
    def as_pose(self):
        return True


class InitialSimulationModel():
    def __init__(self,particle_num,cylinder_real_object_start_pos):
        self.particle_num = particle_num
        self.cylinder_real_object_start_pos = cylinder_real_object_start_pos
        self.particle_cloud = []
        self.pybullet_particle_env_collection = []
        self.fake_robot_id_collection = []
        self.cylinder_particle_no_visual_id_collection = []
        self.cylinder_particle_with_visual_id_collection =[]
    def initial_particle(self):
        for i in range(self.particle_num):
            x = random.uniform(self.cylinder_real_object_start_pos[0] - 0.5, self.cylinder_real_object_start_pos[0] + 0.5)
            y = random.uniform(self.cylinder_real_object_start_pos[1] - 0.5, self.cylinder_real_object_start_pos[1] + 0.5)
            z = self.cylinder_real_object_start_pos[2]
            w = 1/self.particle_num
            particle = Particle(x,y,z,w,index=i)
            self.particle_cloud.append(particle)
        object_estimate_pose_x,object_estimate_pose_y = self.compute_estimate_pose_of_object(self.particle_cloud)
        print("initial_object_estimate_pose:",object_estimate_pose_x,object_estimate_pose_y)
    def compute_estimate_pose_of_object(self, particle_cloud):
        x_set = 0
        y_set = 0
        w_set = 0
        
        for index,particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            w_set = w_set + particle.w
        return x_set/w_set,y_set/w_set
        
    def display_particle(self):
        for index, particle in enumerate(self.particle_cloud):
            cylinder_visualize_particle_pos = [particle.x, particle.y, 0.2]
            cylinder_visualize_particle_orientation = p_visualisation.getQuaternionFromEuler([0,0,0])
            cylinder_visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_particle_with_visual.urdf"),
                                                                      cylinder_visualize_particle_pos,
                                                                      cylinder_visualize_particle_orientation)
            self.cylinder_particle_with_visual_id_collection.append(cylinder_visualize_particle_Id)
    def initial_and_set_simulation_env(self):
        for index, particle in enumerate(self.particle_cloud):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT)
            self.pybullet_particle_env_collection.append(pybullet_simulation_env)
            
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            cylinder_fake_robot_start_pos = [0, 0, 0.3]
            cylinder_fake_robot_start_orientation = pybullet_simulation_env.getQuaternionFromEuler([0,0,0])
            fake_robot_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_fake_robot.urdf"),
                                                             cylinder_fake_robot_start_pos,
                                                             cylinder_fake_robot_start_orientation)
            self.fake_robot_id_collection.append(fake_robot_id)
            
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            
            cylinder_particle_no_visual_start_pos = [particle.x, particle.y, 0.2]
            cylinder_particle_no_visual_start_orientation = pybullet_simulation_env.getQuaternionFromEuler([0,0,0])
            cylinder_particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_particle_no_visual.urdf"),
                                                                              cylinder_particle_no_visual_start_pos,
                                                                              cylinder_particle_no_visual_start_orientation)
            self.cylinder_particle_no_visual_id_collection.append(cylinder_particle_no_visual_id)

particle_cloud = []
particle_num = 100
initial_parameter = InitialSimulationModel(particle_num,cylinder_real_object_start_pos)
initial_parameter.initial_particle() #only position of particle
initial_parameter.display_particle()
initial_parameter.initial_and_set_simulation_env()
#initial_parameter.particle_cloud #parameter of particle
#initial_parameter.pybullet_particle_env_collection #env of simulation
#initial_parameter.fake_robot_id_collection #id of robot in simulation
#initial_parameter.cylinder_particle_no_visual_id_collection #id of particle in simulation
#print(initial_parameter.pybullet_particle_env_collection)

class PFMove():
    
    def __init__(self,robot_id=None,real_robot_id=None,object_id=None):
        # init internals
        self.executed_control = []
        self.action_u = [[0.5, 0, 0]]*300
        self.particle_cloud = copy.deepcopy(initial_parameter.particle_cloud)
        self.particle_no_visual_id_collection = copy.deepcopy(initial_parameter.cylinder_particle_no_visual_id_collection)
        self.pybullet_env_id_collection = copy.deepcopy(initial_parameter.pybullet_particle_env_collection)
        self.particle_with_visual_id_collection = copy.deepcopy(initial_parameter.cylinder_particle_with_visual_id_collection)
        
        self.step_size = 0.1
        
        self.judgement_flag = False
        self.d_thresh_limitation = 0.2
        
        self.u_flag = 0
        
        self.sigma = 0.1
        
    #new structure
    def real_robot_control(self):
        self.executed_control = []
        real_robot_start_pos = self.get_real_robot_pos(real_robot_id)
        real_object_last_update_pos = self.get_real_object_pos(real_object_id)
        for u_i in self.action_u:
            self.real_robot_move(real_robot_id, u_i, self.step_size)
            self.executed_control.append(u_i)
            real_object_current_pos = self.get_real_object_pos(real_object_id)
            
            real_robot_speed = p_visualisation.getBaseVelocity(real_robot_id)
            #print("speed:",real_robot_speed[0])
            
            distance_between_current_and_old = self.compute_distance(real_object_current_pos,real_object_last_update_pos)#Cheat
            if distance_between_current_and_old > self.d_thresh_limitation:
                print(1)
                real_robot_curr_pos = self.get_real_robot_pos(real_robot_id)
                print("real robot pos")
                print(real_robot_curr_pos)
                
                #Cheat
                observation = self.get_observation(real_object_id) #get pos of real object
                
                #self.update_particle_filter(self.executed_control, robot_start_pos, observation)
                Flag = self.update_particle_filter_cheat(self.pybullet_env_id_collection, # simulation environment per particle
                                                         initial_parameter.fake_robot_id_collection, # fake robot id per sim_env
                                                         self.executed_control, # execution actions of the fake robot
                                                         real_robot_start_pos,
                                                         observation)
                if Flag is False:
                    return False
                real_object_last_update_pos = real_object_current_pos
                
                self.executed_control = []
                real_robot_start_pos = self.get_real_robot_pos(real_robot_id)
                
    def get_real_robot_pos(self, real_robot_id):
        real_robot_info = p_visualisation.getBasePositionAndOrientation(real_robot_id)
        return real_robot_info[0]
    
    def get_real_object_pos(self, object_id):
        object_info = p_visualisation.getBasePositionAndOrientation(object_id)
        return object_info[0]
    
    def get_observation(self, object_id):
        object_info = self.get_real_object_pos(object_id)
        return object_info
    
    def real_robot_move(self, real_robot_id, u_i, step_size):
        for i in range(int(step_size*240)):
            p_visualisation.resetBaseVelocity(real_robot_id,linearVelocity = u_i)
            p_visualisation.stepSimulation()
            time.sleep(1.0/240)
        
    def compute_distance(self,object_current_pos,object_last_update_pos):
        x_distance = object_current_pos[0] - object_last_update_pos[0]
        y_distance = object_current_pos[1] - object_last_update_pos[1]
        distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
        return distance
     
    def update_particle_filter_cheat(self, pybullet_sim_env, fake_robot_id, executed_control, robot_start_pos, observation):
        self.motion_update(pybullet_sim_env, fake_robot_id, executed_control, robot_start_pos)
        Flag = self.observation_update(observation)
        if Flag is False:
            return False
        print("display particle")
        self.display_particle_in_visual_model(self.particle_cloud)
        # print debug info of all particles here
        #input('hit enter to continue')
        return
    
    def motion_update(self, pybullet_sim_env, fake_robot_id, executed_control, robot_start_pos):
        #for index,particle in enumerate(self.particle_cloud):
            #set_robot_pos = robot_start_pos
        #print("fake robot pos")
        for index, pybullet_env in enumerate(pybullet_sim_env):
            for u_i in executed_control:
                #print(pybullet_env.getBasePositionAndOrientation(fake_robot_id[index]))
                for i in range(int(self.step_size*240)):
                    pybullet_env.resetBaseVelocity(fake_robot_id[index], linearVelocity = u_i)
                    pybullet_env.stepSimulation()
                    #time.sleep(1.0/240)
            sim_particle_old_pos = [self.particle_cloud[index].x,
                                    self.particle_cloud[index].y,
                                    self.particle_cloud[index].z]
            
            sim_particle_cur_pos = self.get_item_pos(pybullet_env,
                                                     initial_parameter.cylinder_particle_no_visual_id_collection[index])
            
            normal_x = self.add_noise(sim_particle_cur_pos[0],sim_particle_old_pos[0])
            normal_y = self.add_noise(sim_particle_cur_pos[1],sim_particle_old_pos[1])
            #print("particle_x_before:",sim_particle_cur_pos[0]," ","particle_y_before:",sim_particle_cur_pos[1])
            self.particle_cloud[index].x = normal_x
            self.particle_cloud[index].y = normal_y
            #print("particle_x__after:",self.particle_cloud[index].x," ","particle_y__after:",self.particle_cloud[index].y)
            
            #print(pybullet_env.getBasePositionAndOrientation(fake_robot_id[index])[0])
            #execute the control in executed_control
            #add noise on particle filter

        
    def observation_update(self, observation):
        pos_of_real_object = observation #pos of real object [1,2,3]
        for index,particle in enumerate(self.particle_cloud):
            
            particle_x = particle.x
            particle_y = particle.y
            
            real_object_pos = pos_of_real_object
            
            real_object_pos_x = real_object_pos[0]
            real_object_pos_y = real_object_pos[1]
            
            distance = math.sqrt((particle_x - real_object_pos_x) ** 2 + (particle_y - real_object_pos_y) ** 2)
            
            x = distance
            mean = 0
            sigma = self.sigma
            #weight = self.normal_distribution(x, mean, sigma) * sigma
            weight = self.normal_distribution(x, mean, sigma)
            
            particle.w = weight
            
        Flag = self.normalize_particles()
        if Flag is False:
            return False
        self.resample_particles()
        self.set_paticle_in_each_sim_env()
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
            part_pos = pybullet_env.getBasePositionAndOrientation(self.particle_no_visual_id_collection[index])
            #print("particle:",part_pos[0][0],part_pos[0][1],part_pos[0][2])
        object_estimate_pose_x,object_estimate_pose_y = self.compute_estimate_pose_of_object(self.particle_cloud)
        print("object_estimate_pose:",object_estimate_pose_x,object_estimate_pose_y)
        print("object_real_____pose:",pos_of_real_object[0],pos_of_real_object[1])
            
        return
    
    def get_item_pos(self,pybullet_env,item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0]
    
    def add_noise(self,current_pos,old_pos):
        distance = math.fabs(current_pos - old_pos)
        mean = current_pos
        sigma = self.sigma
        #sigma = self.sigma * 2
        #print ("sigma:",sigma)
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
    
    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)
    
    def normalize_particles(self):
        tot_weight = sum([particle.w for particle in self.particle_cloud])
        if tot_weight == 0:
            print("Error!,total weight is 0")
            return False
        for particle in self.particle_cloud:
            particle_w = particle.w/tot_weight
            particle.w = particle_w
            
        #tot_weight_test = sum([particle.w for particle in self.particle_cloud])
        
        #print("tot_weight_test:",tot_weight_test)
            
    def resample_particles(self):
        particles_w = []
        newParticles = [] 
        n_particle = len(self.particle_cloud)
        for particle in self.particle_cloud:
            particles_w.append(particle.w)
        
        particle_array= np.random.choice(a = n_particle, size = n_particle, replace=True, p= particles_w)
        particle_array_list = list(particle_array)
        for index,i in enumerate(particle_array_list):
            particle = Particle(self.particle_cloud[i].x,self.particle_cloud[i].y,self.particle_cloud[i].z,self.particle_cloud[i].w,index)
            newParticles.append(particle)
            
        self.particle_cloud = copy.deepcopy(newParticles)
        
    def set_paticle_in_each_sim_env(self):
        
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
            visual_particle_pos = [self.particle_cloud[index].x, self.particle_cloud[index].y, 0.2]
            visual_particle_orientation = p.getQuaternionFromEuler([0,0,0])
            
            pybullet_env.resetBasePositionAndOrientation(self.particle_no_visual_id_collection[index],
                                                         visual_particle_pos,
                                                         visual_particle_orientation)
        
        return
        
        
    def display_particle_in_visual_model(self, particle_cloud):
        for index, particle in enumerate(particle_cloud):
            visual_particle_pos = [particle.x, particle.y, 0.2]
            visual_particle_orientation = p.getQuaternionFromEuler([0,0,0])
            
            p_visualisation.resetBasePositionAndOrientation(self.particle_with_visual_id_collection[index],
                                                            visual_particle_pos,
                                                            visual_particle_orientation)
            
            #particle_pos = self.get_item_pos(pybullet_env[index],initial_parameter.cylinder_particle_no_visual_id_collection[index])

    def compute_estimate_pose_of_object(self, particle_cloud):
        x_set = 0
        y_set = 0
        w_set = 0
        
        for index,particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            w_set = w_set + particle.w
        return x_set/w_set,y_set/w_set
    
    
    
    
robot1 = PFMove()
time.sleep(2)
while 1:
    Flag = robot1.real_robot_control()
    if Flag is False:
        break
    #if robot1.initial == True:
        #print("Begin to use observation model")
        #robot1.observation_model_cheating()

p.disconnect()

