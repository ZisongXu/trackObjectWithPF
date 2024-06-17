import multiprocessing
import time
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
import os
import math
import random
import pprint

should_print = True
robot_instance = None

def init_worker():
    global robot_instance
    robot_instance = Particle()

def set_joint_angles_worker(joint_values):
    global robot_instance
    robot_instance.set_joint_positions(joint_values)

def set_object_position_worker(object_position):
    global robot_instance
    robot_instance.set_object_position(object_position)

def get_object_position_worker(args):
    global robot_instance
    robot_instance.print_object_position()

def get_joint_angles_worker(args):
    global robot_instance
    robot_instance.print_robot_joints()


class Particle:
    def __init__(self):
        self.p_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
        self.p_env.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p_env.setGravity(0, 0, -9.81)

        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)

        self.p_env.resetDebugVisualizerCamera(cameraDistance=1., cameraYaw=90, cameraPitch=-50, cameraTargetPosition=[0.1, 0.15, 0.35])
        self.add_static_obstacles()
        self.add_movable_obstacles()
        self.robot = self.add_robot()

    def add_static_obstacles(self):
        plane_id = self.p_env.loadURDF("plane.urdf")
        pos, ori = self.p_env.getBasePositionAndOrientation(plane_id)
        self.table_pos_1 = [0.46, -0.01, 0.710]
        table_ori_1 = self.p_env.getQuaternionFromEuler([0, 0, 0])
        table_id_1 = self.p_env.loadURDF("table.urdf", self.table_pos_1, table_ori_1, useFixedBase=1)

        barry_pos_1 = [-0.694, 0.443, 0.895]
        barry_ori_1 = self.p_env.getQuaternionFromEuler([0, math.pi / 2, 0])
        barry_id_1 = self.p_env.loadURDF("barrier.urdf", barry_pos_1, barry_ori_1, useFixedBase=1)

        barry_pos_2 = [-0.694, -0.607, 0.895]
        barry_ori_2 = self.p_env.getQuaternionFromEuler([0, math.pi / 2, 0])
        barry_id_2 = self.p_env.loadURDF("barrier.urdf", barry_pos_2, barry_ori_2, useFixedBase=1)

        barry_pos_3 = [0.459, -0.972, 0.895]
        barry_ori_3 = self.p_env.getQuaternionFromEuler([0, math.pi / 2, math.pi / 2])
        barry_id_3 = self.p_env.loadURDF("barrier.urdf", barry_pos_3, barry_ori_3, useFixedBase=1)

        barry_pos_4 = [-0.549, 0.61, 0.895]
        barry_ori_4 = self.p_env.getQuaternionFromEuler([0, math.pi / 2, math.pi / 2])

        barry_pos_5 = [0.499, 0.61, 0.895]
        barry_ori_5 = self.p_env.getQuaternionFromEuler([0, math.pi / 2, math.pi / 2])

        board_pos_1 = [0.274, 0.581, 0.87575]
        board_ori_1 = self.p_env.getQuaternionFromEuler([math.pi / 2, math.pi / 2, 0])
        board_id_1 = self.p_env.loadURDF("board.urdf", board_pos_1, board_ori_1, useFixedBase=1)

    def add_movable_obstacles(self):
        ketchup_pos_1 = [0.274, 0.381, 0.085 + self.table_pos_1[2]]
        ketchup_ori_1 = self.p_env.getQuaternionFromEuler([math.pi / 2, 0, 0])
        ketchup_id_1 = self.p_env.loadURDF(os.path.expanduser("YcbTomatoSoupCan/model.urdf"), ketchup_pos_1, ketchup_ori_1)

    def add_robot(self):
        panda_robot_start_pos = [0, 0, 0.02 + self.table_pos_1[2]]
        panda_robot_start_ori = [0, 0, 0, 1]
        panda_robot_id = self.p_env.loadURDF(os.path.expanduser("franka_panda/panda.urdf"), panda_robot_start_pos, panda_robot_start_ori, useFixedBase=1)
        return panda_robot_id

    def set_joint_positions(self, joint_values):
        if should_print:
            print("")
            print(f'setting joint positions to: {joint_values}')

        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                self.p_env.setJointMotorControl2(self.robot,
                                                 joint_index + 2,
                                                 self.p_env.POSITION_CONTROL,
                                                 targetPosition=joint_values[joint_index])
            else:
                self.p_env.setJointMotorControl2(self.robot,
                                                 joint_index,
                                                 self.p_env.POSITION_CONTROL,
                                                 targetPosition=joint_values[joint_index])
        for _ in range(240):
            self.p_env.stepSimulation()

    def set_object_position(self, object_position):
        self.p_env.resetBasePositionAndOrientation(5, object_position, [0.5, 0.5,-0.5, 0.5])
        for _ in range(240):
            self.p_env.stepSimulation()
            time.sleep(1/240.0)

    def print_object_position(self):
        position = self.p_env.getBasePositionAndOrientation(5)[0]
        if should_print:
            print(f'x: {position[0]}, y: {position[1]}, z: {position[2]}')

    def print_robot_joints(self):
        num = self.p_env.getNumJoints(self.robot)
        joint_values = []
        for i in range(num):
            joint_state = self.p_env.getJointState(self.robot, i)
            joint_values.append(joint_state)
        if should_print:
            print(f'joint values: {joint_values}')

class ParticleFiltering:
    def __init__(self):
        self.env_num = particles
        self.particle_pool = None

    def setup_pool(self):
        self.particle_pool = self.particles_pool = multiprocessing.Pool(processes=self.env_num, initializer=init_worker)

    def set_robot_joint_angles_randomly_to_all_particles(self):
        joint_values_list = [generate_random_joint_values() for _ in range(self.env_num)]
        self.particles_pool.map(set_joint_angles_worker, joint_values_list)

    def set_object_position_to_all_particles(self):
        object_positions = [
            [x, y, z]
            for x, y, z in zip(range(self.env_num), range(self.env_num), range(self.env_num))
        ]
        print(object_positions)
        self.particles_pool.map(set_object_position_worker, object_positions)

    def print_robot_joint_angles_from_all_particles(self):
        self.particles_pool.map(get_joint_angles_worker, range(self.env_num))

    def print_object_position_from_all_particles(self):
        self.particles_pool.map(get_object_position_worker, range(self.env_num))

    def close_pool(self):
        if self.particles_pool:
            self.particles_pool.close()
            self.particles_pool.join()
            self.particles_pool = None


def generate_random_joint_values():
    a1 = random.uniform(-1, 1)
    a2 = random.uniform(-1, 1)
    a3 = random.uniform(-1, 1)
    a4 = -2.133
    a5 = random.uniform(-1, 1)
    a6 = random.uniform( 0, 1)
    a7 = random.uniform(-1, 1)
    a8 = 0.0002
    a9 = 0.0002

    return [a1, a2, a3, a4, a5, a6, a7, a8, a9]

def sequential_test():
    robot = Particle()
    s = time.time()
    for i in range(particles):
        robot.set_joint_positions(generate_random_joint_values())
    e = time.time()
    print()
    print(f'Sequential (no thread/multiprocessing) with {particles} robots: {e-s:.2f} seconds')
    print()

def multi_processing_test():
    try:
        s = time.time()
        particle_filtering_mp.set_robot_joint_angles_randomly_to_all_particles()
        e = time.time()

        particle_filtering_mp.set_object_position_to_all_particles()

        particle_filtering_mp.print_robot_joint_angles_from_all_particles()
        particle_filtering_mp.print_object_position_from_all_particles()

        print(f'Stepping simulation using multi-processing with {particles} particles took {e-s:.2f} seconds')
    finally:
        particle_filtering_mp.close_pool()

if __name__ == "__main__":
    particles = 100
    particle_filtering_mp = ParticleFiltering()
    particle_filtering_mp.setup_pool()
    # init_worker()

    # sequential_test()
    multi_processing_test()

