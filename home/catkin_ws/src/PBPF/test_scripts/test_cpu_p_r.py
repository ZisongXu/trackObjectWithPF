import multiprocessing
import time
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
import os
import math
import random

robot_instance = None

def init_worker():
    global robot_instance
    robot_instance = Robot()

def worker(joint_values):
    global robot_instance
    robot_instance.set_joint_positions(joint_values)
    return True

class Robot:
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
        print("")
        print("ketchup_id_1:", ketchup_id_1)
    def add_robot(self):
        panda_robot_start_pos = [0, 0, 0.02 + self.table_pos_1[2]]
        panda_robot_start_ori = [0, 0, 0, 1]
        panda_robot_id = self.p_env.loadURDF(os.path.expanduser("franka_panda/panda.urdf"), panda_robot_start_pos, panda_robot_start_ori, useFixedBase=1)
        return panda_robot_id

    def set_joint_positions(self, joint_values):
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

        # print(f'I am a robot with these {joint_values} joint values')

class ParticleFiltering:
    def __init__(self):
        self.env_num = particles

class ParticleFilteringMultiProcess(ParticleFiltering):
    def __init__(self):
        super().__init__()
        self.pool = None

    def setup_pool(self):
        self.pool = multiprocessing.Pool(processes=self.env_num, initializer=init_worker)

    def operate(self):
        if self.pool is None:
            raise RuntimeError("Pool has not been initialised. Call setup_pool() first.")

        joint_values_list = [generate_random_joint_values() for _ in range(self.env_num)]
        results = self.pool.map(worker, joint_values_list)

    def close_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

def generate_random_joint_values():
    a1 = random.uniform(-1, 1)
    a2 = random.uniform(-1, 1)
    a3 = random.uniform(-1, 1)
    a4 = -2.133
    a5 = random.uniform(-1, 1)
    a6 = random.uniform(-1, 1)
    a7 = random.uniform(-1, 1)
    a8 = 0.0002
    a9 = 0.0002

    return [a1, a2, a3, a4, a5, a6, a7, a8, a9]


if __name__ == "__main__":
    particles = 10
    particle_filtering_mp = ParticleFilteringMultiProcess()
    particle_filtering_mp.setup_pool()
    # init_worker()

    # s = time.time()
    # robot = Robot()
    # for i in range(particles):
    #     robot.set_joint_positions(generate_random_joint_values())
    # e = time.time()
    # print()
    # print(f'Sequential (no thread/multiprocessing) with {particles} robots: {e-s:.2f} seconds')
    # print()

    try:
        s = time.time()
        particle_filtering_mp.operate()
        e = time.time()
        print(f'Multi-processing with 70 robots: {e-s:.2f} seconds')
    finally:
        particle_filtering_mp.close_pool()

