"""A blocks environment written in Pybullet.

Based on the environment described in ZPK.
"""
from ndr.structs import Predicate, LiteralConjunction, Type, Anti
from .spaces import LiteralSpace, LiteralSetSpace

from gym import utils, spaces
from gym.utils import seeding
import gym
import numpy as np
import itertools

import pybullet as p

import glob
import os
import pdb


DIR_PATH = os.path.dirname(os.path.abspath(__file__))

### First set up an environment that is just the low-level physics
def inverse_kinematics(body_id, end_effector_id, target_position, target_orientation, joint_indices, physics_client_id=-1):
    """
    Parameters
    ----------
    body_id : int
    end_effector_id : int
    target_position : (float, float, float)
    target_orientation : (float, float, float, float)
    joint_indices : [ int ]
    
    Returns
    -------
    joint_poses : [ float ] * len(joint_indices)
    """
    lls, uls, jrs, rps = get_joint_ranges(body_id, joint_indices, physics_client_id=physics_client_id)

    all_joint_poses = p.calculateInverseKinematics(body_id, end_effector_id, target_position,
        targetOrientation=target_orientation,
        lowerLimits=lls, upperLimits=uls, jointRanges=jrs, restPoses=rps,
        physicsClientId=physics_client_id)

    # Find the free joints
    free_joint_indices = []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(body_id, idx, physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joint_indices.append(idx)

    # Find the poses for the joints that we want to move
    joint_poses = []

    for idx in joint_indices:
        free_joint_idx = free_joint_indices.index(idx)
        joint_pose = all_joint_poses[free_joint_idx]
        joint_poses.append(joint_pose)

    return joint_poses

def get_joint_ranges(body_id, joint_indices, physics_client_id=-1):
    """
    Parameters
    ----------
    body_id : int
    joint_indices : [ int ]

    Returns
    -------
    lower_limits : [ float ] * len(joint_indices)
    upper_limits : [ float ] * len(joint_indices)
    joint_ranges : [ float ] * len(joint_indices)
    rest_poses : [ float ] * len(joint_indices)
    """
    lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)

    for i in range(num_joints):
        joint_info = p.getJointInfo(body_id, i, physicsClientId=physics_client_id)

        # Fixed joint so ignore
        qIndex = joint_info[3]
        if qIndex <= -1:
            continue

        ll, ul = -2., 2.
        jr = 2.

        # For simplicity, assume resting state == initial state
        rp = p.getJointState(body_id, i, physicsClientId=physics_client_id)[0]

        # Fix joints that we don't want to move
        if i not in joint_indices:
            ll, ul =  rp-1e-8, rp+1e-8
            jr = 1e-8

        lower_limits.append(ll)
        upper_limits.append(ul)
        joint_ranges.append(jr)
        rest_poses.append(rp)

    return lower_limits, upper_limits, joint_ranges, rest_poses

def get_kinematic_chain(robot_id, end_effector_id, physics_client_id=-1):
    """
    Get all of the free joints from robot base to end effector.

    Includes the end effector.

    Parameters
    ----------
    robot_id : int
    end_effector_id : int
    physics_client_id : int

    Returns
    -------
    kinematic_chain : [ int ]
        Joint ids.
    """
    kinematic_chain = []
    while end_effector_id > 0:
        joint_info = p.getJointInfo(robot_id, end_effector_id, physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector_id)
        end_effector_id = joint_info[-1]
    return kinematic_chain


class LowLevelPybulletBlocksEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, use_gui=True, sim_steps_per_action=20, physics_client_id=None, max_joint_velocity=0.1):

        self.sim_steps_per_action = sim_steps_per_action
        self.max_joint_velocity = max_joint_velocity

        self.distance_threshold = 0.05

        self.base_position = [0.405 + 0.2869, 0.48 + 0.2641, 0.0]
        self.base_orientation = [0., 0., 0., 1.]

        self.table_height = 0.42 + 0.205

        self.camera_distance = 1.5
        self.yaw = 90
        self.pitch = -24
        self.camera_target = [1.65, 0.75, 0.42]

        if physics_client_id is None:
            if use_gui:
                self.physics_client_id = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(self.camera_distance, self.yaw, self.pitch, self.camera_target)
            else:
                self.physics_client_id = p.connect(p.DIRECT)
        else:
            self.physics_client_id = physics_client_id

        self.use_gui = use_gui
        self.setup()
        self.seed()

    def sample_initial_state(self):
        """Sample blocks
        """
        # Block name : block state
        state = {}
        block_name_counter = itertools.count()

        # For now, constant orientation (quaternion) for all blocks.
        orn_x, orn_y, orn_z, orn_w = 0., 0., 0., 1.

        num_piles = self.np_random.randint(1, 4)
        for pile in range(num_piles):
            num_blocks_in_pile = self.np_random.randint(1, 4)
            # Block stack blocks.
            x, y = 1.5, 0.5 + pile*0.2
            previous_block_top = 0.5
            for i in range(num_blocks_in_pile):
                block_name = "block{}".format(next(block_name_counter))
                w, l, h, mass, friction = self.sample_block_static_attributes()
                z = previous_block_top + h/2.
                previous_block_top += h
                attributes = [w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction]
                state[block_name] = attributes

        return state

    def sample_block_static_attributes(self):
        w, l, h = self.np_random.normal(0.1, 0.01, size=(3,))
        mass = self.np_random.uniform(0.05, 0.2)
        friction = self.np_random.uniform(0.1, 1.0)
        return w, l, h, mass, friction

    def setup(self):
        p.resetSimulation(physicsClientId=self.physics_client_id)

        # Load plane
        p.setAdditionalSearchPath(DIR_PATH)
        p.loadURDF("assets/urdf/plane.urdf", [0, 0, -1], useFixedBase=True, physicsClientId=self.physics_client_id)

        # Load Fetch
        self.fetch_id = p.loadURDF("assets/urdf/robots/fetch.urdf", useFixedBase=True, physicsClientId=self.physics_client_id)
        p.resetBasePositionAndOrientation(self.fetch_id, self.base_position, self.base_orientation, physicsClientId=self.physics_client_id)

        # Get end effector
        joint_names = [p.getJointInfo(self.fetch_id, i, physicsClientId=self.physics_client_id)[1].decode("utf-8") \
                       for i in range(p.getNumJoints(self.fetch_id, physicsClientId=self.physics_client_id))]

        self.ee_id = joint_names.index('gripper_axis')
        self.ee_orientation = [1., 0., -1., 0.]

        self.arm_joints = get_kinematic_chain(self.fetch_id, self.ee_id, physics_client_id=self.physics_client_id)
        self.left_finger_id = joint_names.index("l_gripper_finger_joint")
        self.right_finger_id = joint_names.index("r_gripper_finger_joint")
        self.arm_joints.append(self.left_finger_id)
        self.arm_joints.append(self.right_finger_id)

        # Load table
        table_urdf = "assets/urdf/table.urdf"
        table_id = p.loadURDF(table_urdf, useFixedBase=True, physicsClientId=self.physics_client_id)
        p.resetBasePositionAndOrientation(table_id, (1.65, 0.75, 0.0), [0., 0., 0., 1.], physicsClientId=self.physics_client_id)

        # Blocks are created at reset
        self.block_ids = {}

        # Set gravity
        p.setGravity(0., 0., -10., physicsClientId=self.physics_client_id)

        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        # Move the arm to a good start location
        joint_values = inverse_kinematics(self.fetch_id, self.ee_id, [1., 0.5, 0.5], self.ee_orientation, self.arm_joints, 
            physics_client_id=self.physics_client_id)

        # Set arm joint motors
        for joint_idx, joint_val in zip(self.arm_joints, joint_values):
            p.resetJointState(self.fetch_id, joint_idx, joint_val, physicsClientId=self.physics_client_id)

        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        # Record the initial state so we can reset to it later
        self.initial_state_id = p.saveState(physicsClientId=self.physics_client_id)

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def set_state(self, state):
        # Blocks are always recreated on reset because their size, mass, friction changes
        self.static_block_attributes = {}
        self.block_ids = {}

        for block_name, block_state in state.items():
            color = self.get_color_from_block_name(block_name)
            block_id = self.create_block(block_state, color=color)
            self.block_ids[block_name] = block_id

        # Let the world run for a bit
        for _ in range(250):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def get_color_from_block_name(self, block_name):
        colors = [
            (0.95, 0.05, 0.1, 1.),
            (0.05, 0.95, 0.1, 1.),
            (0.1, 0.05, 0.95, 1.),
            (0.4, 0.05, 0.6, 1.),
            (0.6, 0.4, 0.05, 1.),
            (0.05, 0.04, 0.6, 1.),
            (0.95, 0.95, 0.1, 1.),
            (0.95, 0.05, 0.95, 1.),
            (0.05, 0.95, 0.95, 1.),
        ]
        block_num = int(block_name[len("block"):])
        return colors[block_num % len(colors)]

    def create_block(self, attributes, color=(0., 0., 1., 1.)):
        w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction = attributes

        # Create the collision shape
        half_extents = [w/2., l/2., h/2.]
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.physics_client_id)

        # Create the visual_shape
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, 
            physicsClientId=self.physics_client_id)

        # Create the body
        block_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id, 
            baseVisualShapeIndex=visual_id, basePosition=[x, y, z], baseOrientation=[orn_x, orn_y, orn_z, orn_w],
            physicsClientId=self.physics_client_id)
        p.changeDynamics(block_id, -1, lateralFriction=friction, physicsClientId=self.physics_client_id)

        # Cache block static attributes
        self.static_block_attributes[block_id] = (w, l, h, mass, friction)

        return block_id

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        for block_id in self.block_ids.values():
            p.removeBody(block_id, physicsClientId=self.physics_client_id)

        p.restoreState(stateId=self.initial_state_id, physicsClientId=self.physics_client_id)
        p.resetBasePositionAndOrientation(self.fetch_id, self.base_position, self.base_orientation, physicsClientId=self.physics_client_id)

        initial_state = self.sample_initial_state()
        self.set_state(initial_state)

        return self.get_state(), {}

    def step(self, action):
        action *= 0.05

        ee_delta, finger_action = action[:3], action[3]

        current_position, current_orientation = p.getLinkState(self.fetch_id, self.ee_id, physicsClientId=self.physics_client_id)[4:6]
        target_position = np.add(current_position, ee_delta)

        joint_values = inverse_kinematics(self.fetch_id, self.ee_id, target_position, self.ee_orientation, self.arm_joints, 
            physics_client_id=self.physics_client_id)

        # Set arm joint motors
        for joint_idx, joint_val in zip(self.arm_joints, joint_values):
            p.setJointMotorControl2(bodyIndex=self.fetch_id, jointIndex=joint_idx, controlMode=p.POSITION_CONTROL,
                targetPosition=joint_val, physicsClientId=self.physics_client_id)

        # Set finger joint motors
        for finger_id in [self.left_finger_id, self.right_finger_id]:
            current_val = p.getJointState(self.fetch_id, finger_id, physicsClientId=self.physics_client_id)[0]
            target_val = current_val + finger_action
            p.setJointMotorControl2(bodyIndex=self.fetch_id, jointIndex=finger_id, controlMode=p.POSITION_CONTROL,
                targetPosition=target_val, physicsClientId=self.physics_client_id)

        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        obs = self.get_state()
        done = False
        info = {}
        reward = 0.

        return obs, reward, done, info

    def close(self):
        p.disconnect(self.physics_client_id)

    def get_state(self):
        gripper_position, gripper_velocity  = p.getLinkState(self.fetch_id, self.ee_id, physicsClientId=self.physics_client_id)[4:6]
        left_finger_pos = [p.getJointState(self.fetch_id, self.left_finger_id, physicsClientId=self.physics_client_id)[0]]

        obs = {
            'gripper' : [gripper_position, gripper_velocity, left_finger_pos],
            'blocks' : {},
        }

        for block_name, block_id in self.block_ids.items():
            obs['blocks'][block_name] = self.get_block_attributes(block_id)

        return obs

    def get_block_attributes(self, block_id):
        w, l, h, mass, friction = self.static_block_attributes[block_id]

        (x, y, z), (orn_x, orn_y, orn_z, orn_w) = p.getBasePositionAndOrientation(block_id, 
            physicsClientId=self.physics_client_id)

        attributes = [w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction]
        return attributes

    def render(self, mode='human', close=False):

        if not self.use_gui:
            raise Exception("Rendering only works with GUI on. See https://github.com/bulletphysics/bullet3/issues/1157")

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.physics_client_id)

        width, height = 2*(3350//8), 2*(1800//8)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width / height),
            nearVal=0.1, farVal=100.0,
            physicsClientId=self.physics_client_id)

        (_, _, px, _, _) = p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physics_client_id)

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


class PybulletBlocksEnv(LowLevelPybulletBlocksEnv):
    pass


