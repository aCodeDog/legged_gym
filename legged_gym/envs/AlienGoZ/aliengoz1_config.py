# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import sys
sys.path.append("/home/g54/issac_gym")
sys.path.append("/home/g54/issac_gym/rsl_rl")
print(sys.path)
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class aliengoZ1RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 64
        num_actions = 19
        num_observations = 265

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
   
    class commands( LeggedRobotCfg ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1, 1] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.05, 0.05]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.67,     # [rad]
            'RL_thigh_joint': 0.67,   # [rad]
            'FR_thigh_joint': 0.67,     # [rad]
            'RR_thigh_joint': 0.67,   # [rad]

            'FL_calf_joint': -1.3,   # [rad]
            'RL_calf_joint': -1.3,    # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3,    # [rad]

            'arm_joint1':0.,
            'arm_joint2':0.,
            'arm_joint3':0.1,
            'arm_joint4':0.,
            'arm_joint5':0.1,
            'arm_joint6':0.,
            'jointGripper':0.,
        }
    class rewards:
        class scales:
            termination = -0.0001
            tracking_lin_vel = 3
            tracking_ang_vel =3
            lin_vel_z = -0.1
            ang_vel_xy = -0.05
            orientation = -0.7
            base_height = -0.9
            torques = -0.000001
            dof_vel = -0.
            dof_acc = -2.5e-9
            stand_still = -0.001
            lifting = 0.
            object_distance = 0
            joint_pos = -0.
            orientation_quat = 5

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.4 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.7 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.7
        soft_torque_limit = 0.8
        base_height_target = 0.34
        max_contact_force = 100. # forces above this value are penalized

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 100.,'thigh':300,'calf':300,'arm':300}  # [N*m/rad]
        damping = {'hip': 5,'thigh':8,'calf':8,'arm':5}     # [N*m*s/rad]:300
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        task ="arm"

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengoZ1/urdf/aliengoZ1.urdf'
        name = "aliengoZ1"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        self_collisions =0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True

class aliengoZ1RoughCfgPPO( LeggedRobotCfgPPO ):
    seed =10
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        #experiment_name = 'aliengoZ1_0'
        experiment_name = '48machine'
        #resume = True
        num_steps_per_env = 36 # per iteration
        max_iterations = 3000
        load_run = "/home/g54/issac_gym/legged_gym/logs/"+ experiment_name# -1 = last run
        checkpoint = 6001 # -1 = last saved model
        #load_run = "/home/g54/issac_gym/legged_gym/logs/48machine"
        #checkpoint = 3001