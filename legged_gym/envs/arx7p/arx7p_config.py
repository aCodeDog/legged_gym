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
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Arx7RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 6000
        num_actions = 12
        num_observations = 244
    class commands( LeggedRobotCfg ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 1.0] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [0,0]    # min max [rad/s]
            heading = [-3.14, 3.14]
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
        terrain_proportions = [0.1, 0.1, 0.8, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
   
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.60] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'right_ankle': 0.66,   # [rad]
            'right_knee': -1.54,   # [rad]
            'right_hip': 0.88 ,  # [rad]

            'left_ankle': 0.66,   # [rad]
            'left_knee': -1.54,   # [rad]
            'left_hip': 0.88 ,  # [rad]

            
            'joint0':0.,           
            'joint1':0.,
            'joint2':0.,
            'joint3':0.,
            'joint4':0.,
            'joint5':0.,

        }
    class rewards:
        class scales:
            termination = -0.1
            tracking_lin_vel = 4.0
            tracking_ang_vel = 4.0
            lin_vel_z = -0.01
            ang_vel_xy = -0.9
            orientation = -1
            torques = -0.0001
            dof_vel = -2.5e-7
            dof_acc = -2.5e-7
            base_height = -0.5
            feet_air_time =  1.5
            collision = -0.2
            feet_stumble = -0.01
            action_rate = -0.01
            stand_still = -1.
            joint_pos = -0.3
            dof_pos_limits = -0.1
            arm_pos = -0.
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.4 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.
        base_height_target = 0.55
        max_contact_force = 100. # forces above this value are penalized

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 10.,"ankle":10,"knee":10,"hip":10}  # [N*m/rad]
        damping = {'joint': 0.1,"ankle":0.2,"knee":0.2,"hip":0.2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        wheel_speed = 0.3 # [rad/s]
        task = "rough_terrain"

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/arx7p/urdf/arx7p.urdf'
        name = "arx7p"
        foot_name = "wheel"  #link name
        arm_name = "joint"  #joint name
        penalize_contacts_on = ["base","knee","hip"]  #link name
        terminate_after_contacts_on = []  #link name
        wheel_name =[ "ankle"] #wheel joints name, joint name
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter "base","calf","hip","thigh"
  

class Arx7RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'arx7p_0'
        #resume = True
        num_steps_per_env = 48 # per iteration
        max_iterations = 3020
        load_run = "/home/g54/issac_gym/legged_gym/logs/"+ experiment_name# -1 = last run
        #load_run = "/home/g54/issac_gym/legged_gym/logs/48machine"
        checkpoint = 3000 
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'selu'   # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    