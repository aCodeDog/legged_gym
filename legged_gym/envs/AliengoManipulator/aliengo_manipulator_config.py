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
from legged_gym.envs.base.legged_robot_manipulator_config import LeggedManipuatorRobotCfg, LeggedManipuatorRobotCfgPPO
class AliengoManipulatorRoughCfg( LeggedManipuatorRobotCfg ):
    class env(LeggedManipuatorRobotCfg.env):
        num_envs = 4096
        num_actions = 19
        num_observations = 265
    
    class commands( LeggedManipuatorRobotCfg.commands ):
        curriculum = False
        max_curriculum = 4
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1, 1] # min max [m/s]
            lin_vel_y = [-1., 1.]   # min max [m/s]
            ang_vel_yaw = [0.5, 0.5]    # min max [rad/s]
            heading = [0, 0]
            pos_x = [-5,5]
            pos_y = [-5,5]
            pos_z = [0,0]
            griper_pos_x = [-0.3,0.3]
            griper_pos_y = [-0.3,0.3]
            griper_pos_z = [0.5,0.8]
    class init_state( LeggedManipuatorRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0,   # [rad]
            'RL_hip_joint': 0,   # [rad]
            'FR_hip_joint': -0 ,  # [rad]
            'RR_hip_joint': -0,   # [rad]

            'FL_thigh_joint': 0.67,     # [rad]
            'RL_thigh_joint': 0.67,   # [rad]
            'FR_thigh_joint': 0.67,     # [rad]
            'RR_thigh_joint': 0.67,   # [rad]

            'FL_calf_joint': -1.3,   # [rad]
            'RL_calf_joint': -1.3,    # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3,    # [rad]

            'arm_joint1':0.,
            'arm_joint2':0,
            'arm_joint3':0,
            'arm_joint4':0.,
            'arm_joint5':0.,
            'arm_joint6':0.,
            'jointGripper':0.,
        }
    class rewards( LeggedManipuatorRobotCfg.rewards ):
        class scales( LeggedManipuatorRobotCfg.rewards.scales):
            termination = -0.5
            tracking_lin_vel = 4
            tracking_ang_vel = 1
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -1
            torques = -0.00001
            torques_rate = -0.00001
            dof_vel = -2.5e-7
            dof_acc = -2.5e-8
            base_height = -0.5
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0. 
            action_rate = -0.0001
            stand_still = -0.05
            lifting =0.
            object_distance = 2.
            joint_pos = -0.05


        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.4 # tracking reward = exp(-error^2/sigma)
        object_sigma = 0.4 # grasp reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.7 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.7
        soft_torque_limit = 7.
        base_height_target = 0.34
        max_contact_force = 100. # forces above this value are penalized

    class control( LeggedManipuatorRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 100.,'thigh':300,'calf':300,'arm':300}  # [N*m/rad]
        damping = {'hip': 5,'thigh':8,'calf':8,'arm':5}     # [N*m*s/rad]:300
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 3
        wheel_speed = 0.5 # [rad/s]

    class asset( LeggedManipuatorRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengoZ1/urdf/aliengoZ1.urdf'
        name = "aliengoZ1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf","hip","base"]
        terminate_after_contacts_on = ["base"] #,"arm","hip","thigh", "calf" "RL_foot","RR_foot"
        wheel_name =[] #wheel joints name
        arm_name = ["arm"]
        Gripper = ["Gripper"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

class AliengoManipulatorRoughCfgPPO( LeggedManipuatorRobotCfgPPO ):
    seed =11 
    class algorithm( LeggedManipuatorRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedManipuatorRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'aliengoManipulatorInit_0'
        #resume = True
        num_steps_per_env = 48 # per iteration
        max_iterations = 3000
        #load_run="/home/g54/issac_gym/legged_gym/logs/aliengoManipulatorInit/Dec14_21-58-17_"
        #load_run = "/home/g54/issac_gym/legged_gym/logs/aliengoZ1/Dec06_14-48-41_" # -1 = last run
        #checkpoint = 2000 # -1 = last saved model
        
        #load_run = "/home/g54/issac_gym/legged_gym/logs/aliengoManipulatorInit_1/Dec25_15-06-21_" # -1 = last run
        #checkpoint = 12000 # -1 = last saved model
        #experiment_name = '48machine'
        load_run = "/home/g54/issac_gym/legged_gym/logs/"+experiment_name
        #load_run = "/home/g54/issac_gym/legged_gym/logs/48machine"
        checkpoint = 5000