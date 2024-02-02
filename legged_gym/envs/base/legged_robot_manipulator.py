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
sys.path.append("/home/g54/issac_gym/legged_gym")
sys.path.append("/home/g54/issac_gym/rsl_rl")
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
from copy import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
import math
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_manipulator_config import LeggedManipuatorRobotCfg

class LeggedManipulatorRobot(BaseTask):
    def __init__(self, cfg: LeggedManipuatorRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.actor_num = 1 # legged robot,box,table
        self.create_box = False
        
        self.arm_u = torch.zeros((self.cfg.env.num_envs, 12), dtype=torch.float, device=sim_device)
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.last_torques = self.torques
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            #print("self.torques.shape:",self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(-1, 13)
        self.robot_root_states = self.root_states[self.robot_env_ids,:]
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.gripperMover_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "gripperMover")
        self.base_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "base")
        self._gripper_state = self.rigid_body_states[:, self.gripperMover_handles][:, 0:13]
        self._gripper_pos = self.rigid_body_states[:, self.gripperMover_handles][:, 0:3]
        self._gripper_rot = self.rigid_body_states[:, self.gripperMover_handles][:, 3:7]
        self.base_pos = self.rigid_body_states[:, self.base_handles][:, 0:3]
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.robot_root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        self.episode_length_buf += 1
        self.common_step_counter += 1

        

        # prepare quantities
        self.base_quat[:] = self.robot_root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()
        if(self.create_box ==True):
        # #update cube state
            self.cube_object_state = self.root_states[self.cube_object_indices, 0:13]
            self.cube_object_pose = self.root_states[self.cube_objects_indices, 0:7]
            self.cube_object_pos = self.root_states[self.cube_object_indices, 0:3]
            self.cube_object_rot = self.root_states[self.cube_object_indices, 3:7]
            self.cube_object_linvel = self.root_states[self.cube_object_indices, 7:10]
            self.cube_object_angvel = self.root_states[self.cube_object_indices, 10:13]
        else:
            self.update_obj_pos()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.robot_root_states[:, 7:13]
        

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        body_force=self.contact_forces[:, self.termination_contact_indices, :]
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        #print("self._reset_dofs(env_ids):",env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        #cube reset
        self.lifted_object[env_ids] = False
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            
            # if(name =="object_distance" ):
            #     print("object_distance rew:",rew)
               
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.dof_pos,
                                    self.actions
                                    ,
                                    #self.dof_pos,
                                    #self.default_dof_pos,
                                    self._gripper_pos,
                                    self.cube_object_pos,
                                    (self._gripper_pos-self.cube_object_pos)* self.obs_scales.object_distance
                                    ),dim=-1)
        # add perceptive inputs if not blind
        #print("####compute_observations:",(self.dof_pos - self.default_dof_pos).shape,self.dof_pos.shape,self.default_dof_pos.shape)
        #print("self.obs_buf.shape",self.obs_buf.shape)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.robot_root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]) 
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.) # yaw error 

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, 0]  =  self.arm_u[env_ids,0]
        # self.commands[env_ids, 1]  =  self.arm_u[env_ids,1]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    def orientation_error(self,desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def control_ik(self,j_eef,dpose):
        #global damping, j_eef, num_envs
        # solve damped least squares

        damping = 0.5
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device= self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 12) #Z1 hava 6 Dof
        return u
       #pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
    # def control_osc(dpose):
    #     global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    #     mm_inv = torch.inverse(mm)
    #     m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    #     m_eef = torch.inverse(m_eef_inv)
    #     u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
    #         kp * dpose - kd * hand_vel.unsqueeze(-1))

    #     # Nullspace control torques `u_null` prevents large changes in joint configuration
    #     # They are added into the nullspace of OSC so that the end effector orientation remains constant
    #     # roboticsproceedings.org/rss07/p31.pdf
    #     j_eef_inv = m_eef @ j_eef @ mm_inv
    #     u_null = kd_null * -dof_vel + kp_null * (
    #         (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    #     u_null = u_null[:, :7]
    #     u_null = mm @ u_null
    #     u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    #     return u.squeeze(-1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        robot_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        robot_dof_dict = self.gym.get_asset_dof_dict(self.robot_asset)
        #print("robot_dof_dict:",robot_dof_dict) #：arm 12~18 arm_joint1~jointGripper
        #print("robot_link_dict:",robot_link_dict)  # arm: 18~ 24arm_link01~gripperMover
        Z1_hand_index = robot_link_dict["gripperMover"]
        Z1_dof_index_start = robot_dof_dict["arm_joint1"]
        Z1_dof_index_end = robot_dof_dict["arm_joint6"]
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        _jacobian = gymtorch.wrap_tensor(self._jacobian)
        Z1_j_eef_1 = _jacobian[:, Z1_hand_index , :, :6]
        Z1_j_eef_2 = _jacobian[:, Z1_hand_index , :, Z1_dof_index_start+6:Z1_dof_index_end+6+1]
        Z1_j_eef  = torch.cat((Z1_j_eef_1,Z1_j_eef_2),dim=-1)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        arm_joint_state = self.dof_vel[:,Z1_dof_index_start:Z1_dof_index_end+1]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        base_link_state = torch.cat((self.base_lin_vel,self.base_ang_vel),dim=-1)
        Q_j = torch.cat((base_link_state,arm_joint_state),dim=-1)
        self.gripperMover_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "gripperMover")
        self._gripper_state = self.rigid_body_states[:, self.gripperMover_handles][:, 0:13]
        self._gripper_pos = self.rigid_body_states[:, self.gripperMover_handles][:, 0:3]
        self._gripper_rot = self.rigid_body_states[:, self.gripperMover_handles][:, 3:7]
        #print("Z1_j_eef.shape",Z1_j_eef.shape)
        # print("arm_joint_state.shape:",arm_joint_state.shape)
        # print("base_link_state:",base_link_state.shape)
        #print("Q_j.shape:",Q_j.shape)
        if(self.create_box ==True):
            

            #j_eef_T = torch.transpose(j_eef, 1, 2)
            #print("Q_j:",Q_j)
            #print("Q_j:",Q_j.unsqueeze(-1))
            #V_EE = Z1_j_eef @ Q_j.unsqueeze(-1)
            self.cube_object_pos = self.root_states[self.cube_object_indices, 0:3]
            self.cube_object_rot = self.root_states[self.cube_object_indices, 3:7]
            pos_err = self.cube_object_pos - self._gripper_pos
            orn_err = self.orientation_error(self.cube_object_rot, self._gripper_rot)
            base_pos_err = self.cube_object_pos - self.robot_root_states[:, 0:3]
            base_orn_err = self.orientation_error(self.cube_object_rot, self.robot_root_states[:, 3:7])
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            self.arm_u = self.control_ik(Z1_j_eef,dpose)
        else:
            self.cube_object_rot = torch.zeros((self.cfg.env.num_envs, 4), dtype=torch.float, device=self.device)
            self.arm_u =torch.zeros((self.cfg.env.num_envs, 12), dtype=torch.float, device=self.device)
        #print("V_EE.shape:",V_EE.shape)
        #print("V_EE:",V_EE)
        #print("gripperMover:",self._gripper_state[:,7:13])
        
        if control_type=="P":
            #arm
            #pos_err = self.cube_object_pos - hand_pos
            self.default_dof_pos[:,12:18] = self.default_dof_pos[:,12:18]+self.arm_u[:,6:12]*self.dt
            dof_err = self.default_dof_pos - self.dof_pos
        #print("1dof_err:",dof_err)
            dof_err[:,self.wheel_indices] =  self.cfg.control.wheel_speed
            #joint_pos_err=self.default_dof_pos- self.dof_pos
            #dof_err[:,12:18]=self.arm_u[:,6:12]
            torques = self.p_gains*(actions_scaled + dof_err) - self.d_gains*self.dof_vel  #actions_scaled:[num_envs,19] ,self.default_dof_pos:[1,19],self.default_dof_pos:[19],
            #print("shape:",actions_scaled.shape,self.default_dof_pos.shape,self.p_gains.shape)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.
        #print("default_dof_pos.shape",self.default_dof_pos.shape)
        #self.default_dof_pos[0,12:]=torch_rand_float(-1.5, 1.5, (1, 7), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._global_indices = torch.arange(self.num_envs*self.actor_num, dtype=torch.int32,device=self.device).view(self.num_envs, -1)
        env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        #print("env_ids",env_ids)
        self._global_indices = torch.arange(self.num_envs*self.actor_num, dtype=torch.long,device=self.device).view(self.num_envs, -1)
        env_ids_int32 = self._global_indices[env_ids, 0].flatten() #robot env_ids
        #print("env_ids_int32",env_ids_int32)
        #print("env_ids",env_ids)
        
        if self.custom_origins:
            self.root_states[env_ids_int32] = self.base_init_state
            #print("self.base_init_state.shape:",self.base_init_state.shape)
            self.root_states[env_ids_int32, :3] += self.env_origins[env_ids]
            self.root_states[env_ids_int32, :2] += torch_rand_float(-1., 1., (len(env_ids_int32), 2), device=self.device) # xy position within 1m of the center

            #reset cube state
            if(self.create_box ==True):
                self.root_states[env_ids_int32+2] =self.cube_object_init_state[env_ids]
            else:
                self.choice_obj_seed[:,env_ids_int32] = torch.randint(0, len(self.virtual_obj_pos_indices), (1, len(env_ids_int32)), device=self.device)
        else:
            self.root_states[env_ids_int32] = self.base_init_state
            self.root_states[env_ids_int32, :3] += self.env_origins[env_ids]
            if(self.create_box ==True):
                self.root_states[env_ids_int32+2] =self.cube_object_init_state[env_ids]
                self.root_states[env_ids_int32+1] =self.table_object_init_state[env_ids]
            else:
                self.choice_obj_seed[:,env_ids_int32] = torch.randint(0, len(self.virtual_obj_pos_indices), (1, len(env_ids_int32)), device=self.device)
        # base velocities
        #print("self.base_init_state.shape:",self.base_init_state.shape)
        self.root_states[env_ids_int32, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids_int32), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids_int32.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[self.robot_env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        self._global_indices = torch.arange(self.num_envs*self.actor_num, dtype=torch.int32,device=self.device).view(self.num_envs, -1)
        env_ids_int32 = self._global_indices[env_ids, 0].flatten()        
        distance = torch.norm(self.robot_root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        self._global_indices = torch.arange(self.num_envs*1,device=self.device).view(self.num_envs, -1)# self.num_envs x 3 [0 ~ num_envs x 3]
        # print(self._global_indices )
        envs_num = torch.arange(self.num_envs,device=self.device).view(self.num_envs, -1)
        # print(envs_num)
        self.robot_env_ids = self._global_indices[envs_num, 0].flatten()
        self.robot_env_ids = to_torch(self.robot_env_ids, dtype=torch.long, device=self.device)
        #print(robot_env_ids)
        print("env_ids_int32:",self.robot_env_ids)
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(-1, 13)
        self.robot_root_states = self.root_states[self.robot_env_ids,:]

        #get rigid body name
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.gripperMover_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "gripperMover")
        #print("###self.rigid_body_states.shappe:",self.rigid_body_states)
        #print("####### self.gripperMover_handles:", self.gripperMover_handles)
        #print("######rigid_body_names:",body_names)
        #self.rigid_body_states
        self._gripper_state = self.rigid_body_states[:, self.gripperMover_handles][:, 0:13]
        self._gripper_pos = self.rigid_body_states[:, self.gripperMover_handles][:, 0:3]
        self._gripper_rot = self.rigid_body_states[:, self.gripperMover_handles][:, 3:7]
        #print("##self._palm_state:",self._palm_state)
        #print("####self.rigid_body_states。shape:",self.rigid_body_states.shape)
        #print("####self.rigid_body_states:",self.rigid_body_states)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        #print("self.dof_pos.shape",self.dof_pos.shape)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        print("self.dof_vel.shape",self.dof_vel.shape)
        self.base_quat = self.robot_root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.robot_root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.cube_object_pose = torch.zeros((self.cfg.env.num_envs, 7), dtype=torch.float, device=self.device)
        self.cube_object_pos = torch.zeros((self.cfg.env.num_envs, 3), dtype=torch.float, device=self.device)
        self.cube_object_rot = torch.zeros((self.cfg.env.num_envs, 4), dtype=torch.float, device=self.device)
        
        
        ranges = [(3,13),(2,9)]
        virtual_obj_pos_indices = []
        for j in range(17):
            for k in range(11):
                if j in range(ranges[0][0],ranges[0][1]) and k in range(ranges[1][0],ranges[1][1]):
                    continue
                else:
                    virtual_obj_pos_indices.append(j*11+k)
                        # x = self.cube_object_pos[i, 0] 
                        # y = self.cube_object_pos[i, 1] 
                        # z = self.cube_object_pos[i, 2]

        self.virtual_obj_pos_indices = to_torch(virtual_obj_pos_indices, dtype=torch.long, device=self.device)
        choice_obj_seed = np.random.randint(low=0, high=len(self.virtual_obj_pos_indices), size=(1, self.num_envs))
        self.choice_obj_seed = to_torch(choice_obj_seed, dtype=torch.long, device=self.device)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            print("self.default_dof_pos:",name)
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs,1)
        print("####self.default_dof_pos:shape",self.default_dof_pos.shape)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        print("#######_reward_func_name:",)
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
            print(" ",name,"+ ",scale)
        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        #get rigid body states
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.gripperMover_handles = self.gym.find_asset_rigid_body_index(robot_asset, "gripperMover")

        #print("###self.rigid_body names:",body_names)
        # print("####### self.gripperMover_handles:", self.gripperMover_handles)
        # print("######rigid_body_names:",body_names)
        #self.rigid_body_states
        # self._palm_state = self.rigid_body_states[:, self.gripperMover_handles][:, 0:13]
        # self._palm_pos = self.rigid_body_states[:, self.gripperMover_handles][:, 0:3]
        # self._palm_rot = self.rigid_body_states[:, self.gripperMover_handles][:, 3:7]
        # print("##self._palm_state:",self._palm_state)
        # save body names from the asset
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        #print("###self.dof names:",body_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        wheel_names =[]
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        arm_names =[]
        for name in self.cfg.asset.arm_name:
            arm_names.extend([s for s in self.dof_names if name in s])
        print("###self.rigid_body names:",body_names)
        print("###self.dof names:",body_names)
        print("###penalized_contact_names:",penalized_contact_names)
        print("###termination_contact_names:",termination_contact_names)
        print("###feet_names:",feet_names)
        print("###wheels name:",wheel_names)
        print("###arm_names:",arm_names)
        

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        #creat table and box to grasp

        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        table_opts = gymapi.AssetOptions()
        #table_opts.fix_base_link = True
        table_width = 0.3
        table_lenth = 0.3
        table_thickness = 0.4
        table_dims = gymapi.Vec3(table_width, table_lenth, table_thickness)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(1, 0.0, 0.5 * table_dims.z)
        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)
        
        # Define start pose for table
        table_start_pose = gymapi.Transform()
        #table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        #self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        #self.reward_settings["table_height"] = self._table_surface_pos[2]
        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # Create cubeA asset
        self.cubeA_size = 0.050
        self.keypoint_scale =1.5
        self.object_base_size = self.cubeA_size
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)
        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)



        # create box asset
        # box_size = 0.045
        # asset_options = gymapi.AssetOptions()
        # box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        # self.box_idxs = []
        num_franka_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_bodies = num_franka_bodies+3     # 1 for table, table stand, box
        max_agg_shapes = num_franka_shapes+3     # 1 for table, table stand, box
        
        
        cube_object_init_state=[]
        table_object_init_state=[]
        cube_object_indices = []
        table_object_indices = []
        cube_object_keypoint_offsets=[]
        cube_object_scale = [0.05,0.05,0.05]
        self.keypoints_offsets = self._object_keypoint_offsets()
        self.num_keypoints = len(self.keypoints_offsets)
        object_offsets = []
        #create lift reward index
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.gym.begin_aggregate(env_handle, max_agg_bodies, max_agg_shapes, True)

            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1, 1, (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # Create table
            robot_pos=self.env_origins[i].clone()
            
            table_pos_start = torch.tensor([0.7, 0, 0],device=self.device) +robot_pos
            table_start_pose.p = gymapi.Vec3(*table_pos_start)

            if(self.create_box ==True):
                #table_actor = self.gym.create_actor(env_handle, table_asset, table_start_pose, "table", i+self.num_envs, 1, 0)
                #table_object_idx = self.gym.get_actor_index(env_handle, table_actor, gymapi.DOMAIN_SIM)
                # table_stand_actor = self.gym.create_actor(env_handle, table_stand_asset, table_stand_start_pose, "table_stand",
                #                                            i, 1, 0)

                # Create cubes
                
                cubeA_pos = table_pos_start+torch.tensor([0, 0, table_thickness],device=self.device)
                # cubeA_pos[0]= cubeA_pos[0]-1
                cubeA_pos[:2] += torch_rand_float(0, -table_width/2, (2,1), device=self.device).squeeze(1)
                cubeA_start_pose.p=gymapi.Vec3(*cubeA_pos)
                self._cubeA_id = self.gym.create_actor(env_handle, cubeA_asset, cubeA_start_pose, "cubeA", i+self.num_envs, 1, 0)
                #self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
                # Set colors
                self.gym.set_rigid_body_color(env_handle, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
                table_object_init_state.append(
                    [
                        table_start_pose.p.x,
                        table_start_pose.p.y,
                        table_start_pose.p.z,
                        table_start_pose.r.x,
                        table_start_pose.r.y,
                        table_start_pose.r.z,
                        table_start_pose.r.w,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )            
                #self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)
                cube_object_init_state.append(
                    [
                        cubeA_start_pose.p.x,
                        cubeA_start_pose.p.y,
                        cubeA_start_pose.p.z,
                        cubeA_start_pose.r.x,
                        cubeA_start_pose.r.y,
                        cubeA_start_pose.r.z,
                        cubeA_start_pose.r.w,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )            
                cube_object_idx = self.gym.get_actor_index(env_handle, self._cubeA_id, gymapi.DOMAIN_SIM)
                #print("cube_object_idx:",cube_object_idx)
                #table_object_indices.append(table_object_idx)
                cube_object_indices.append(cube_object_idx)

                for keypoint in self.keypoints_offsets:
                    #print("keypoint:",keypoint)
                    keypoint = copy(keypoint)
                    for coord_idx in range(3):
                        keypoint[coord_idx] *= cube_object_scale[coord_idx] * self.cubeA_size * self.keypoint_scale / 2   #self.object_base_size:0.05, self.keypoint_scale:1.5
                    object_offsets.append(keypoint)
                cube_object_keypoint_offsets.append(object_offsets)
            
            #print("object_idx:",object_idx)
            self.gym.end_aggregate(env_handle)   
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        if(self.create_box ==True):
            self.cube_object_init_state = to_torch(cube_object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
            self.object_keypoint_offsets = to_torch(cube_object_keypoint_offsets, dtype=torch.float, device=self.device)
            self.cube_object_indices = to_torch(cube_object_indices, dtype=torch.long, device=self.device)
            self.table_object_indices = to_torch(table_object_indices, dtype=torch.long, device=self.device)
            self.table_object_init_state = to_torch(table_object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])
            
        self.arm_indices = torch.zeros(len(arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(arm_names)):
            self.arm_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], arm_names[i])
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        print("####self.reward_scales :",self.reward_scales )
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    def obj_visualize(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        heights = self.measured_heights
        height_points = quat_apply_yaw(self.base_quat.repeat(heights.shape[0]), self.height_points)
        for i in range(self.num_envs):
            x = height_points[150, 0] + self.robot_root_states[i, 0]
            y = height_points[150, 1] + self.robot_root_states[i, 1]
            z = height_points[150, 2] +self.robot_root_states[i, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)             

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.robot_root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
                
    def update_obj_pos(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # if not self.terrain.cfg.measure_heights:
        #     return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # obj_num = 1
        # 
        
        # heights = self.measured_heights
        height_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.robot_root_states[:, :3]).unsqueeze(1)
#        obj_pos_indices = 
       
        
        #self.cube_object_pos[:,:3] = torch.cat((x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)),dim=1)
        #select a indice from self.virtual_obj_pos_indices
        
        #self.seed = np.random.randint(low=0, high=len(self.virtual_obj_pos_indices), size=(1, self.num_envs))
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        virtual_indices = self.virtual_obj_pos_indices.repeat(self.num_envs,1)
        seed = to_torch(self.choice_obj_seed, device=self.device, dtype=torch.long).view(self.num_envs,1)
        obj_pos_indices = virtual_indices.gather(1, seed).view(self.num_envs,1)
        obj_pos_indices = obj_pos_indices.repeat(1,3).view(self.num_envs,1,3)
        
        self.virtual_obj_pos = height_points.gather(1,obj_pos_indices).view(self.num_envs,3)
        self.cube_object_pos = self.virtual_obj_pos
        for i in range(self.num_envs):
            x = self.cube_object_pos[i, 0] 
            y = self.cube_object_pos[i, 1] 
            z = self.cube_object_pos[i, 2] 
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.robot_root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.robot_root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    def _object_keypoint_offsets(self):
        return [
            [1, 1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.robot_root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_torques_rate(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques-self.last_torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        #print("_reward_tracking_lin_vel:",torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma).shape)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        #print("rew_airTime:",rew_airTime.shape)
        return rew_airTime
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        #print("self.dof_pos.shape",self.dof_pos.shape)
        #return torch.sum(torch.abs(self.dof_pos[:, 0:12] - self.default_dof_pos[:,0:12]), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1) #only calculate leg 
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1) #only calculate leg 
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    def _reward_lifting(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""

        z_lift = 0.05 + self.cube_object_pos[:, 2] - self.cube_object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)
        self.lifting_bonus_threshold = 0.15
        self.lifting_bonus=300
        # this flag tells us if we lifted an object above a certain height compared to the initial position
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self.lifted_object

        # Since we stop rewarding the agent for height after the object is lifted, we should give it large positive reward
        # to compensate for "lost" opportunity to get more lifting reward for sitting just below the threshold.
        # This bonus depends on the max lifting reward (lifting reward coeff * threshold) and the discount factor
        # (i.e. the effective future horizon for the agent)
        # For threshold 0.15, lifting reward coeff = 3 and gamma 0.995 (effective horizon ~500 steps)
        # a value of 300 for the bonus reward seems reasonable
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold

        # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the
        # keypoint reward
        #lifting_rew *= ~lifted_object

        # update the flag that describes whether we lifted an object above the table or not
        self.lifted_object = lifted_object
        #return lifting_rew, lift_bonus_rew, lifted_object
        return torch.sum(torch.square(lifting_rew))
    def _reward_joint_pos(self):
        # Penalize motion at zero commands
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    def _reward_object_distance(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""
        dis_err = torch.sum(torch.square(self._gripper_pos-self.cube_object_pos), dim=1)
        #dis_err = torch.sum(torch.square(self._gripper_pos-self.robot_root_states[:,0:3]+torch.tensor([0.5, 0.3, 0.4],device=self.device)), dim=1)
        #print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return torch.exp(-dis_err/self.cfg.rewards.object_sigma)
    def _reward_orientation_quat(self):
    # Penalize non flat base orientation
    
        orientation_error = torch.sum(torch.square(self.robot_root_states[:,:7] - self.base_init_state[0:7]),dim=1)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # print("shape:",torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma).shape)
        # print("&****shape:",torch.exp(-orientation_error/self.cfg.rewards.tracking_sigma).shape)
        return torch.exp(-orientation_error/self.cfg.rewards.tracking_sigma)   