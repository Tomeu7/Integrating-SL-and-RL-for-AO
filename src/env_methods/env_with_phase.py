import os
from src.env_methods.env import AoEnv
from src.unet.unet import UnetGenerator
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from gym import spaces
from collections import deque
import math
import time
import matplotlib.pyplot as plt


class AoEnvWithPhase(AoEnv):
    def __init__(self,
                 gain_factor_phase, config_env_rl=None, parameter_file=None, seed=None, device=None):

        print("----------------------")
        print("--- Env with phase ---")
        print("----------------------")
        self.gain_factor_phase = gain_factor_phase

        super(AoEnvWithPhase, self).__init__(config_env_rl, parameter_file, seed, device, override_generate_phase_projectors=True)

        self.s_dm_residual_phase_history = \
            deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_phase'])
        self.s_dm_residual_phase_tt_history = \
            deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_phase_tt'])

        self.list_of_keys_of_state = ['s_dm', 's_dm_residual', 's_dm_residual_rl', 'a_for_reward', 's_dm_residual_tt',
                                      's_dm_tt', 's_dm_residual_phase', 's_dm_residual_phase_tt']

        self.out_mask = self.supervisor.get_s_pupil()
        self.phase_reconstruction = None

    def define_state_action_space(self):

        state_size_channel_0 = int(self.config_env_rl['number_of_previous_s_dm']) +\
                               int(self.config_env_rl['number_of_previous_s_dm_residual']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_rl']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_tt']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_tt']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_phase']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_phase_tt']) + \
                               int(self.config_env_rl['s_dm_residual_rl']) + \
                               int(self.config_env_rl['s_dm_residual']) + \
                               int(self.config_env_rl['s_dm']) + \
                               int(self.config_env_rl['s_dm_tt']) + \
                               int(self.config_env_rl['s_dm_residual_tt']) + \
                               int(self.config_env_rl['s_dm_residual_phase']) + \
                               int(self.config_env_rl['s_dm_residual_phase_tt'])

        observation_shape = (state_size_channel_0,) + self.action_2d_shape
        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
        action_space = spaces.Box(low=-math.inf, high=math.inf, shape=self.action_2d_shape, dtype=np.float32)

        return state_size_channel_0, observation_shape, observation_space, action_space

    def process_with_projectors(self, out, no_filter):

        out_final = self.phase2modes.dot(out[self.out_mask == 1])

        if no_filter:
            pass
        else:
            out_final_tt_from_pzt = self.pzt2tt(out_final[:-2])
            out_final = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(out_final)
            out_final[-2:] += out_final_tt_from_pzt # we sum tt component in pzt
            """
            out_tt = self.pzt2tt(out_final[-2:])
            out_final = np.concatenate([out_pzt, out_tt])
            """

        return out_final

    def step_only_phase(self, no_filter=False):

        # 1)  Non-linear
        phase_reconstruction = self.calculate_phase_residual(no_filter=no_filter)

        phase_reconstruction_with_gain = self.gain_factor_phase * phase_reconstruction

        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(action=phase_reconstruction_with_gain, controller_type="Phase")
        # 3) Compute new wfs image
        self.supervisor.move_atmos_compute_wfs_reconstruction()

    def calculate_phase_residual(self, no_filter,
                                 update_phase_reconstruction_variable=False):
        # 1) Non-linear
        phase = self.get_phase_and_process_it()
        phase_reconstruction = - self.process_with_projectors(phase,
                                                              no_filter=no_filter)  # negative because we need inverse prediction
        if update_phase_reconstruction_variable:
            self.phase_reconstruction = phase_reconstruction
        return phase_reconstruction

    def build_state(self):

        self.build_state_simple()

        # here we append to self.s_next_main

        if self.config_env_rl['s_dm_residual_phase'] or self.config_env_rl['s_dm_residual_phase_tt']:
            phase_rec = self.calculate_phase_residual(update_phase_reconstruction_variable=True,
                                                      no_filter=False)
            if self.config_env_rl['s_dm_residual_phase']:
                s_phase_pzt = \
                    self.preprocess_dm_info(phase_rec,
                                            sum_tt_projection=False)
                self.s_next_main = self.add_s_dm_info(self.s_next_main, s_phase_pzt,
                                                      key_attr="s_dm_residual_phase", key_norm="dm_residual")

            if self.config_env_rl['s_dm_residual_phase_tt']:
                s_phase_tt = self.config_env_rl['scaling_for_residual_tt'] * phase_rec[-2:]
                s_phase_pzt_from_tt = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2pzt(s_phase_tt))
                self.s_next_main = self.add_s_dm_info(self.s_next_main, s_phase_pzt_from_tt,
                                                      key_attr="s_dm_residual_phase_tt", key_norm="dm_residual")

    def calculate_reward(self, a, a_pzt, a_tt):

        phase_reconstruction = self.phase_reconstruction
        # For metrics
        self.reconstruction_for_reward = phase_reconstruction
        # Build reward
        r2d_phase_rec = -np.square(self.preprocess_dm_info(phase_reconstruction, sum_tt_projection=self.config_env_rl['joint_tt_into_reward'], normalise_for_reward=self.config_env_rl['normalise_for_reward']))
        # Divide by number of actuators
        r2d_phase_rec /= r2d_phase_rec.reshape(-1).shape[0]
        if not self.config_env_rl['joint_tt_into_reward'] and self.control_tt:
            tt_reconstruction = self.config_env_rl['scaling_for_residual_tt'] * non_linear_reconstruction[-2:]
            pzt_reconstruction_from_tt_2d = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2pzt(tt_reconstruction))
            if self.config_env_rl['normalise_for_reward']:
                assert "dm_residual_tt" in self.norm_parameters
                pzt_reconstruction_from_tt_2d = self.standardise(pzt_reconstruction_from_tt_2d, key="dm_residual_tt")
            # Squared and multiply by -1
            r2d_pzt_reconstruction_from_tt = -np.square(pzt_reconstruction_from_tt_2d)
            r2d_phase_rec = np.stack([r2d_phase_rec, r2d_pzt_reconstruction_from_tt])

        # Scale
        r2d_phase_rec = r2d_phase_rec * self.config_env_rl['reward_scale']

        if self.config_env_rl['reward_type'] == "scalar_actuators":
            return r2d_phase_rec.mean()
        elif self.config_env_rl['reward_type'] == "2d_actuators":
            return r2d_phase_rec
        else:
            raise NotImplementedError

    def step(self, a, controller_type):
        a_2d = None
        if controller_type == "RL":
            # 1) Rescale actions and transform them into 1D
            a, a_2d = self.step_process_rl_action(a)
        elif controller_type == "Phase" or controller_type == "Phasev2":
            a = self.phase_reconstruction

        phase_command = self.phase_reconstruction
        if self.config_env_rl['correction_pzt_only_rl_tt']:
            phase_command[-2:] = 0
        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(a,
                                                   controller_type=controller_type,
                                                   unet_or_phase_command=phase_command)

        r = self.calculate_reward(a_2d, a_pzt=self.a_pzt, a_tt=self.a_pzt_from_tt)
        sr_se, sr_le, _, _ = self.supervisor.target.get_strehl(0)
        info = {"sr_se": sr_se,
                "sr_le": sr_le}
        # 3) Move atmos, compute WFS and reconstruction
        self.supervisor.move_atmos_compute_wfs_reconstruction()
        # 4) Build state
        self.build_state()
        s = self.get_next_state(return_dict=False)

        if self.supervisor.iter % (self.config_env_rl['reset_strehl_every_and_print'] + 1) == 0 and \
                self.supervisor.iter > 1:
            self.supervisor.target.reset_strehl(0)

        return s, r, False, info
