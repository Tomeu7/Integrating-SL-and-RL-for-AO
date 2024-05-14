from abc import ABC
import gym
from collections import deque, OrderedDict
import numpy as np
from shesha.supervisor.rlSupervisor import RlSupervisor as Supervisor
from shesha.util.utilities import load_config_from_file
import math
from gym import spaces
from src.env_methods.projectors import ProjectorCreator
import torch
from typing import Tuple

class AoEnv(gym.Env, ABC):

    #            Initialization
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def __init__(self,
                 config_env_rl: dict,
                 parameter_file: str,
                 seed: int,
                 pyr_gpu_ids: list=[0],
                 override_generate_phase_projectors: bool=False,
                 normalization_noise_value_linear:float=-1,
                 noise:float=None) -> None:
        """
        Environment to interact with COMPASS
        :param config_env_rl: RL configuration (it can be empty if used to compute UNet or Linear rec. performance)
        :param parameter_file: parameter file used
        :param seed: seed used for experiments
        :param device: device used for MVMs
        :param pyr_gpu_ids: device used for pyr simulation
        :param override_generate_phase_projectors: if True, phase projectors are created even if tt is not controlled
        :param normalization_noise_value_linear: if >= 0 we clipd values <0 from WFS image for linear rec.
        :param noise: if True the noise from the WFS in the param file is changed
        """
        super(AoEnv, self).__init__()

        self.config_env_rl = config_env_rl
        self.device = pyr_gpu_ids[0]
        # For control TT
        self.control_tt = config_env_rl['control_tt']
        self.two_output_actor = config_env_rl['separate_tt_into_two_actions'] and config_env_rl['control_tt']
        self.a_tt_from_pzt = None
        self.a_pzt_from_tt = None
        self.a_pzt = None
        # For metrics of TT/PZT
        self.reconstruction_for_reward = None

        # Loading Compass config
        fd_parameters = "data/parameter_files/"
        config_compass = load_config_from_file(fd_parameters + parameter_file)
        config_compass.p_loop.set_devices(pyr_gpu_ids)
        if noise is not None:
            config_compass.p_wfs0.noise = noise

        self.supervisor = Supervisor(config=config_compass,
                                     n_reverse_filtered_from_cmat=self.config_env_rl['n_reverse_filtered_from_cmat'],
                                     filter_commands=config_env_rl['filter_commands'],
                                     command_clip_value=config_env_rl['command_clip_value'],
                                     initial_seed=seed,
                                     which_modal_basis="Btt",
                                     mode=self.config_env_rl['mode'],
                                     device=self.device,
                                     control_tt=self.control_tt,
                                     use_second_version_of_modal_basis=self.config_env_rl['use_second_version_of_modal_basis'],
                                     leaky_integrator_for_rl=self.config_env_rl['leaky_integrator_for_rl'],
                                     leak=self.config_env_rl['leak'],
                                     reset_when_clip=self.config_env_rl['reset_when_clip'],
                                     reduce_gain_tt_to=self.config_env_rl['reduce_gain_tt_to']
                                     )

        self.s_pupil = self.supervisor.get_s_pupil()
        # From supervisor for easy access
        self.command_shape = self.supervisor.command_shape  # always in 1D
        self.action_1d_shape = self.supervisor.action_1d_shape
        self.action_2d_shape = self.supervisor.action_2d_shape
        self.pzt_shape = self.supervisor.pzt_shape  # always in 1D
        self.tt_shape = self.supervisor.tt_shape  # always in 1D
        self.mask_valid_actuators = self.supervisor.mask_valid_actuators
        self.num_dm = self.supervisor.num_dm
        # Initializy the history
        self.s_dm_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm'])
        self.s_dm_residual_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual'])
        self.s_dm_residual_rl_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_rl'])
        # tt state
        self.s_dm_residual_tt_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_tt'])
        self.s_dm_tt_history = deque(maxlen=self.config_env_rl['number_of_previous_s_dm_tt'])
        # for a reward
        self.a_for_reward_history = deque(maxlen=self.config_env_rl['number_of_previous_a_for_reward'])
        self.a_for_reward_history_pzt = deque(maxlen=self.config_env_rl['number_of_previous_a_for_reward'])
        self.a_for_reward_history_tt = deque(maxlen=self.config_env_rl['number_of_previous_a_for_reward'])
        # keys
        self.list_of_keys_of_state = ['s_dm', 's_dm_residual', 's_dm_residual_rl', 'a_for_reward', 's_dm_residual_tt', 's_dm_tt']
        # Mask
        # Mask influence
        if config_env_rl['use_mask_influence_function']:
            from src.env_methods.projectors import get_mask_influence_function
            self.mask_saturation = get_mask_influence_function(self.supervisor, parameter_file, plot_influence=True)
            self.mask_saturation_inverse = np.where(self.mask_saturation != 0, np.divide(1, self.mask_saturation, where=self.mask_saturation != 0), self.mask_saturation)
        else:
            self.mask_saturation, self.mask_saturation_inverse = None, None
        # Observation/action space
        self.state_size_channel_0, self.observation_shape,\
            self.observation_space, self.action_space = self.define_state_action_space()

        if self.config_env_rl['running_norm']:
            from src.env_methods.helper import RunningMeanStd
            if self.config_env_rl['running_norm_mode'] == "scalar":
                self.norm_parameters = {"dm": RunningMeanStd(shape=(1,)),
                                        "dm_residual":RunningMeanStd(shape=(1,))
                                        }
                if self.config_env_rl['s_dm_tt'] or int(self.config_env_rl['number_of_previous_s_dm_tt']) > 0:
                    self.norm_parameters["dm_tt"] = RunningMeanStd(shape=(1,))
                if self.config_env_rl['s_dm_residual_tt'] or int(self.config_env_rl['number_of_previous_s_dm_residual_tt']) > 0 or \
                   self.config_env_rl['s_dm_residual_non_linear'] or int(self.config_env_rl['number_of_previous_s_dm_residual_non_linear_tt']) > 0:
                    self.norm_parameters["dm_residual_tt"] = RunningMeanStd(shape=(1,))
            elif self.config_env_rl['running_norm_mode'] == "array" or self.config_env_rl['running_norm_mode'] == "arrayv2":
                self.norm_parameters = {"dm": RunningMeanStd(shape=(40,40)),
                                        "dm_residual":RunningMeanStd(shape=(40,40))
                                        }
                if self.config_env_rl['s_dm_tt'] or int(self.config_env_rl['number_of_previous_s_dm_tt']) > 0:
                    self.norm_parameters["dm_tt"] = RunningMeanStd(shape=(40, 40))
                if self.config_env_rl['s_dm_residual_tt'] or int(self.config_env_rl['number_of_previous_s_dm_residual_tt']) > 0 or \
                   self.config_env_rl['s_dm_residual_non_linear'] or int(self.config_env_rl['number_of_previous_s_dm_residual_non_linear_tt']) > 0:
                    self.norm_parameters["dm_residual_tt"] = RunningMeanStd(shape=(40, 40))
            else:
                raise NotImplementedError
        else:
            # Normalization

            self.norm_parameters = {"dm":
                                        {"mean": 0.0,
                                         "std": self.config_env_rl['dm_std']},
                                    "dm_residual":
                                        {"mean": 0.0,
                                         "std": self.config_env_rl['dm_residual_std']}
                                    }
            if self.config_env_rl['s_dm_tt'] or int(self.config_env_rl['number_of_previous_s_dm_tt']) > 0:
                self.norm_parameters["dm_tt"] = {"mean": 0.0, "std": self.config_env_rl['dm_std']}
            if self.config_env_rl['s_dm_residual_tt'] or int(self.config_env_rl['number_of_previous_s_dm_residual_tt']) > 0 or \
               self.config_env_rl['s_dm_residual_non_linear'] or int(self.config_env_rl['number_of_previous_s_dm_residual_non_linear_tt']) > 0:
                self.norm_parameters["dm_residual_tt"] = {"mean": 0.0, "std": self.config_env_rl['dm_residual_std']}
                
        # Delayed assignment
        self.delayed_assignment = self.config_env_rl['delayed_assignment']
        # Defined for the state
        self.s_next_main = None
        # Some values for Unet
        self.override_generate_phase_projectors = override_generate_phase_projectors

        print("Parameter file {} Observation space {} Action space {} Compass device {}".format(
            parameter_file, self.observation_space.shape, self.action_space.shape, self.device))
        self.modes2phase, self.phase2modes = None, None
        self.modespzt2phase, self.phase2modespzt = None, None
        self.modestt2phase, self.phase2modestt = None, None
        self.modespzt2phase_torch, self.phase2modespzt_torch = None, None
        self.modestt2phase_torch, self.phase2modestt_torch = None, None
        if self.control_tt or self.override_generate_phase_projectors:
            self.modes2phase, self.phase2modes, \
            self.modespzt2phase, self.phase2modespzt, \
            self.modestt2phase, self.phase2modestt = self.create_projectors_of_phase(parameter_file)

            self.supervisor.modestt2phase = self.modestt2phase
            self.supervisor.phase2modespzt = self.phase2modespzt

            if self.device != -1:
                # add matrices to GPU

                self.modespzt2phase_torch, self.phase2modespzt_torch, \
                self.modestt2phase_torch, self.phase2modestt_torch = \
                    torch.FloatTensor(self.modespzt2phase).to(self.device),\
                    torch.FloatTensor(self.phase2modespzt).to(self.device), \
                    torch.FloatTensor(self.modestt2phase).to(self.device), \
                    torch.FloatTensor(self.phase2modestt).to(self.device)
        
            
        self.wfs_xpos, self.wfs_ypos = self.supervisor.config.p_wfs0._validsubsx, self.supervisor.config.p_wfs0._validsubsy
        if self.supervisor.config.p_controllers[0].get_type() != "geo":
            # Change reference slopes
            if config_compass.p_wfs0.type == 'pyrhr' and config_compass.p_wfs0.get_pyr_ampl() == 0:
                # Check if we are using pyramid with no modulation and without geometric controller to change reference slopes
                pass
                # self.change_reference_slopes()
        
            self.slopes_ref = self.supervisor.rtc.get_ref_slopes() # hardcoded for simulation
            self.linear_reconstructor = self.supervisor.rtc.get_command_matrix(0)
        else:
            self.slopes_ref, self.linear_reconstructor = [None] * 2
        # Wfs mask
        self.wfs_mask = np.zeros(self.supervisor.wfs.get_wfs_image(0).shape)
        self.wfs_mask[self.wfs_xpos, self.wfs_ypos] = 1
        self.normalization_noise_value_linear = normalization_noise_value_linear

        if self.control_tt:
            assert self.num_dm > 1, "You must have a TT DM if you want to control it"
        
       
    #           Basic environment
    #
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def change_reference_slopes(self, num_darks:int=10000):
        """
        COMPASS with unmodulated has a small error where the reference slopes are badly set
        This function, changes the reference slopes
        num_darks: number of wfs images used to average slope
        """
        print("Adjusting reference slopes for unmodulated pyramid")
        slopes_ct = 0
        zeroes = np.zeros(self.command_shape, np.float32)
        self.supervisor.reset()
        for i in range(num_darks):
            self.supervisor.rtc.set_command(controller_index=0, com=zeroes)
            self.supervisor.rtc.apply_control(0, comp_voltage=False)
            self.supervisor.target.raytrace(0, tel=self.supervisor.tel, dms=self.supervisor.dms)
            self.supervisor.wfs.raytrace(0, tel=self.supervisor.tel)
            self.supervisor.wfs.raytrace(0, dms=self.supervisor.dms, ncpa=False, reset=False)
            self.supervisor.wfs.compute_wfs_image(0)
            self.supervisor.rtc.compute_slopes(0)
            self.supervisor.rtc.do_control(0)
            self.supervisor.target.comp_tar_image(0)
            self.supervisor.target.comp_strehl(0)
            wfs_image = self.supervisor.wfs.get_wfs_image(0)
            slopes = wfs_image[self.wfs_xpos, self.wfs_ypos] / np.mean(wfs_image[self.wfs_xpos, self.wfs_ypos])
            slopes_ct += slopes
            if i < 10:
                print("slopes", slopes)
        print("slopes_ct", slopes_ct/num_darks)
        self.supervisor.rtc.set_ref_slopes(slopes_ct/num_darks)

    
    def create_projectors_of_phase(self, parameter_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates projectors to project between phase and mirrors
        :param parameter_file: parameter file used
        """
        projector_creator = ProjectorCreator(parameter_file,
                                             self.supervisor,
                                             second_dm_index=1,
                                             debug_targetphase2modes=False)
        if self.config_env_rl['basis_projectors'] == "btt":
            raise NotImplementedError
            modes2phase, phase2modes = projector_creator.get_projector_targetphase2modesphase()
            modespzt2phase = modes2phase[:, :-2].copy()
            modestt2phase = modes2phase[:, -2:].copy()
            phase2modespzt = phase2modes[:-2, :].copy()
            phase2modestt = phase2modes[-2:, :].copy()
        elif self.config_env_rl['basis_projectors'] == "actuators":
            # Note this one has small errors
            modes2phase, phase2modes, modespzt2phase, phase2modespzt, modestt2phase, phase2modestt = \
                projector_creator.get_projector_targetphase2actuator()
        else:
            raise NotImplementedError
        del projector_creator

        return modes2phase, phase2modes, modespzt2phase, phase2modespzt, modestt2phase, phase2modestt

    def define_state_action_space(self) -> Tuple[int, tuple, spaces.Box, spaces.Box]:
        """
        Defines state action space (num channels in state num channels in action and matrix shape)
        """

        state_size_channel_0 = int(self.config_env_rl['number_of_previous_s_dm']) +\
                               int(self.config_env_rl['number_of_previous_s_dm_residual']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_rl']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_tt']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_tt']) + \
                               int(self.config_env_rl['s_dm_residual_rl']) + \
                               int(self.config_env_rl['s_dm_residual']) + \
                               int(self.config_env_rl['s_dm']) + \
                               int(self.config_env_rl['s_dm_tt']) + \
                               int(self.config_env_rl['s_dm_residual_tt'])

        observation_shape = (state_size_channel_0,) + self.action_2d_shape
        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
        action_space = spaces.Box(low=-math.inf, high=math.inf, shape=self.action_2d_shape, dtype=np.float32)

        return state_size_channel_0, observation_shape, observation_space, action_space

    # For history
    def append_to_attr(self, attr, value):
        self.__dict__[attr].append(value)

    def clear_attr(self, attr):
        self.__dict__[attr].clear()

    def reset_history(self, attributes_list):
        idx = 0
        for attr in attributes_list:
            attr_history = attr + "_history"
            shape_of_history = np.zeros(self.action_2d_shape, dtype="float32")
            self.clear_attr(attr_history)
            rang = self.config_env_rl["number_of_previous_" + attr]
            for _ in range(rang):
                self.append_to_attr(attr_history, shape_of_history)
            idx += 1

    def reset(self,
              only_reset_dm: bool = False,
              return_dict: bool = False,
              add_one_to_seed: bool = True,
              noise: float = None) -> np.ndarray:
        """
        Resets simulator and some metrics related to RL
        :param only_reset_dm: if the reset is only on the DM
        :param return_dict: if we return a dictionary instead of a np.ndarray
        :param add_one_to_seed: if we add one to seed that manages simulation
        :param noise: if it is != None, the value of seed is changed
        """

        if add_one_to_seed:
            self.supervisor.add_one_to_seed()
            print("Resetting add one to seed", self.supervisor.current_seed)
        self.supervisor.reset(only_reset_dm, noise=noise)
        # Override step if first controller in param file is "geo":
        if self.supervisor.config.p_controllers[0].get_type() != "geo":
            # We reset state history and a_for_reward history
            self.reset_history(self.list_of_keys_of_state + ["a_for_reward"])
            if not only_reset_dm:
                self.supervisor.move_atmos_compute_wfs_reconstruction()

            self.build_state()
            s = self.get_next_state(return_dict=return_dict)
            return s
        else:
            s = np.array([0,])
            return s

    def reset_without_rl(self, only_reset_dm: bool=False, noise: float=None, add_one_to_seed: int=False):
        """
        Reset only Compass without resetting anything related to RL
        :param only_reset_dm: bool = False,
        :param add_one_to_seed: if we add one to seed that manages simulation
        :param noise: if it is != None, the value of seed is changed
        """
        if add_one_to_seed:
            self.supervisor.add_one_to_seed()
            print("Resetting add one to seed", self.supervisor.current_seed)
        self.supervisor.reset(only_reset_dm, noise)
        if not only_reset_dm:
            self.supervisor.move_atmos_compute_wfs_reconstruction()

    def standardise(self, inpt: np.ndarray, key: str, update_running_norm:bool=False):
        """
        standardises
        :param inpt: state to be normalized
        :param key: "wfs" or "dm"
        :return: input normalized
        """

        if self.config_env_rl['running_norm'] and update_running_norm:
            if self.config_env_rl['running_norm_mode'] == "scalar":
                reshaping = inpt[self.mask_valid_actuators==1].mean().reshape(1, 1)
                self.norm_parameters[key].update(reshaping)
                norm_inpt = (inpt - self.norm_parameters[key].mean) / np.sqrt(self.norm_parameters[key].var + 1e-8)
            else:
                inpt = np.expand_dims(inpt, axis=0)
                self.norm_parameters[key].update(inpt)
                if self.config_env_rl['running_norm_mode'] == "arrayv2":
                    inpt = inpt / np.sqrt(self.norm_parameters[key].var + 1e-8)
                else:
                    inpt = (inpt - self.norm_parameters[key].mean) / np.sqrt(self.norm_parameters[key].var + 1e-8)
                norm_inpt = inpt[0]
        elif self.config_env_rl['running_norm']:
            norm_inpt = (inpt - self.norm_parameters[key].mean) / np.sqrt(self.norm_parameters[key].var + 1e-8)
        else:
            mean = self.norm_parameters[key]['mean']
            std = self.norm_parameters[key]['std']
            norm_inpt = (inpt - mean) / std
        return norm_inpt

    def filter_actions(self, a: np.ndarray, exploratory: bool=False) -> np.ndarray:
        """
        Filter actions from RL
        :param a: action
        :param exploratory: if True the action comes from an integrator instead of RL and we do need to change operations slightly
        """

        if exploratory:
            # assert self.num_dm > 1
            if self.control_tt:
                self.a_tt_from_pzt = self.pzt2tt(a[:-2]) + a[-2:]
                self.a_pzt_from_tt = self.tt2pzt(self.a_tt_from_pzt)

            a = a[:-2]
            if self.config_env_rl['filter_state_actuator_space_with_btt']:
                a = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a,
                                                                                   add_tip_tilt_to_not_break=True)
            if self.control_tt:
                a = a + self.a_pzt_from_tt

            a = self.supervisor.apply_projector_volts1d_to_volts2d(a)
        elif self.control_tt and self.two_output_actor:
            a_pzt = a[0]
            self.a_pzt_from_tt = self.supervisor.apply_projector_volts2d_to_volts1d(a[1])
            self.a_tt_from_pzt = self.pzt2tt(self.a_pzt_from_tt)

            if self.config_env_rl['filter_state_actuator_space_with_btt']:
                a_pzt = self.supervisor.apply_projector_volts2d_to_volts1d(a_pzt)
                a_pzt = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a_pzt,
                                                                                       add_tip_tilt_to_not_break=True if self.num_dm > 1 else False)
                a = a.cpu().numpy()
                a[0] = self.supervisor.apply_projector_volts1d_to_volts2d(a_pzt)
            self.a_pzt_from_tt = self.a_pzt_from_tt.cpu().numpy()
        else:
            # 0) To 1D
            a = self.supervisor.apply_projector_volts2d_to_volts1d(a)
            if self.control_tt:
                self.a_tt_from_pzt = self.pzt2tt(a)
                self.a_pzt_from_tt = self.tt2pzt(self.a_tt_from_pzt)

            # 1) In case of actuator space filter with Btt if necessary
            if self.config_env_rl['filter_state_actuator_space_with_btt']:
                a = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a,
                                                                                   add_tip_tilt_to_not_break=True if self.num_dm > 1 else False)
                if self.control_tt:
                    a = a + self.a_pzt_from_tt

            a = self.supervisor.apply_projector_volts1d_to_volts2d(a)

        return a

    def get_phase_and_process_it(self) -> np.ndarray:
        """
        Get the phase and remove piston
        """
        phase = self.supervisor.target.get_tar_phase(0)
        phase = np.multiply(phase, self.s_pupil)
        phase[self.s_pupil == 1] -= phase[self.s_pupil == 1].mean()
        return phase

    def add_s_dm_info(self, s_next, s_dm_info, key_attr, key_norm):

        key_attribute_history = key_attr + "_history"
        current_history = getattr(self, key_attribute_history)
        for idx in range(len(current_history)):
            past_s_dm_info = getattr(self, key_attribute_history)[idx]
            past_s_dm_info = self.process_dm_state(past_s_dm_info, key=key_norm)
            s_next[key_attribute_history + "_" + str(len(current_history) - idx)] = \
                past_s_dm_info

        if self.config_env_rl["number_of_previous_" + key_attr] > 0:
            self.append_to_attr(key_attribute_history, s_dm_info)

        # 2 Add current residual to the state
        if self.config_env_rl[key_attr]:
            update_running_norm = True if (key_attr == "s_dm" or key_attr == "s_dm_residual_non_linear" or key_attr == "s_dm_residual_non_linear_tt" or  key_attr == "s_dm_residual" or key_attr == "s_dm_tt" or key_attr == "s_dm_residual_tt" or key_attr == "s_dm_residual_rl") else False
            s_dm = self.process_dm_state(s_dm_info, key=key_norm, update_running_norm=update_running_norm)
            s_next[key_attr] = s_dm.copy()
        return s_next

    def process_dm_state(self, s_dm, key, update_running_norm:bool=False):
        if self.config_env_rl['normalization_bool']:
            s_dm = self.standardise(s_dm, key=key, update_running_norm=update_running_norm)
        return s_dm

    def calculate_linear_reconstruction(self, slopes) -> None:
        """
        Calculates linear reconstruction
        :param slopes: pyramid pixels or sh-wfs slopes
        """
        linear_reconstruction = self.linear_reconstructor.dot(slopes)
        linear_reconstruction = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(linear_reconstruction)
        return linear_reconstruction

    def calculate_linear_residual(self) -> None:
        """
        Calculates linear reconstruction
        """
        slopes = self.get_processed_slopes()
        linear_reconstruction = self.supervisor.linear_gain * self.linear_reconstructor.dot(slopes)
        if self.config_env_rl['filter_state_actuator_space_with_btt']:
            c_linear = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(c_linear)
        return linear_reconstruction


    def calculate_reward(self, a: np.ndarray, a_pzt: np.ndarray, a_tt: np.ndarray) -> np.ndarray:
        """
        Calculates reward
        :param a: current action
        :param a_pzt: current action from pzt mirror in 1d
        :param a_tt: current action from tt mirror in 1d
        """
        linear_reconstruction = self.calculate_linear_residual()
        # For metrics
        self.reconstruction_for_reward = linear_reconstruction
        # Build reward
        processed_linear_rec = self.preprocess_dm_info(linear_reconstruction, sum_tt_projection=self.config_env_rl['joint_tt_into_reward'], normalise_for_reward=self.config_env_rl['normalise_for_reward'])
        # Squared and multiply by -1
        r2d_linear_rec = -np.square(processed_linear_rec)
        # Divide by number of actuators
        r2d_linear_rec /= r2d_linear_rec.reshape(-1).shape[0]
        # If we use two rewards, one for TT and one for PZT separate them
        if not self.config_env_rl['joint_tt_into_reward'] and self.control_tt:
            tt_reconstruction = self.config_env_rl['scaling_for_residual_tt'] * linear_reconstruction[-2:]
            pzt_reconstruction_from_tt_2d = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2pzt(tt_reconstruction))
            if self.config_env_rl['normalise_for_reward']:
                assert "dm_residual_tt" in self.norm_parameters
                pzt_reconstruction_from_tt_2d = self.standardise(pzt_reconstruction_from_tt_2d, key="dm_residual_tt")
            # Squared and multiply by -1
            r2d_pzt_reconstruction_from_tt = -np.square(pzt_reconstruction_from_tt_2d)
            r2d_linear_rec = np.stack([r2d_linear_rec, r2d_pzt_reconstruction_from_tt])

        # Scale
        r2d_linear_rec =  r2d_linear_rec * self.config_env_rl['reward_scale']

        if self.config_env_rl['reward_type'] == "scalar_actuators":
            return r2d_linear_rec.mean()
        elif self.config_env_rl['reward_type'] == "2d_actuators":
            return r2d_linear_rec
        else:
            raise NotImplementedError

    #
    #         Step methods
    # This works for both level = Correction and level = Gain
    #  __  __      _   _               _
    # |  \/  | ___| |_| |__   ___   __| |___
    # | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    # | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    @torch.no_grad()
    def pzt2tt(self, dm_pzt: np.ndarray) -> np.ndarray:
        """
        Maps pzt info to TT
        :param dm_pzt: info in pzt space
        """
        if self.config_env_rl['basis_projectors'] == "btt":
            raise NotImplementedError
            # If phase projects to modes change to modal basis
            dm_pzt = np.concatenate([dm_pzt, [0,0]])
            dm_pzt = self.supervisor.env.supervisor.volts2modes.dot(dm_pzt)

        # self.pzt_shape to phase <- this should already by filtered by Btt
        if self.device > -1:
            if not torch.is_tensor(dm_pzt):
                dm_pzt = torch.FloatTensor(dm_pzt).to(self.device)
            phase = self.modespzt2phase_torch @ dm_pzt
        else:
            phase = self.modespzt2phase.dot(dm_pzt)
        # remove piston
        phase = phase - phase.mean()
        # phase to self.tt_shape
        if self.device > -1:
            dm_tt_from_pzt = (self.phase2modestt_torch @ phase).cpu().numpy()
        else:
            dm_tt_from_pzt = self.phase2modestt.dot(phase)

        if self.config_env_rl['basis_projectors'] == "btt":
            raise NotImplementedError
            # If phase projects to modes change to modal basis and remove TT
            dm_tt_from_pzt = self.supervisor.env.supervisor.modes2volts.dot(dm_tt_from_pzt)[:-2]
        return dm_tt_from_pzt

    @torch.no_grad()
    def tt2pzt(self, dm_tt: np.ndarray) -> np.ndarray:
        """
        Maps pzt info to TT
        :param dm_pzt: info in pzt space
        """
        # On the contrary to pzt2tt, tt have the same values on modal space or actuator space

        # self.tt_shape to phase shape
        if self.device > -1:
            dm_tt_torch = torch.FloatTensor(dm_tt).to(self.device)
            phase = self.modestt2phase_torch @ dm_tt_torch
        else:
            phase = self.modestt2phase.dot(dm_tt)
        # remove piston
        phase = phase - phase.mean()
        # phase to self.tt_shape
        if self.device > -1:
            dm_pzt_from_tt = (self.phase2modespzt_torch @ phase).cpu().numpy()
        else:
            dm_pzt_from_tt = self.phase2modespzt.dot(phase)

        return dm_pzt_from_tt

    def preprocess_dm_info(self, s_dm_info: np.ndarray, sum_tt_projection: bool=False, normalise_for_reward: bool = False) -> np.ndarray:
        """
        Preprocess information about the DM either command or reconstruction
        Args:
            s_dm_info: current s_dm_info in vector form
            sum_tt_projection: Do we sum tt projection even if we are controlling tt
            normalise_for_reward: if a normalisation procedure is conducted (only for reward)

        Returns: s_dm_info after processing matrix form
        """
        if self.num_dm == 2:
            s_pzt_info = s_dm_info[:-2]
            if self.control_tt and sum_tt_projection:
                # Case in which PZT is summed to TT component projected to PZT
                s_tt = self.config_env_rl['scaling_for_residual_tt'] * s_dm_info[-2:]
                s_pzt_from_tt = self.tt2pzt(s_tt)
                # If normalise for reward and we have two sets of norm params "dm_residual" and "dm_residual_tt"
                if normalise_for_reward and "dm_residual_tt" in self.norm_parameters.keys():
                    s_pzt_info_2d = self.standardise(self.supervisor.apply_projector_volts1d_to_volts2d(s_pzt_info), key="dm_residual")
                    s_pzt_from_tt_2d = self.standardise(self.supervisor.apply_projector_volts1d_to_volts2d(s_pzt_from_tt), key="dm_residual_tt")
                    s_dm = s_pzt_info_2d + s_pzt_from_tt_2d
                else:
                    # Else we do not need to standardise with two sets of parameters
                    s_dm_1d = s_pzt_info + s_pzt_from_tt
                    s_dm = self.supervisor.apply_projector_volts1d_to_volts2d(s_dm_1d)
                    if normalise_for_reward:
                        s_dm = self.standardise(s_dm)
            else:
                # Else we ignore the tt component
                # For the case of that we control_tt and there is tt state, it will created in another method
                # For the case of that we do not control_tt, we don't need much else
                s_dm = self.supervisor.apply_projector_volts1d_to_volts2d(s_pzt_info)
        else:
            raise NotImplementedError

        if self.mask_saturation is not None:
            s_dm *= self.mask_saturation

        return s_dm

    def build_state(self):
        self.build_state_simple()

    def build_state_simple(self) -> None:
        """
        Builds state based on current rec, past commands, past recs and others depending on config
        The state is saved on self.s_next_main
        """
        s_next = OrderedDict()

        # TT state
        if self.config_env_rl['s_dm_residual_tt']:
            linear_rec = self.calculate_linear_residual()
            s_tt = self.config_env_rl['scaling_for_residual_tt'] * linear_rec[-2:]
            s_pzt_from_tt = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2pzt(s_tt))
            s_next = self.add_s_dm_info(s_next, s_pzt_from_tt, key_attr="s_dm_residual_tt", key_norm="dm_residual_tt")

        if self.config_env_rl['number_of_previous_s_dm_tt'] > 0 or self.config_env_rl['s_dm_tt']:
            s_tt = self.supervisor.past_command[-2:]
            s_pzt_from_tt = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2pzt(s_tt))
            s_next = self.add_s_dm_info(s_next, s_pzt_from_tt, key_attr="s_dm_tt", key_norm="dm_tt")

        # PZT state or TT + PZT state
        if self.config_env_rl['number_of_previous_s_dm'] > 0 or self.config_env_rl['s_dm']:
            s_dm = self.preprocess_dm_info(self.supervisor.past_command,
                                           sum_tt_projection=self.config_env_rl['joint_tt_into_s_dm'])
            s_next = self.add_s_dm_info(s_next, s_dm, key_attr="s_dm", key_norm="dm")

        if self.config_env_rl['s_dm_residual_rl']:
            # Actions work a bit different that other dm info
            if self.control_tt:
                past_a = self.preprocess_dm_info(self.supervisor.past_action_rl)
            else:
                past_a = self.supervisor.apply_projector_volts1d_to_volts2d(self.supervisor.past_action_rl)
            s_next = self.add_s_dm_info(s_next, past_a, key_attr="s_dm_residual_rl", key_norm="dm_residual")

        if self.config_env_rl['s_dm_residual']:
            s_dm_residual = self.preprocess_dm_info(self.calculate_linear_residual(),
                                                    sum_tt_projection=self.config_env_rl['joint_tt_into_s_dm_residual'])
            s_next = self.add_s_dm_info(s_next, s_dm_residual, key_attr="s_dm_residual", key_norm="dm_residual")

        self.s_next_main = s_next

    def get_next_state(self, return_dict: bool) -> np.ndarray:
        """
        Gets state saved on self.s_next_main
        :param return_dict: if True, the state is returned as a dict instead as a np.ndarray
        """
        if return_dict:
            return self.s_next_main
        else:
            return np.stack(np.array(list(self.s_next_main.values())))

    def step_process_rl_action(self, a: np.ndarray) -> None:
        """
        Given RL action, we process it by multiplying by scale and filtering
        :param a: np.ndarray
        """
        if self.two_output_actor:
            a = a[0]
        a_2d = a * self.config_env_rl['action_scale']
        a_1d = self.supervisor.apply_projector_volts2d_to_volts1d(a_2d)
        if self.control_tt:
            self.a_pzt_from_tt *= self.config_env_rl['action_scale']
            self.a_tt_from_pzt *= self.config_env_rl['action_scale']

            # TODO I think filtering here is not necessary
            self.a_pzt = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(a_1d - self.a_pzt_from_tt,
                                                                                        add_tip_tilt_to_not_break=True)

            a_1d = np.concatenate([self.a_pzt, self.a_tt_from_pzt])

        return a_1d, a_2d

    def step(self, a: np.ndarray, controller_type: str) -> None:
        """
        Step in the COMPASS environment
        :param a: action from RL. If controller_type is not "RL", "Phasev2", "UNet+Linearv2" it is not used.
        :param controller_type: "RL", "Integrator", "UNet+Linear", "Phase", "Phasev2", "UNet+Linearv2"
        """

        # Override step if first controller in param file is "geo":
        if self.supervisor.config.p_controllers[0].get_type() == "geo":
            self.supervisor.geom_next()
            r = 0
            s = np.array([0,])
            return s, r, False, None
        a_2d = None
        if controller_type == "RL":
            # 1) Rescale actions and transform them into 1D
            a, a_2d = self.step_process_rl_action(a)
        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(a,
                                                   controller_type=controller_type)

        r = self.calculate_reward(a_2d, a_pzt=self.a_pzt, a_tt=self.a_pzt_from_tt)
        sr_se, sr_le, _, _ = self.supervisor.target.get_strehl(0)
        info = {"sr_se": sr_se,
                "sr_le": sr_le}
        # 3) Move atmos, compute WFS and reconstruction
        self.supervisor.move_atmos_compute_wfs_reconstruction()
        # 4) Build state
        self.build_state()
        s = self.get_next_state(return_dict=False)

        if self.supervisor.iter % (self.config_env_rl['reset_strehl_every_and_print'] + 1) == 0 and\
                self.supervisor.iter > 1:
            self.supervisor.target.reset_strehl(0)

        return s, r, False, info

    def get_processed_slopes(self, wfs_index:int = 0, rtc_index:int=0) -> np.ndarray:
        """
        Gets current slopes which can be processed if normalization_noise_value_linear >= 0
        """
        if self.normalization_noise_value_linear >= 0:
            wfs_image = self.supervisor.wfs.get_wfs_image(wfs_index) - self.normalization_noise_value_linear
            wfs_image[wfs_image < 0] = 0
            slopes = wfs_image[self.wfs_xpos, self.wfs_ypos] / np.mean(
                wfs_image[self.wfs_xpos, self.wfs_ypos]) - self.slopes_ref
        else:
            slopes = self.supervisor.rtc.get_slopes(rtc_index)

        return slopes

    def step_only_linear(self) -> None:
        """
        Does a step in compass with Integrator controller with linear reconstruction
        """
        self.supervisor.move_dm_and_compute_strehl(None, controller_type="Integrator")
        self.supervisor.move_atmos_compute_wfs_reconstruction()

    def get_modal_projection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.supervisor.modes2volts, self.supervisor.volts2modes
