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
import matplotlib.pyplot as plt
from src.env_methods.debugging import save_prediction
from src.global_cte import FOLDER_CHECKPOINTS_UNET

class AoEnvNonLinear(AoEnv):
    def __init__(self,
                 unet_dir: str,
                 unet_name: str,
                 unet_type: str,
                 only_predict_non_linear: bool,
                 device_unet: int,
                 gain_factor_unet: float,
                 gain_factor_linear:float=0,
                 normalize_flux:bool =False,
                 normalization_095_005:bool=True,
                 config_env_rl:dict=None,
                 parameter_file:str=None,
                 seed:int=None,
                 device_compass:int=None,
                 normalization_noise_unet:bool=True,
                 normalization_noise_value_unet:float=3,
                 use_wfs_mask_unet:bool=False,
                 normalization_noise_value_linear:float=-1,
                 noise:float=None):
        """
        AoEnvNonLinear - interface to talk with compass and generate with some additionalities
        In the non linear version, UNet is available
        :param unet_dir: Path to UNet dir
        :param unet_name: UNet name
        :param unet_type: UNet type
        :param only_predict_non_linear: if UNet only predict only non-linear rec.
        :param device_unet: device used for UNet
        :param gain_factor_unet: factor of gain for UNet rec.
        :param gain_factor_linear: factor of gain for linear rec.
        :param normalize_flux: if flux is normalized for UNet
        :param normalization_095_005: if we do the normalization 0.95-0.05
        :param config_env_rl: configuration for RL exps
        :param parameter_file: parameter file used
        :param seed: current seed
        :param device_compass: device used for COMPASS
        :param normalization_noise_unet: if we do remove expected RMS noise
        :param normalization_noise_value_unet: value to subtract if we do normalization_noise_unet
        :param use_wfs_mask_unet: if a wfs is used to remove values outside the WFS valid zone
        :param normalization_noise_value_linear: if >= 0 we clipd values <0 from WFS image for linear rec.
        :param noise: if True the noise from the WFS in the param file is changed
        """

        self.device_unet = device_unet
        self.normalize_flux = normalize_flux
        self.unet_type = unet_type
        self.unet_name = unet_name
        self.unet_dir = unet_dir
        self.only_predict_non_linear = only_predict_non_linear
        self.normalization_noise_unet = normalization_noise_unet
        self.normalization_noise_value_unet = normalization_noise_value_unet
        self.combination_reconstruction, self.lin_reconstruction, self.non_lin_reconstruction = None, None, None
        self.normalization_095_005 = normalization_095_005
        # We save the non-linear rec as a variable so we only have to predict it once
        if self.only_predict_non_linear:
            assert unet_type == "volts", "Unet type must be volts if only_predict_non_linear is True"
        self.gain_factor_unet = gain_factor_unet
        self.gain_factor_linear = gain_factor_linear
        self.no_subtract_mean_from_phase = config_env_rl['no_subtract_mean_from_phase']
        self.use_wfs_mask_unet = use_wfs_mask_unet
        print("----------------")
        print("-- Unet Model --")
        print("U-Net name: ", unet_name)
        print("U-Net dir: ", unet_dir)
        print("U-Net type: ", unet_type)
        print("U-Net no_subtract_mean_from_phase", self.no_subtract_mean_from_phase)
        print("U-Net use_wfs_mask_unet", self.use_wfs_mask_unet)
        print("U-Net normalization_noise_unet", self.normalization_noise_unet)
        print("U-Net normalization_noise_value_unet", self.normalization_noise_value_unet)
        self.model, unet_full_dir = self.load_unet(unet_dir, unet_name, unet_type)

        super(AoEnvNonLinear, self).__init__(config_env_rl,
                                             parameter_file,
                                             seed,
                                             pyr_gpu_ids=device_compass,
                                             override_generate_phase_projectors=True,
                                             normalization_noise_value_linear=normalization_noise_value_linear,
                                             noise=noise)

        self.min_wfs_image, self.scale_wfs, \
        self.value_left_up_1, self.value_left_up_2, self.value_right_down_1, self.value_right_down_2, \
        self.min_phase, self.scale_phase, self.out_mask, self.wfs_image_pad_value, self.out_unpad_value = \
            self.prepare_values_norm_pad(unet_full_dir, unet_type, normalize_flux, normalization_095_005)

        self.s_dm_residual_non_linear_history = \
            deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_non_linear'])
        self.s_dm_residual_non_linear_tt_history = \
            deque(maxlen=self.config_env_rl['number_of_previous_s_dm_residual_non_linear_tt'])

        self.list_of_keys_of_state = ['s_dm', 's_dm_residual', 's_dm_residual_rl', 'a_for_reward', 's_dm_residual_tt',
                                      's_dm_tt', 's_dm_residual_non_linear', 's_dm_residual_non_linear_tt']


    def update_unet_params(self, unet_dir:str, unet_name: str, unet_type: str, normalization_noise_unet: bool):
        """
        Updates parameters of UNet
        :param unet_dir: path to directory
        :param unet_name: path to UNet
        :param unet_type: type of UNet
        :param normalization_noise_unet: if we do remove expected RMS noise
        """
        self.unet_type = unet_type
        self.unet_name = unet_name
        self.unet_dir = unet_dir

        self.model, unet_full_dir = self.load_unet(unet_dir, unet_name, unet_type)

        self.normalization_noise_unet = True

        self.min_wfs_image, self.scale_wfs, \
        self.value_left_up_1, self.value_left_up_2, self.value_right_down_1, self.value_right_down_2, \
        self.min_phase, self.scale_phase, self.out_mask, self.wfs_image_pad_value, self.out_unpad_value = \
            self.prepare_values_norm_pad(unet_full_dir, unet_type, self.normalize_flux, self.normalization_095_005)


    def define_state_action_space(self):
        """
        Defines action space for env non-linear
        """
        state_size_channel_0 = int(self.config_env_rl['number_of_previous_s_dm']) +\
                               int(self.config_env_rl['number_of_previous_s_dm_residual']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_rl']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_tt']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_tt']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_non_linear']) + \
                               int(self.config_env_rl['number_of_previous_s_dm_residual_non_linear_tt']) + \
                               int(self.config_env_rl['s_dm_residual_rl']) + \
                               int(self.config_env_rl['s_dm_residual']) + \
                               int(self.config_env_rl['s_dm']) + \
                               int(self.config_env_rl['s_dm_tt']) + \
                               int(self.config_env_rl['s_dm_residual_tt']) + \
                               int(self.config_env_rl['s_dm_residual_non_linear']) + \
                               int(self.config_env_rl['s_dm_residual_non_linear_tt'])

        observation_shape = (state_size_channel_0,) + self.action_2d_shape
        observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=observation_shape, dtype=np.float32)
        action_space = spaces.Box(low=-math.inf, high=math.inf, shape=self.action_2d_shape, dtype=np.float32)

        return state_size_channel_0, observation_shape, observation_space, action_space

    def load_unet(self, unet_dir: str, unet_name: str, unet_type: str) -> (UnetGenerator, str):
        """
        Loads UNet model into this class
        :param unet_dir: path to directory
        :param unet_name: path to UNet
        :param unet_type: type of UNet
        """

        unet_full_dir = os.path.join(FOLDER_CHECKPOINTS_UNET, unet_dir)
        unet_full_path = os.path.join(unet_full_dir, unet_name)
        if unet_type == "phase":
            model = UnetGenerator(input_nc=4, output_nc=1, num_downs=9, ngf=64).to(self.device_unet)
        elif unet_type == "volts":
            model = UnetGenerator(input_nc=4, output_nc=1, num_downs=6, ngf=64).to(self.device_unet)
        else:
            raise NotImplementedError
        state_dict = torch.load(unet_full_path, map_location=self.device_unet)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' from key
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        return model, unet_full_dir

    def prepare_values_norm_pad(self, unet_full_dir, unet_type, normalize_flux, normalization_095_005) -> np.ndarray:
        """
        Hardcoded values for current simulation
        :param unet_full_dir: path to UNet
        :param unet_type: Type of UNet
        :param normalize_flux: if we do normalise flux (in paper experiments this is not used)
        :param normalization_095_005: if we put the bounds of max and min to 0.05 to 0.95 instead 0 and 1 for max-min norm
        """
        num_pix_wfs = 56  # full size 256, divide by 2 128, we have 56 pix in the par file
        # for padding wfs
        edge_offset_wfs = 44
        center_offset_wfs = 56
        value_left_up_1 = edge_offset_wfs - 4
        value_left_up_2 = edge_offset_wfs + num_pix_wfs + 4
        value_right_down_1 = edge_offset_wfs + num_pix_wfs + center_offset_wfs - 4
        value_right_down_2 = -edge_offset_wfs + 4
        if normalize_flux:
            df_norm = pd.read_csv(unet_full_dir + "/info_normalize_flux.csv")
        else:
            df_norm = pd.read_csv(unet_full_dir + "/info.csv")
        if unet_type == "volts":
            if self.only_predict_non_linear:
                min_phase = df_norm['Min voltage linear subtracted'].values[0]
                max_phase = df_norm['Max voltage linear subtracted'].values[0]
            else:
                min_phase = df_norm['Min voltage'].values[0]
                max_phase = df_norm['Max voltage'].values[0]
            num_pix_phase = 40  # 40 because its commands
            unet_size = 64  # 128 max(wfs,phase)
            out_unpad_value = int((unet_size - num_pix_phase) / 2.0)
            out_mask = self.supervisor.mask_valid_actuators
            wfs_image_pad_value = 0
        elif unet_type == "phase":
            min_phase = df_norm['Min phase'].values[0]
            max_phase = df_norm['Max phase'].values[0]
            num_pix_phase = 448  # 448
            unet_size = 512  # 512
            out_mask = self.supervisor.get_s_pupil()
            wfs_image_pad_value = int(unet_size / 2) - 32  # 192
            out_unpad_value = int((unet_size - num_pix_phase)/2.0)
        else:
            raise NotImplementedError

        if normalization_095_005:
            scale_phase = (max_phase - min_phase) / 0.9
            min_phase = min_phase - 0.05 * scale_phase
        min_wfs_image = df_norm['Min wfs'].values[0]
        max_wfs_image = df_norm['Max wfs'].values[0]
        if self.normalization_noise_unet:
            min_wfs_image = 0
        scale_wfs = (max_wfs_image - min_wfs_image)

        scale_phase = (max_phase - min_phase)
        return min_wfs_image, scale_wfs, value_left_up_1, value_left_up_2, value_right_down_1, value_right_down_2,\
               min_phase, scale_phase, out_mask, wfs_image_pad_value, out_unpad_value

    def pad_expand(self, wfs_image: np.ndarray) -> np.ndarray:
        """
        Pads wfs image and adds one axis
        :param wfs_image: current wfs image
        """
        if self.wfs_image_pad_value > 0:
            wfs_image = np.pad(wfs_image, ((self.wfs_image_pad_value, self.wfs_image_pad_value),
                                           (self.wfs_image_pad_value, self.wfs_image_pad_value)), 'constant')
        wfs_image = np.expand_dims(wfs_image, axis=0)
        return wfs_image

    def prepare_wfs_image(self, wfs_image: np.ndarray) -> np.ndarray:
        """
        Prepares WFS image for UNet inference
        Rearranges, subtracts noise and pads
        :param wfs_image: current wfs image
        :return:
        """

        if self.normalization_noise_unet:
            # Things below normalization_noise_value_unet... maybe forget about them?
            wfs_image = wfs_image - self.normalization_noise_value_unet
            wfs_image[wfs_image < 0] = 0

        if self.use_wfs_mask_unet:
            wfs_image *= self.wfs_mask

        if self.normalize_flux:
            wfs_image /= wfs_image.sum()


        # Remove extra dimensions
        wfs_image = np.squeeze(wfs_image)
        wfs_channel_1 = wfs_image[self.value_left_up_1:self.value_left_up_2,
                                  self.value_left_up_1:self.value_left_up_2]
        # Lower left
        wfs_channel_2 = wfs_image[self.value_left_up_1:self.value_left_up_2,
                                  self.value_right_down_1:self.value_right_down_2]
        # Upper right
        wfs_channel_3 = wfs_image[self.value_right_down_1:self.value_right_down_2,
                                  self.value_left_up_1:self.value_left_up_2]
        # Lower right
        wfs_channel_4 = wfs_image[self.value_right_down_1:self.value_right_down_2,
                                  self.value_right_down_1:self.value_right_down_2]

        # Pad, important before normalization because we are not centered at 0, we are centered at min_wfs_image
        wfs_channel_1 = self.pad_expand(wfs_channel_1)
        wfs_channel_2 = self.pad_expand(wfs_channel_2)
        wfs_channel_3 = self.pad_expand(wfs_channel_3)
        wfs_channel_4 = self.pad_expand(wfs_channel_4)
        # Concatenate
        wfs_image_multiple_channels = np.concatenate([wfs_channel_1, wfs_channel_2, wfs_channel_3, wfs_channel_4],
                                                     axis=0)
        wfs_image_multiple_channels_norm = (wfs_image_multiple_channels - self.min_wfs_image) / self.scale_wfs

        wfs_image_multiple_channels_norm = np.expand_dims(wfs_image_multiple_channels_norm, axis=0)

        # To torch
        wfs_image_multiple_channels_norm_torch =\
            torch.FloatTensor(wfs_image_multiple_channels_norm).to(self.device_unet)

        return wfs_image_multiple_channels_norm_torch

    def process_output(self, out: np.ndarray):
        """
        Unpads/denormalises and multiplies by mask
        :params: out: output of UNet unfiltered
        """
        # 1) To numpy
        out = out.cpu().numpy()

        # 2) Unpad
        out = out[0, 0, self.out_unpad_value:-self.out_unpad_value, self.out_unpad_value:-self.out_unpad_value]

        # 3) Denormalize
        out = out * self.scale_phase  # + self.min_phase - We remove the min from the concept

        # 4) Multiply mask
        out = np.multiply(out, self.out_mask)

        # 5) Remove mean
        if self.no_subtract_mean_from_phase:
            pass
        else:
            out[self.out_mask == 1] -= out[self.out_mask == 1].mean()
        return out

    def process_with_projectors(self, out, no_filter) -> np.ndarray:
        """
        Process output of UNet with projectors
        We need to extract TT and PZT components
        :param out: output of UNet
        :param no_filter: if we do not filter with Btt.T(Btt)
        """

        # 7) Project to volts (only phase) or to 1D (only volts)
        if self.unet_type == "phase":
            out_final = self.phase2modes.dot(out[self.out_mask == 1])
            if no_filter:
                pass
            else:
                out_final = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(out_final)
        elif self.unet_type == "volts":
            out_1d = self.supervisor.apply_projector_volts2d_to_volts1d(out)
            if no_filter:
                out_final = np.concatenate([out_1d, [0, 0]])
            else:
                out_pzt = self.supervisor.filter_dm_info_actuator_space_with_modal_basis(out_1d,
                                                                                         add_tip_tilt_to_not_break=True)
                out_tt = self.pzt2tt(out_1d)
                out_final = np.concatenate([out_pzt, out_tt])
        else:
            raise NotImplementedError

        return out_final

    @torch.no_grad()
    def inferr(self, wfs_image, no_filter=False) -> np.ndarray:
        """
        From wfs_image predict with UNet the phase predicted into the DM
        :param wfs_image: np.ndarray
        :param no_filter: if we DO not filter after output with Btt.T.(Btt)
        """
        wfs_image_processed = self.prepare_wfs_image(wfs_image)
        out = self.model(wfs_image_processed)
        processed_output = self.process_output(out)
        processed_output_final = self.process_with_projectors(processed_output, no_filter)
        return processed_output_final

    def add_linear_reconstruction(self, out, slopes):
        linear_reconstruction = self.calculate_linear_reconstruction(slopes)
        return out + linear_reconstruction

    def step_only_unet(self, tt_linear=False, i_save_prediction=None):
        """
        Step in the environment with UNet
        :param tt_linear: if we use linear rec. for TT
        :param i_save_prediction: if we save the prediction for debugging
        """
        # 1)  Non-linear
        wfs_image = self.supervisor.wfs.get_wfs_image(0)
        non_linear_reconstruction = - self.inferr(wfs_image)  # negative because we need inverse prediction

        if self.only_predict_non_linear:
            slopes = self.get_processed_slopes()
            non_linear_reconstruction = self.add_linear_reconstruction(non_linear_reconstruction, slopes)

        non_linear_reconstruction_with_gain = self.gain_factor_unet * non_linear_reconstruction
        if tt_linear:
            linear_reconstruction = self.supervisor.rtc.get_err(0)
            non_linear_reconstruction_with_gain[-2:] = linear_reconstruction[-2:]
        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(action=non_linear_reconstruction_with_gain, controller_type="UNet")

        # Debugging
        # -------------------------------------------------------------------------------------------------------------
        if i_save_prediction is not None:
            slopes = self.get_processed_slopes()
            linear_reconstruction = - self.calculate_linear_reconstruction(slopes)
            tt_nl = self.tt2pzt(non_linear_reconstruction[-2:])
            pzt_nl = non_linear_reconstruction[:-2]
            non_linear_reconstruction_2d = \
                self.supervisor.apply_projector_volts1d_to_volts2d(pzt_nl + tt_nl)

            real_phase = self.supervisor.target.get_tar_phase(0)
            real_phase_on_pupil = real_phase[np.where(self.supervisor.get_s_pupil())]
            real = self.phase2modes.dot(real_phase_on_pupil)
            tt_real = self.tt2pzt(real[-2:])
            pzt_real = real[:-2]
            real_2d = \
                self.supervisor.apply_projector_volts1d_to_volts2d(pzt_real + tt_real)

            tt_linear = self.tt2pzt(linear_reconstruction[-2:])
            pzt_linear = linear_reconstruction[:-2]
            linear_reconstruction_2d = \
                self.supervisor.apply_projector_volts1d_to_volts2d(pzt_linear + tt_linear)

            non_linear_reconstruction_phase_on_pupil = self.modes2phase.dot(non_linear_reconstruction)
            linear_reconstruction_phase_on_pupil = self.modes2phase.dot(linear_reconstruction)

            real_phase_tt_on_pupil = self.modestt2phase.dot(real[-2:])

            non_linear_reconstruction_phase_tt_on_pupil = self.modestt2phase.dot(non_linear_reconstruction[-2:])
            linear_reconstruction_phase_tt_on_pupil = self.modestt2phase.dot(linear_reconstruction[-2:])
            save_prediction(real_2d,
                            linear_reconstruction_2d,
                            non_linear_reconstruction_2d,
                            self.mask_valid_actuators,
                            real_phase_on_pupil,
                            non_linear_reconstruction_phase_on_pupil,
                            linear_reconstruction_phase_on_pupil,
                            real_phase_tt_on_pupil,
                            non_linear_reconstruction_phase_tt_on_pupil,
                            linear_reconstruction_phase_tt_on_pupil,
                            self.supervisor.get_s_pupil(),
                            self.supervisor.target.get_tar_phase(0).shape,
                            i_save_prediction)
        # -------------------------------------------------------------------------------------------------------------
        # 3) Compute new wfs image
        self.supervisor.move_atmos_compute_wfs_reconstruction()

    def step_only_combined_with_linear(self) -> None:
        """
        Step in the environment of UNet + Lin
        """
        assert self.gain_factor_linear is not None
        reconstruction = self.calculate_non_linear_residual()
        # 1) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(action=reconstruction, controller_type="UNet+Linear")
        # 2) Compute new wfs image and linear reconstruction
        self.supervisor.move_atmos_compute_wfs_reconstruction()

    def calculate_non_linear_residual(self, update_non_linear_reconstruction_variable=False, tt_linear=False) -> np.ndarray:
        """
        Calculates non_linear reconstruction
        :param update_non_linear_reconstruction_variable: if we update the class variable that keeps the non_linear_rec.
        """
        # 1) Non-linear
        wfs_image = self.supervisor.wfs.get_wfs_image(0)
        non_linear_reconstruction = - self.inferr(wfs_image)  # negative because we need inverse prediction

        # 2) linear
        slopes = self.get_processed_slopes()
        linear_reconstruction = - self.calculate_linear_reconstruction(slopes)  # negative because we need inverse prediction
        if self.only_predict_non_linear:
            # TODO is this the correct sign?
            non_linear_reconstruction = non_linear_reconstruction + linear_reconstruction

        if tt_linear:
            linear_reconstruction = self.supervisor.rtc.get_err(0)
            non_linear_reconstruction_with_gain[-2:] = linear_reconstruction[-2:]

        # 3) Sum
        non_linear_reconstruction_with_gain = self.gain_factor_unet * non_linear_reconstruction
        linear_reconstruction_with_gain = self.gain_factor_linear * linear_reconstruction
        reconstruction = non_linear_reconstruction_with_gain + linear_reconstruction_with_gain
        if update_non_linear_reconstruction_variable:
            self.lin_reconstruction = non_linear_reconstruction_with_gain
            self.non_lin_reconstruction = linear_reconstruction_with_gain
            self.combination_reconstruction = reconstruction

        return reconstruction

    def get_reconstruction(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Returns all reconstructions
        """
        return self.lin_reconstruction, self.non_lin_reconstruction, self.combination_reconstruction

    def build_state(self) -> None:
        """
        Builds state with non-linear rec.
        """

        self.build_state_simple() # Original build state
        if self.config_env_rl['s_dm_residual_non_linear'] or self.config_env_rl['s_dm_residual_non_linear_tt']:
            non_linear_rec = self.calculate_non_linear_residual(update_non_linear_reconstruction_variable=True)
            if self.config_env_rl['s_dm_residual_non_linear']:
                s_non_linear = \
                    self.preprocess_dm_info(non_linear_rec,
                                            sum_tt_projection=self.config_env_rl['joint_tt_into_s_dm_residual_non_linear'])
                self.s_next_main = self.add_s_dm_info(self.s_next_main, s_non_linear,
                                                      key_attr="s_dm_residual_non_linear", key_norm="dm_residual")

            if self.config_env_rl['s_dm_residual_non_linear_tt']:
                s_non_linear_tt = self.config_env_rl['scaling_for_residual_tt'] * non_linear_rec[-2:]
                s_non_linear_pzt_from_tt = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2pzt(s_non_linear_tt))
                self.s_next_main = self.add_s_dm_info(self.s_next_main, s_non_linear_pzt_from_tt,
                                                      key_attr="s_dm_residual_non_linear_tt", key_norm="dm_residual_tt")

    def step(self, a, controller_type) -> (np.ndarray, np.ndarray, bool, dict):
        """
        Does an step in the environment. This format is typicall in RL environments.
        :param a: Action
        :param controller_type: "RL", "UNet+Linear", "UNet+Linearv2", "Linear", "Phase"
        """
        a_2d = None
        if controller_type == "RL":
            # 1) Rescale actions and transform them into 1D
            a, a_2d = self.step_process_rl_action(a)
        elif controller_type == "UNet+Linear" or controller_type == "UNet+Linearv2":
            a = self.combination_reconstruction

        unet_command = self.combination_reconstruction
        if self.config_env_rl['correction_pzt_only_rl_tt']:
            unet_command[-2:] = 0

        # 2) Move DM and compute strehl
        self.supervisor.move_dm_and_compute_strehl(a,
                                                   controller_type=controller_type,
                                                   unet_or_phase_command=unet_command)

        r = self.calculate_reward(a_2d, self.a_pzt, self.a_pzt_from_tt)
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

    def calculate_reward(self, a, a_pzt, a_tt) -> np.ndarray:
        """
        Computes the reward
        :param a: action array
        :param a_pzt: pzt component of the action
        :param a_tt: tt component of the action
        """
        non_linear_reconstruction = self.combination_reconstruction
        # For metrics - todo perhaps can be removed
        self.reconstruction_for_reward = non_linear_reconstruction
        # Build reward and normalise if necessary
        processed_non_linear_rec = self.preprocess_dm_info(non_linear_reconstruction, sum_tt_projection=self.config_env_rl['joint_tt_into_reward'], normalise_for_reward=self.config_env_rl['normalise_for_reward'])
        # Squared and multiply by -1
        r2d_non_linear_rec = -np.square(processed_non_linear_rec)
        # Divide by number of actuators
        r2d_non_linear_rec /= r2d_non_linear_rec.reshape(-1).shape[0]
        # If we use two rewards, one for TT and one for PZT separate them
        if not self.config_env_rl['joint_tt_into_reward'] and self.control_tt:
            tt_reconstruction = self.config_env_rl['scaling_for_residual_tt'] * non_linear_reconstruction[-2:]
            pzt_reconstruction_from_tt_2d = self.supervisor.apply_projector_volts1d_to_volts2d(self.tt2pzt(tt_reconstruction))
            if self.config_env_rl['normalise_for_reward']:
                assert "dm_residual_tt" in self.norm_parameters
                pzt_reconstruction_from_tt_2d = self.standardise(pzt_reconstruction_from_tt_2d, key="dm_residual_tt")
            # Squared and multiply by -1
            r2d_pzt_reconstruction_from_tt = -np.square(pzt_reconstruction_from_tt_2d)
            r2d_non_linear_rec = np.stack([r2d_non_linear_rec, r2d_pzt_reconstruction_from_tt])

        # Scale
        r2d_non_linear_rec = r2d_non_linear_rec * self.config_env_rl['reward_scale']

        if self.config_env_rl['reward_type'] == "scalar_actuators":
            return r2d_non_linear_rec.mean()
        elif self.config_env_rl['reward_type'] == "2d_actuators":
            return r2d_non_linear_rec
        else:
            raise NotImplementedError