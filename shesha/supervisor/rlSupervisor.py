from shesha.supervisor.genericSupervisor import GenericSupervisor
from shesha.supervisor.mlSupervisor import MlSupervisor
from shesha.supervisor.components import AtmosCompass, DmCompass, RtcCompass, TargetCompass, TelescopeCompass, \
    WfsCompass
from shesha.supervisor.optimizers import ModalBasis, Calibration
import shesha.ao.basis as bas
import numpy as np
import torch

debug_reset_act = False

class RlSupervisor(MlSupervisor):
    def __init__(self,
                 config,
                 n_reverse_filtered_from_cmat, *,
                 filter_commands=True,
                 command_clip_value=1000,
                 initial_seed=1234,
                 which_modal_basis="Btt",
                 mode="only_rl",
                 device=-1,
                 control_tt=False,
                 use_second_version_of_modal_basis=False,
                 leaky_integrator_for_rl=False,
                 leak=-1,
                 reset_when_clip=False,
                 reduce_gain_tt_to=-1):
        """ Instantiates a RlSupervisor object

        Args:
            config: (config module) : Configuration module
            n_reverse_filtered_from_cmat :

        Kwargs:
            initial_seed: seed to start experiments
        """

        MlSupervisor.__init__(self,
                              config,
                              n_reverse_filtered_from_cmat=n_reverse_filtered_from_cmat,
                              filter_commands=filter_commands,
                              command_clip_value=command_clip_value,
                              initial_seed=initial_seed,
                              which_modal_basis=which_modal_basis,
                              mode=mode,
                              device=device,
                              control_tt=control_tt,
                              use_second_version_of_modal_basis=use_second_version_of_modal_basis,
                              leaky_integrator_for_rl=leaky_integrator_for_rl,
                              leak=leak,
                              reset_when_clip=reset_when_clip,
                              reduce_gain_tt_to=reduce_gain_tt_to)

        self.past_action_rl = None

        if self.num_dm > 1:
            if control_tt:
                self.action_1d_shape = self.command_shape
            else:
                self.action_1d_shape = self.pzt_shape
        else:
            self.action_1d_shape = self.pzt_shape

        self.action_2d_shape = self.pzt_2d_shape

    #
    #   __             __  __     _   _            _
    #  |   \          |  \/  |___| |_| |_  ___  __| |___
    #  |   |          | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |__/ efault    |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def reset(self, only_reset_dm=False, noise=None):
        """ Reset the simulation to return to its original state
        """
        if not only_reset_dm:
            self.atmos.reset_turbu(self.current_seed)
            self.wfs.reset_noise(self.current_seed, noise)
            for tar_index in range(len(self.config.p_targets)):
                self.target.reset_strehl(tar_index)
        self.dms.reset_dm()
        self.rtc.open_loop()
        self.rtc.close_loop()
        self.reset_past_commands()  # Only necessary for Only RL
        self.iter = 0
        self.number_clipped_actuators = 0
        self.mask_clipped_actuators = np.zeros(self.action_2d_shape).astype(np.float32)

    def reset_past_commands(self):
        """
        Resetting past commands (i.e. command at frame t-1)
        Past command RL is used either for only_rl_integrated or with slope_space + slope_space_correction
        """
        self.past_command = np.zeros(self.command_shape)
        self.past_action_rl = np.zeros(self.action_1d_shape)


    #   _____   __        __  __     _   _            _
    #  |  _  | |  |      |  \/  |___| |_| |_  ___  __| |___
    #  | |/ /  |  |__    | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  | |\ \  |_ _ _|   |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    #                            I
    #   _____   __               __  __     _   _            _
    #  |  _  | |  |             |  \/  |___| |_| |_  ___  __| |___
    #  | |/ /  |  |__           | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  | |\ \  |_ _ _| control  |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    #
    #       I a) Methods used by only RL and correction
    #
    #
    #

    def rl_control(self,
                   action: np.ndarray,
                   ncontrol: int,
                   unet_or_phase_command: np.ndarray = None,
                   controller_type: str = "RL"):

        if self.mode == "correction" and controller_type == "RL":
            final_command = self.correction_control(action=action,
                                                    unet_or_phase_command=unet_or_phase_command)
        else:
            final_command = self.only_rl_control(action=action)
        self.past_command = final_command
        self.past_action_rl = action

        self.rtc.set_command(ncontrol, final_command)

        #
        #       II b) Methods only used by only rl
        #
        #
        #

    def correction_control(self,
                           action: np.ndarray,
                           unet_or_phase_command: np.ndarray = None):  # this can be unet_or_phase_command

        if unet_or_phase_command is None:
            err = self.rtc.get_err(0)
        else:
            err = unet_or_phase_command.copy()
        if self.reduce_gain_tt_to > -1:
            err[-2:] *= self.reduce_gain_tt_to

        if self.leaky_integrator_for_rl:
            final_command_actuator = self.leak * self.past_command + err
        else:
            final_command_actuator = self.past_command + err

        if self.num_dm > 1:
            if self.control_tt:
                # Command has shape pzt + tt. action pzt + tt
                final_command_actuator += action
            else:
                # Command has shape pzt + tt. action pzt
                final_command_actuator[:-2] += action


            self.number_clipped_actuators = (np.abs(final_command_actuator[:-2]) >= self.command_clip_value).sum()
            self.mask_clipped_actuators = (np.abs(final_command_actuator[:-2]) >= self.command_clip_value).astype(float)
            # We clip for otherwise it can diverge
            final_command_actuator[:-2] = np.clip(final_command_actuator[:-2],
                                                  -self.command_clip_value,
                                                  self.command_clip_value)
            if self.reset_when_clip:
                if (np.abs(final_command_actuator[:-2]) >= self.command_clip_value).any():
                    final_command_actuator = self.reset_clipped_actuator(final_command_actuator)
        else:
            raise NotImplementedError

        if self.filter_commands:
            final_command_actuator = self.filter_dm_info_actuator_space_with_modal_basis(final_command_actuator)

        return final_command_actuator

    def only_rl_control(self,
                        action: np.ndarray):

        if self.num_dm > 1:
            if self.control_tt:
                # Command has shape pzt + tt. action pzt + tt
                final_command_actuator = self.past_command + action
            else:
                # Command has shape pzt + tt. action pzt
                # Only control pzt mirror
                final_command_actuator = self.past_command[:-2] + action
                # Concatenate tt command created by integrator
                commands_tip_tilt = self.rtc.get_command(0)[-2:]
                final_command_actuator = np.concatenate([final_command_actuator, commands_tip_tilt])


            self.number_clipped_actuators = (np.abs(final_command_actuator[:-2]) >= self.command_clip_value).sum()
            self.mask_clipped_actuators = (np.abs(final_command_actuator[:-2]) >= self.command_clip_value).astype(float)
            # Only clip pzt mirror
            final_command_actuator[:-2] = np.clip(final_command_actuator[:-2],
                                                  -self.command_clip_value,
                                                  self.command_clip_value)
        else:
            raise NotImplementedError
        # Filter if necessary
        if self.filter_commands:
            final_command_actuator = self.filter_dm_info_actuator_space_with_modal_basis(final_command_actuator)
        return final_command_actuator

    #
    #   ___  ___         __  __     _   _            _
    #  |   \|  |        |  \/  |___| |_| |_  ___  __| |___
    #  | |\    |        | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |_|  \__|  ext   |_|  |_\___|\__|_||_\___/\__,_/__/
    #


    def move_dm_and_compute_strehl(self,
                                   action: np.ndarray,
                                   controller_type: str = "RL",
                                   unet_or_phase_command: np.ndarray = None
                                   ):

        w = 0
        tar_trace = range(len(self.config.p_targets))
        if controller_type in ["RL", "UNet+Linearv2", "Phasev2"]:
            self.rl_control(action, w, unet_or_phase_command, controller_type)
        elif controller_type == "Integrator_tt2pzt":
            command = self.rtc.get_command(0)
            commant_pzt_from_tt = self.tt2pzt(command[-2:])
            command[:-2] += commant_pzt_from_tt
            command[-2:] = 0
            self.rtc.set_command(0, command)
        elif controller_type in ["UNet", "UNet+Linear", "Phase"]:
            if self.reduce_gain_tt_to > -1:
                action[-2:] *= self.reduce_gain_tt_to

            command = self.past_command + action
            self.past_command = command
            self.rtc.set_command(0, command)
        elif controller_type == "Integrator":
            self.past_command = self.rtc.get_command(0) # Saving past command in case we want to retrieve it
            if self.reduce_gain_tt_to > -1:
                self.past_command[-2:] *= self.reduce_gain_tt_to
            self.rtc.set_command(0, self.past_command)
        else:
            raise NotImplementedError

        self.rtc.apply_control(w)

        for tar_index in tar_trace:
            self.target.comp_tar_image(tar_index)
            self.target.comp_strehl(tar_index)

        self.iter += 1

