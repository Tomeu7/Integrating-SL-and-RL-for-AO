from shesha.supervisor.genericSupervisor import GenericSupervisor
from shesha.supervisor.compassSupervisor import CompassSupervisor
from shesha.supervisor.components import AtmosCompass, DmCompass, RtcCompass, TargetCompass, TelescopeCompass, \
    WfsCompass
from shesha.supervisor.optimizers import ModalBasis, Calibration
import shesha.ao.basis as bas
import numpy as np
import torch

debug_reset_act = False
class MlSupervisor(CompassSupervisor):
    def __init__(self, 
                 config,
                 n_reverse_filtered_from_cmat, 
                 *,
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
                 reduce_gain_tt_to=-1,
                 value_commands_deltapos_v2=-1):
        """ Instantiates a MlSupervisor object

        Args:
            config: (config module) : Configuration module
            n_reverse_filtered_from_cmat : number of modes filtered

        Kwargs:
            initial_seed: seed to start experiments
        """

        GenericSupervisor.__init__(self, config)
        # Generic Compass
        self.basis = ModalBasis(self.config, self.dms, self.target)
        self.calibration = Calibration(self.config, self.tel, self.atmos, self.dms,
                                       self.target, self.rtc, self.wfs)
        # Initial values
        self.initial_seed, self.current_seed = initial_seed, initial_seed
        self.filter_commands = filter_commands
        self.command_clip_value = command_clip_value
        self.num_dm = len(self.config.p_dms)
        self.mode = mode
        self.use_second_version_of_modal_basis = use_second_version_of_modal_basis
        # Integrator
        self.leaky_integrator_for_rl = leaky_integrator_for_rl
        self.leak = leak
        self.reduce_gain_tt_to = reduce_gain_tt_to
        assert 1 > self.reduce_gain_tt_to > 0 or self.reduce_gain_tt_to == -1

        self.x_pos_norm, self.y_pos_norm, self.diff_x, self.diff_y = self.get_positions_actuators()
        self.mask_valid_actuators = self.create_mask_valid_actuators()
        self.device = device
        # Save past commands and past actions
        self.past_command = None

        # dm/action shapes
        self.command_shape = self.rtc.get_command(0).shape  # always in 1D

        self.past_action_rl = None
        if self.num_dm > 1:
            self.pzt_shape = self.rtc.get_command(0)[:-2].shape
            self.tt_shape = self.rtc.get_command(0)[-2:].shape
        else:
            self.pzt_shape = self.rtc.get_command(0).shape
            self.tt_shape = None
        self.pzt_2d_shape = self.mask_valid_actuators.shape

        # Modes
        # Only create basis if first controller is not geometric
        if self.config.p_controllers[0].get_type() != "geo":
            self.modes2volts, self.volts2modes, self.modal_basis, self.modal_basis_2d, self.modes_filtered, self.modes2volts_torch, \
                self.volts2modes_torch, self.zeros_tt, self.modal_basis_torch = self.create_modal_basis_variables(use_second_version_of_modal_basis, which_modal_basis, n_reverse_filtered_from_cmat)
        else:
            self.modes2volts, self.volts2modes = np.ones((1307, 1310), np.float32), None
            self.modal_basis, self.modal_basis_2d, self.modes_filtered, self.modes2volts_torch, \
                self.volts2modes_torch, self.zeros_tt, self.modal_basis_torch = [None] * 7

        # Control TT
        self.control_tt = control_tt
        if self.control_tt:
            assert self.num_dm > 1, "To control the TT you need a PZT and TT mirrors"

        self.actuatortt2phase = None
        self.phase2actuatorpzt = None
        self.reset_when_clip = reset_when_clip
        if self.reset_when_clip or value_commands_deltapos_v2:
            self.look_up_table, self.distance_matrix_sorted = self.compute_distances_for_reset_actuators()
        else:
            self.look_up_table, self.distance_matrix_sorted = None, None

        self.linear_gain = self.rtc._rtc.d_control[0].gain
        self.number_clipped_actuators = 0
        self.mask_clipped_actuators = np.zeros(self.pzt_2d_shape).astype(np.float32)


    #
    #   __             __  __     _   _            _
    #  |   \          |  \/  |___| |_| |_  ___  __| |___
    #  |   |          | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |__/ efault    |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def create_modal_basis_variables(self, use_second_version_of_modal_basis: bool, which_modal_basis: str, n_reverse_filtered_from_cmat: int):
        """
        Creates modal basis
        :param use_second_version_of_modal_basis: if we use Btt.T.dot(Btt) to directly filter
        :param which_modal_basis: "Btt" is only implemented for now
        :param n_reverse_filtered_from_cmat: number of modes filtered
        """
        dms = []
        p_dms = []
        if type(self.config.p_controllers) == list:
            for config_p_controller in self.config.p_controllers:
                if config_p_controller.get_type() != "geo":
                    for dm_idx in config_p_controller.get_ndm():
                        dms.append(self.dms._dms.d_dms[dm_idx])
                        p_dms.append(self.config.p_dms[dm_idx])

        else:
            dms = self.dms._dms.d_dms
            p_dms = self.config.p_dms

        modes2volts, volts2modes =\
                self.basis.compute_modes_to_volts_basis(dms, p_dms, modal_basis_type=which_modal_basis)

        if use_second_version_of_modal_basis:
            modes2volts_filtered, volts2modes_filtered = self.filter_modal_basis(modes2volts, volts2modes, nfilt=n_reverse_filtered_from_cmat)
            modal_basis = modes2volts_filtered.dot(volts2modes_filtered)
            modal_basis_2d = self.create_modal_basis_for_2d()
        else:
            modal_basis, modal_basis_2d = None, None

        modes_filtered = self.manage_filtered_modes(modes2volts, n_reverse_filtered_from_cmat)
        self.obtain_and_set_cmat_filtered(modes2volts, n_reverse_filtered_from_cmat)

        # Device and if we do some calculations with torch on GPU
        if self.device >= 0:
            if modal_basis is not None:
                modes2volts_torch, volts2modes_torch = None, None
                modal_basis_torch = torch.FloatTensor(modal_basis).to(self.device)
            else:
                modal_basis_torch = None
                modes2volts_torch, volts2modes_torch = \
                    torch.FloatTensor(modes2volts).to(self.device), torch.FloatTensor(volts2modes).to(self.device)
            zeros_tt = torch.zeros(2).to(self.device)
        else:
            modes2volts_torch, volts2modes_torch = None, None
            zeros_tt = None
            modal_basis_torch = None
        return modes2volts, volts2modes, modal_basis, modal_basis_2d, modes_filtered, modes2volts_torch, volts2modes_torch, zeros_tt, modal_basis_torch

    def set_gain(self, g: float) -> None:
        """
        Sets a certain gain in the system for the linear approach with integrator
        :param g: gain to be set
        """
        self.linear_gain = g
        if np.isscalar(g):
            self.rtc._rtc.d_control[0].set_gain(g)
        else:
            raise ValueError("Cannot set array gain w/ generic + integrator law")

    def filter_modal_basis(self, modes2volts_unfiltered, volts2modes_unfiltered, nfilt):

        if self.num_dm == 1:
            raise NotImplementedError
        else:
            # Filtering on Btt modes
            modes2volts = np.zeros((modes2volts_unfiltered.shape[0], modes2volts_unfiltered.shape[1] - nfilt))
            modes2volts[:, :modes2volts.shape[1] - 2] = modes2volts_unfiltered[:, :modes2volts_unfiltered.shape[1] - (nfilt + 2)]
            # TT part
            modes2volts[:, modes2volts.shape[1] - 2:] = modes2volts_unfiltered[:, modes2volts_unfiltered.shape[1] - 2:]
            volts2modes = np.zeros((volts2modes_unfiltered.shape[0] - nfilt, volts2modes_unfiltered.shape[1]))
            volts2modes[:volts2modes.shape[0] - 2, :] = volts2modes_unfiltered[:volts2modes_unfiltered.shape[0] - (nfilt + 2)]
            # TT part
            volts2modes[volts2modes.shape[0] - 2:, :] = volts2modes_unfiltered[volts2modes_unfiltered.shape[0] - 2:, :]

        return modes2volts, volts2modes
    """
    def create_projector_volts1d_to_volts2d(self):

        def get_linspace_for_projection_volts1d_to_2d():
            
            x_pos_actuators_ = self.config.p_dms[0]._xpos

            y_pos_actuators_ = self.config.p_dms[0]._ypos
            x_pos_linspace_ = np.linspace(self.config.p_geom._p1 - 0.5,
                                         self.config.p_geom._p2 + 0.5,
                                         self.config.p_dms[0].nact)
            y_pos_linspace_ = np.linspace(self.config.p_geom._p1 - 0.5,
                                         self.config.p_geom._p2 + 0.5,
                                         self.config.p_dms[0].nact)

            return x_pos_linspace_, y_pos_linspace_, x_pos_actuators_, y_pos_actuators_

        x_pos_linspace, y_pos_linspace, x_pos_actuators, y_pos_actuators = \
            get_linspace_for_projection_volts1d_to_2d()

        if self.num_dm > 1:
            total_len = len(self.rtc.get_command(0)) - 2
        else:
            total_len = len(self.rtc.get_command(0))

        epsilon = 1e-3
        # Projector has the shape number of commands - 2 (which is tip tilt)
        projector_volts1d_to_volts2d = np.zeros((total_len,
                                                 self.config.p_dms[0].nact *
                                                 self.config.p_dms[0].nact))

        for i in range(len(x_pos_linspace)):
            for j in range(len(y_pos_linspace)):
                which_idx = -1
                for idx in range(len(x_pos_actuators)):
                    # adding epsilon solves bug of precission for some parameter files
                    if np.abs(x_pos_actuators[idx] - x_pos_linspace[i]) < epsilon \
                            and np.abs(y_pos_actuators[idx] - y_pos_linspace[j]) < epsilon:
                        which_idx = idx
                if which_idx != -1:
                    projector_volts1d_to_volts2d[which_idx, int(i * self.config.p_dms[0].nact) + int(j)] = 1

        projector_volts2d_to_volts1d = np.linalg.pinv(projector_volts1d_to_volts2d)

        return projector_volts1d_to_volts2d, projector_volts2d_to_volts1d
    """

    def manage_filtered_modes(self, modes2volts: np.ndarray, n_reverse_filtered_from_cmat: int) -> np.ndarray:

        if self.num_dm > 1:
            modes_filtered = np.arange(modes2volts.shape[1] - n_reverse_filtered_from_cmat - 2,
                                        modes2volts.shape[1] - 2)
        else:
            modes_filtered = np.arange(modes2volts.shape[1] - n_reverse_filtered_from_cmat,
                                       modes2volts.shape[1])

        return modes_filtered

    def add_one_to_seed(self) -> None:
        self.current_seed += 1

    def compute_distances_for_reset_actuators(self):
        """
        Computes two dicts:
        + distance_matrix_sorted: which contains per idx the sorted distances to other valid actuators
        + look_up_table: the same but with indices based on distance_matrix_sorted
        """

        # xpos is a matrix such as
        # 0 1 2 ... 39
        # 0 1 2 ... 39
        # ...
        # 0 1 2 ... 39
        # ypos is a matrix such as
        # 0 0 0 ...
        # 1 1 1 ...
        # ...
        # 39 39 39 ...
        grid = np.arange(40)
        xpos, ypos = np.meshgrid(grid, grid)
        # adding very high or low value for mask outside the valid mask
        xpos[self.mask_valid_actuators == 0] = -99999
        ypos[self.mask_valid_actuators == 0] = -99999
        # reshape
        xpos = xpos.reshape(-1)
        ypos = ypos.reshape(-1)
        # We compute distances
        # On unvalid actuators this will be quite high but it does not matter as we will discard them later
        distance_matrix = ((xpos[None, :] - xpos[:, None]) ** 2 + (ypos[None, :] - ypos[:, None]) ** 2) ** 0.5
        
        # Build lookup table
        look_up_table = {}
        distance_matrix_sorted = {}
        idx = 0
        for i in range(len(self.rtc.get_command(0)) - 2):
            while True:
                # Only compute distance or neighbour if its inside mask_valid_actuators
                if self.mask_valid_actuators.reshape(-1)[idx] == 1:
                    break
                else:
                    idx += 1
            valid_distances = distance_matrix[idx, self.mask_valid_actuators.reshape(-1) == 1]
            distance_matrix_sorted[i] = np.sort(valid_distances)
            closest = np.argsort(valid_distances)
            look_up_table[i] = closest
            idx += 1

        return look_up_table, distance_matrix_sorted

    def obtain_and_set_cmat_filtered(self, btt2act, n_reverse_filtered_from_cmat):

        # print("+ Setting cmat filtered...")

        if n_reverse_filtered_from_cmat > -1:
            # print("Using Btt basis")
            # print("+ Shape Btt basis", btt2act.shape)
            assert type(n_reverse_filtered_from_cmat) == int
            cmat = bas.compute_cmat_with_Btt(rtc=self.rtc._rtc,
                                                Btt=btt2act,
                                                nfilt=n_reverse_filtered_from_cmat)
            print("+ Number of filtered modes", n_reverse_filtered_from_cmat)
        else:
            print("+ WARNING: NO CMAT BUILT FROM BTT, IT IS THE ORIGINAL WITHOUT FILTERING ANY MODES!!!")
            cmat = None

        return cmat

    def get_positions_actuators(self):
        """
        Gets current position of actuators
        """
        xpos = self.config.p_dms[0]._xpos
        ypos = self.config.p_dms[0]._ypos
        diff_x = np.array([xpos[i + 1] - xpos[i] for i in range(len(xpos) - 1)])
        pitch = diff_x[diff_x > 0][0]
        diff_y = np.array([ypos[i + 1] - ypos[i] for i in range(len(ypos) - 1)])
        pitch_y = diff_y[diff_y > 0][0] # Not used
        x_pos_norm = np.round((xpos - xpos.min()) / pitch).astype(np.int32)
        y_pos_norm = np.round((ypos - ypos.min()) / pitch).astype(np.int32)
        return x_pos_norm, y_pos_norm, diff_x, diff_y

    def create_mask_valid_actuators(self) -> np.ndarray:
        """
        Given xpos and ypos of actuator, create a mask of 2D actuators
        """
        mask_valid_actuators = np.zeros((self.config.p_dms[0].nact, self.config.p_dms[0].nact))
        mask_valid_actuators[self.x_pos_norm, self.y_pos_norm] = 1
        return mask_valid_actuators

    def apply_projector_volts1d_to_volts2d(self, command_1d: np.ndarray) -> np.ndarray:
        """
        Converts commands in 1D format to 2D format
        :param command_1d: current commands in 1D format
        """
        if command_1d.ndim == 3:
            command_1d = command_1d[0, :, :]
        command_2d = np.zeros((self.config.p_dms[0].nact, self.config.p_dms[0].nact))
        command_2d[self.x_pos_norm, self.y_pos_norm] = command_1d
        return command_2d

    def apply_projector_volts2d_to_volts1d(self, command_2d: np.ndarray) -> np.ndarray:
        """
        Convert commands in 2D to commands in 1D
        :param command_2d: current commands in 2D format
        """
        if command_2d.ndim == 3:
            command_2d = command_2d[0, :, :]
        return command_2d[self.x_pos_norm, self.y_pos_norm]
    
    def create_modal_basis_for_2d(self):
        """
        Creates modal basis for filtering for input that is in 2D
        """
        new_modal_basis = np.zeros((self.mask_valid_actuators.reshape(-1).shape[0], self.mask_valid_actuators.reshape(-1).shape[0]))
        mask_valid_actuators_1d = self.mask_valid_actuators.reshape(-1)
        rows, cols = np.ix_(mask_valid_actuators_1d == 1, mask_valid_actuators_1d == 1)
        new_modal_basis[rows, cols] = self.modal_basis[:-2, :-2]
        # test = np.random.normal(size=env.command_shape)
        # test_2d = env.supervisor.apply_projector_volts1d_to_volts2d(test[:-2])
        # test_original = env.supervisor.modal_basis.dot(test)
        # test_new = new_modal_basis.dot(test_2d.reshape(-1))
        # test_new_1d = env.supervisor.apply_projector_volts2d_to_volts1d(test_new.reshape((40,40)))
        return new_modal_basis


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
        self.iter = 0
        self.number_clipped_actuators = 0
        self.mask_clipped_actuators = np.zeros(self.pzt_2d_shape).astype(np.float32)

    #     ___                  _      __  __     _   _            _
    #    / __|___ _ _  ___ _ _(_)__  |  \/  |___| |_| |_  ___  __| |___
    #   | (_ / -_) ' \/ -_) '_| / _| | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #    \___\___|_||_\___|_| |_\__| |_|  |_\___|\__|_||_\___/\__,_/__/

    def _init_tel(self):
        """Initialize the telescope component of the supervisor as a TelescopeCompass
        """
        self.tel = TelescopeCompass(self.context, self.config)

    def _init_atmos(self):
        """Initialize the atmosphere component of the supervisor as a AtmosCompass
        """
        self.atmos = AtmosCompass(self.context, self.config)

    def _init_dms(self):
        """Initialize the DM component of the supervisor as a DmCompass
        """
        self.dms = DmCompass(self.context, self.config)

    def _init_target(self):
        """Initialize the target component of the supervisor as a TargetCompass
        """
        if self.tel is not None:
            self.target = TargetCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_wfs(self):
        """Initialize the wfs component of the supervisor as a WfsCompass
        """
        if self.tel is not None:
            self.wfs = WfsCompass(self.context, self.config, self.tel)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")

    def _init_rtc(self):
        """Initialize the rtc component of the supervisor as a RtcCompass
        """
        if self.wfs is not None:
            self.rtc = RtcCompass(self.context, self.config, self.tel, self.wfs,
                                  self.dms, self.atmos, cacao=False)
        else:
            raise ValueError("Configuration not loaded or Telescope not initilaized")


    #
    #
    #
    #
    #

    def filter_dm_info_actuator_space_with_modal_basis(self,
                                                       dm_info_to_filter,
                                                       add_tip_tilt_to_not_break=False,
                                                       return_tt=False):

        if self.device > -1:
            if isinstance(dm_info_to_filter, np.ndarray):
                dm_info_to_filter = torch.FloatTensor(dm_info_to_filter).to(self.device)
            else:
                dm_info_to_filter = dm_info_to_filter.to(self.device)
            dm_info_filtered = self.filter_dm_info_actuator_space_with_modal_basis_torch(dm_info_to_filter,
                                                                                         add_tip_tilt_to_not_break,
                                                                                         return_tt)
        else:
            dm_info_filtered = self.filter_dm_info_actuator_space_with_modal_basis_numpy(dm_info_to_filter,
                                                                                         add_tip_tilt_to_not_break,
                                                                                         return_tt)

        return dm_info_filtered

    @torch.no_grad()
    def filter_dm_info_actuator_space_with_modal_basis_torch(self,
                                                             dm_info_to_filter,
                                                             add_tip_tilt_to_not_break=False,
                                                             return_tt=False):

        add_tip_tilt_to_not_break_bool = self.num_dm > 1 and\
                                         add_tip_tilt_to_not_break

        if add_tip_tilt_to_not_break_bool:
            dm_info_to_filter = torch.cat([dm_info_to_filter, self.zeros_tt])

        if self.modal_basis_torch is not None:
            dm_info_filtered = self.modal_basis_torch @ dm_info_to_filter
        else:
            dm_info_to_filter_modes = self.volts2modes_torch@dm_info_to_filter
            dm_info_to_filter_modes[self.modes_filtered] = 0
            dm_info_filtered = self.modes2volts_torch@dm_info_to_filter_modes

        if add_tip_tilt_to_not_break_bool and not return_tt:
            dm_info_filtered = dm_info_filtered[:-2]

        return dm_info_filtered.cpu().numpy()

    def filter_dm_info_actuator_space_with_modal_basis_numpy(self,
                                                             dm_info_to_filter,
                                                             add_tip_tilt_to_not_break=False,
                                                             return_tt=False):

        add_tip_tilt_to_not_break_bool = self.num_dm > 1 and\
                                         add_tip_tilt_to_not_break

        if add_tip_tilt_to_not_break_bool:
            dm_info_to_filter = np.concatenate([dm_info_to_filter, [0.0, 0.0]])

        if self.modal_basis_torch is not None:
            dm_info_filtered = self.modal_basis.dot(dm_info_to_filter)
        else:
            dm_info_to_filter_modes = self.volts2modes.dot(dm_info_to_filter)
            dm_info_to_filter_modes[self.modes_filtered] = 0
            dm_info_filtered = self.modes2volts.dot(dm_info_to_filter_modes)

        if add_tip_tilt_to_not_break_bool and not return_tt:
            dm_info_filtered = dm_info_filtered[:-2]

        return dm_info_filtered

    def reset_clipped_actuator(self, final_command_actuator):
        if debug_reset_act:
            np.save("initial.npy", self.apply_projector_volts1d_to_volts2d(final_command_actuator[:-2]))
        # Identify actuators that reached the clip value.
        clipped_indices = np.where(np.abs(final_command_actuator[:-2]) == self.command_clip_value)[0]
        if debug_reset_act:
            print("Clipped indices", clipped_indices)
            print("Commands with clipped", final_command_actuator[:-2][clipped_indices])
        closest_to_clipped = []
        for index in clipped_indices:
            index_in_lookup = 0
            while True:
                closest_index = self.look_up_table[index][index_in_lookup]
                if debug_reset_act:
                    print("Index", index, "index_in_lookup", index_in_lookup, "closest_index", closest_index)
                if closest_index not in clipped_indices:
                    break
                else:
                    index_in_lookup += 1
            closest_to_clipped.append(closest_index)
        if debug_reset_act:
            print("Resetting indices: ", clipped_indices, " to values of: ", np.array(closest_to_clipped))
        # Use advanced indexing to get the values of the two closest non-clipped actuators
        closest_values = final_command_actuator[np.array([closest_to_clipped])]

        # Assign the closest non-clipped actuators
        final_command_actuator[clipped_indices] = closest_values
        if debug_reset_act:
            np.save("final.npy", self.apply_projector_volts1d_to_volts2d(final_command_actuator[:-2]))
        return final_command_actuator

    #
    #   ___  ___         __  __     _   _            _
    #  |   \|  |        |  \/  |___| |_| |_  ___  __| |___
    #  | |\    |        | |\/| / -_)  _| ' \/ _ \/ _` (_-<
    #  |_|  \__|  ext   |_|  |_\___|\__|_||_\___/\__,_/__/
    #

    def raytrace_target(self, ncontrol: int):
        """
        Does the raytacing operation
        :param ncontrol: ncontrol that will have an associated target
        :return: None
        """
        t = ncontrol
        if self.atmos.is_enable:
            self.target.raytrace(t, tel=self.tel, atm=self.atmos, dms=self.dms)
        else:
            self.target.raytrace(t, tel=self.tel, dms=self.dms)

    #
    #           II a) PART TWO METHODS
    #

    def move_dm_and_compute_strehl(self,
                                   action: np.ndarray,
                                   controller_type: str = "RL",
                                   unet_or_phase_command: np.ndarray = None
                                   ):

        w = 0
        tar_trace = range(len(self.config.p_targets))
        if controller_type in ["RL", "UNet+Linearv2", "Phasev2"]:
            raise NotImplementedError
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

    def tt2pzt(self, dm_tt):
        # self.tt_shape to phase shape
        phase = self.actuatortt2phase.dot(dm_tt)
        # remove piston
        phase = phase - phase.mean()
        # phase to self.tt_shape
        dm_pzt_from_tt = self.phase2actuatorpzt.dot(phase)
        return dm_pzt_from_tt

    #
    #           II b) PART ONE METHODS
    #
    #

    def move_atmos_compute_wfs_reconstruction(self) -> None:

        w = 0
        self.atmos.move_atmos()

        self.raytrace_target(w)
        self.wfs.raytrace(w, tel=self.tel, atm=self.atmos)
        self.wfs.raytrace(w, dms=self.dms, ncpa=False, reset=False)
        self.wfs.compute_wfs_image(w)
        self.rtc.do_centroids(w)
        self.rtc.do_control(w)

    #
    #           III c) PART ONE METHODS
    #
    #

    def geom_next(self, t: int = 0, n: int = 0, apply_control: bool = True, do_control: bool = True) -> None:
        """
        A full step in COMPASS with the geometirc controller:
        + t: target index
        + n: controller index
        + apply_control: if True control is applied
        + do_control: if True control is calculated
        """
        self.atmos.move_atmos()
        self.target.raytrace(t, tel=self.tel, atm=self.atmos, ncpa=False)
        if do_control:
            self.rtc.do_control(n, sources=self.target.sources)
            self.target.raytrace(t, dms=self.dms, ncpa=True, reset=False)
            if apply_control:
                self.rtc.apply_control(n)
                self.past_command = self.rtc.get_command(0)
        
        self.target.comp_tar_image(t)
        self.target.comp_strehl(t)

        