import numpy as np
import os
import matplotlib.pyplot as plt
from src.global_cte import FOLDER_INSIGHTS_RL, FOLDER_PROJECTORS

def compute_phase_from_command(supervisor, mode, send_to_dm, dm_index, basis, pupil,
                               debug_targetphase2modes=False, return_phase_original=False):
    supervisor.dms._dms.d_dms[dm_index].set_com(send_to_dm)
    supervisor.dms._dms.d_dms[dm_index].comp_shape()
    supervisor.target.raytrace(index=0, dms=supervisor.dms, ncpa=False, reset=True)
    phase_original = supervisor.target.get_tar_phase(0)

    if debug_targetphase2modes:
        plot_modes_from_phase(phase_original, mode, basis)

    phase = phase_original[np.where(pupil)]
    phase = phase - phase.mean()

    if return_phase_original:
        return phase, phase_original
    else:
        return phase


def plot_modes_from_phase(phase: np.ndarray, mode: int, basis: str) -> None:
    """
    Plots modes from phase
    :param phase: current phase value
    :param mode: current mode
    :param basis: basis used "actuator" or "btt"
    """
    if not os.path.exists(FOLDER_INSIGHTS_RL + "/modes_2dm_debug/"):
        os.mkdir(FOLDER_INSIGHTS_RL + "/modes_2dm_debug/")
    plt.imshow(phase)
    plt.savefig(FOLDER_INSIGHTS_RL + "/modes_2dm_debug/" + str(basis) + "_" + str(mode) + ".png")
    plt.close()


class ProjectorCreator:
    def __init__(self,
                 parameter_file,
                 supervisor,
                 second_dm_index=1,
                 debug_targetphase2modes=False):
        self.supervisor = supervisor
        self.debug_targetphase2modes = debug_targetphase2modes
        self.modes2volts = supervisor.modes2volts
        self.pupil = supervisor.get_s_pupil()
        self.where_is_pupil = np.where(self.pupil)
        self.second_dm_index = second_dm_index
        self.path_projectors = FOLDER_PROJECTORS + parameter_file[:-3]
        self.num_dm = supervisor.num_dm
        self.command_shape = self.supervisor.command_shape

    def get_btt2targetphase_2dm(self):
        self.supervisor.reset(only_reset_dm=False)
        modes2phase = np.zeros((self.where_is_pupil[0].shape[0], self.modes2volts.shape[1]))
        for mode in range(self.modes2volts.shape[1]):
            print("Obtaining projector, btt mode:", mode)
            send_to_dm = self.modes2volts[:, mode]  # This has shape volts x modes
            if mode < (self.modes2volts.shape[1] - 2):
                phase = compute_phase_from_command(self.supervisor,
                                                   mode,
                                                   send_to_dm[:-2],
                                                   dm_index=0,
                                                   basis="btt",
                                                   pupil=self.pupil,
                                                   debug_targetphase2modes=self.debug_targetphase2modes)
            else:
                phase = compute_phase_from_command(self.supervisor,
                                                   mode,
                                                   send_to_dm[-2:],
                                                   dm_index=self.second_dm_index,
                                                   basis="btt",
                                                   pupil=self.pupil,
                                                   debug_targetphase2modes=self.debug_targetphase2modes)

            modes2phase[:, mode] = phase.copy().reshape(-1)
            self.supervisor.reset(only_reset_dm=False)

        phase2modes = np.linalg.pinv(modes2phase)

        return modes2phase, phase2modes

    def get_projector_targetphase2modesphase(self, basis="btt"):
        """
        Note, the "left" side has better computation time than the "right" side.
        :param basis: "btt"
        :return: projector_targetphase2modes
        """
        print("Basis", basis)
        if basis == "btt":
            if os.path.exists(self.path_projectors + "_btt_phase2modes.npy"):
                modes2phase = np.load(self.path_projectors + "_btt_modes2phase.npy")
                phase2modes = np.load(self.path_projectors + "_btt_phase2modes.npy")
            else:
                print("Computing projector")
                if basis == "btt" and self.num_dm == 2:
                    modes2phase, phase2modes = self.get_btt2targetphase_2dm()
                elif basis == "btt" and self.num_dm == 1:
                    raise AssertionError
                else:
                    raise NotImplementedError
                np.save(self.path_projectors + "_btt_modes2phase.npy", modes2phase)
                np.save(self.path_projectors + "_btt_phase2modes.npy", phase2modes)
        else:
            raise NotImplementedError
        return modes2phase, phase2modes

    def get_projector_targetphase2actuator(self, debug_tt=True):

        if os.path.exists(self.path_projectors + "_actuator_actuator2phase.npy"):
            actuator2phase = np.load(self.path_projectors + "_actuator_actuator2phase.npy")
            phase2actuator = np.load(self.path_projectors + "_actuator_phase2actuator.npy")
            actuatorpzt2phase = np.load(self.path_projectors + "_actuator_actuatorpzt2phase.npy")
            phase2actuatorpzt = np.load(self.path_projectors + "_actuator_phase2actuatorpzt.npy")
            actuatortt2phase = np.load(self.path_projectors + "_actuator_actuatortt2phase.npy")
            phase2actuatortt = np.load(self.path_projectors + "_actuator_phase2actuatortt.npy")
        else:
            print("Computing projector")

            actuator2phase, phase2actuator, actuatorpzt2phase, phase2actuatorpzt, actuatortt2phase, phase2actuatortt \
                = self.get_act2targetphase_2dm()

            np.save(self.path_projectors + "_actuator_actuator2phase.npy", actuator2phase)
            np.save(self.path_projectors + "_actuator_phase2actuator.npy", phase2actuator)
            np.save(self.path_projectors + "_actuator_actuatorpzt2phase.npy", actuatorpzt2phase)
            np.save(self.path_projectors + "_actuator_phase2actuatorpzt.npy", phase2actuatorpzt)
            np.save(self.path_projectors + "_actuator_actuatortt2phase.npy", actuatortt2phase)
            np.save(self.path_projectors + "_actuator_phase2actuatortt.npy", phase2actuatortt)

            if debug_tt:
                plt.plot(actuatortt2phase)
                plt.savefig(self.path_projectors + "_debug_tt.png")
                plt.close()

        return actuator2phase, phase2actuator, actuatorpzt2phase, phase2actuatorpzt, actuatortt2phase, phase2actuatortt

    def get_act2targetphase_2dm(self):
        self.supervisor.reset(only_reset_dm=False)
        actuator2phase = np.zeros((self.where_is_pupil[0].shape[0], self.command_shape[0]), np.float32)
        actuatorpzt2phase = np.zeros((self.where_is_pupil[0].shape[0], self.command_shape[0]-2), np.float32)
        actuatortt2phase = np.zeros((self.where_is_pupil[0].shape[0], 2), np.float32)
        for actuator in range(self.command_shape[0]):
            print("Obtaining projector act -> phase, act number:", actuator)
            send_to_dm = np.zeros(self.command_shape)
            send_to_dm[actuator] += 1  # This has shape volts x modes
            if actuator < (self.command_shape[0] - 2):
                phase = compute_phase_from_command(self.supervisor,
                                                   actuator, send_to_dm[:-2], dm_index=0, basis="actuators",
                                                   pupil=self.pupil,
                                                   debug_targetphase2modes=self.debug_targetphase2modes)
                actuatorpzt2phase[:, actuator] = phase.copy().reshape(-1)
            else:
                # TODO found a bug here it was [:-2] instead of [-2:] but how it did work before then?
                # Maybe I added it after I got the first projectors somehow?
                phase = compute_phase_from_command(self.supervisor,
                                                   actuator, send_to_dm[-2:], dm_index=self.second_dm_index,
                                                   basis="actuators",  pupil=self.pupil,
                                                   debug_targetphase2modes=self.debug_targetphase2modes)

                actuatortt2phase[:, actuator - (self.command_shape[0] - 2)] = phase.copy().reshape(-1)
            actuator2phase[:, actuator] = phase.copy().reshape(-1)
            self.supervisor.reset(only_reset_dm=False)

        phase2actuatorpzt = np.linalg.pinv(actuatorpzt2phase)
        phase2actuatortt = np.linalg.pinv(actuatortt2phase)
        phase2actuator = np.linalg.pinv(actuator2phase)

        return actuator2phase, phase2actuator, actuatorpzt2phase, phase2actuatorpzt, actuatortt2phase, phase2actuatortt

    def test_projectors(self, env):
        s_tt = [0.33, 2]
        pzt_from_tt = env.tt2pzt(s_tt)
        # 1
        env.supervisor.reset(only_reset_dm=False)
        env.supervisor.dms._dms.d_dms[1].set_com(s_tt)
        env.supervisor.dms._dms.d_dms[1].comp_shape()
        env.supervisor.target.raytrace(index=0, dms=env.supervisor.dms, ncpa=False, reset=True)
        phase_tt = env.supervisor.target.get_tar_phase(0)
        # 2
        env.supervisor.reset(only_reset_dm=False)
        env.supervisor.dms._dms.d_dms[0].set_com(pzt_from_tt)
        env.supervisor.dms._dms.d_dms[0].comp_shape()
        env.supervisor.target.raytrace(index=0, dms=env.supervisor.dms, ncpa=False, reset=True)
        phase_pzt_from_tt = env.supervisor.target.get_tar_phase(0)
        # 3
        difference = phase_tt - phase_pzt_from_tt


        plt.figure(figsize=(10, 8))

        # Plot phase_tt
        plt.subplot(1, 3, 1)
        plt.imshow(phase_tt)
        plt.legend()

        # Plot phase_pzt_from_tt
        plt.subplot(1, 3, 2)
        plt.imshow(phase_pzt_from_tt)
        plt.legend()

        # Plot difference
        plt.subplot(1, 3, 3)
        plt.imshow(difference)
        plt.legend()

        plt.tight_layout()
        plt.savefig("check_projector2.png")


def get_mask_influence_function(supervisor, parameter_file:str, plot_influence:bool=False, reference_actuator_index:int=500):
    """
    Computes a mask that would normalize the contribution of each actuator
    This is done such that a reward of edge actuators do not overcontribute to the overall problem
    supervisor: instance of compass supervisor
    parameter_file: current parameter file used
    plot_influence: if true the influence functions are plotted
    reference_actuator_index: index of an actuator which is not in the edge. For 40x40 DM used, the value 500
    """
    if os.path.exists(FOLDER_PROJECTORS + parameter_file[:-3] + "_mask_influence_function.npy"):
        return np.load(FOLDER_PROJECTORS + parameter_file[:-3] + "_mask_influence_function.npy")
    supervisor.reset()
    responses = []

    reference_send_to_dm = np.zeros(supervisor.rtc.get_command(0).shape[0])
    reference_send_to_dm[reference_actuator_index] += 1  # this one seems to end ok
    r_pupil, r_full = compute_phase_from_command(supervisor, reference_actuator_index, reference_send_to_dm[:-2], dm_index=0, basis="actuators", pupil=supervisor.get_s_pupil(),
                                                       debug_targetphase2modes=False, return_phase_original=True)

    for mode in range(supervisor.rtc.get_command(0).shape[0] - 2):
        print(mode)
        send_to_dm = np.zeros(supervisor.modes2volts.shape[0])  # This has shape volts x modes
        send_to_dm[mode] += 1
        ph_pupil, ph_full = compute_phase_from_command(supervisor, mode, send_to_dm[:-2], dm_index=0, basis="actuators", pupil=supervisor.get_s_pupil(),
                                                       debug_targetphase2modes=False, return_phase_original=True)
        supervisor.reset()
        responses.append(np.sum(np.abs(ph_pupil)) / np.sum(np.abs(r_pupil)))

    responses_array = np.array(responses)

    responses2d = supervisor.apply_projector_volts1d_to_volts2d(responses_array)

    if plot_influence:
        plt.imshow(responses2d)
        plt.savefig(FOLDER_PROJECTORS + parameter_file[:-3] + "_mask_influence_function.png")
        plt.colorbar()
        plt.close()
    np.save(FOLDER_PROJECTORS + parameter_file[:-3] + "_mask_influence_function.npy", responses2d)
    return responses2d