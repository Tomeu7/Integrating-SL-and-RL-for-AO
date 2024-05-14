import numpy as np
from src.reinforcement_learning.shesha_modifications.compute_projectors import ProjectorManager


def rlHelper(supervisor):
    btt2act, act2btt = create_btt(supervisor)
    ptt2act, act2ptt, ptt2act_twott, act2ptt_twott = create_ptt(supervisor, btt2act, act2btt)

    # For SCExAO
    # + ptt2act_raw (1319, 1415) act2ptt_raw (1415, 1319)
    current_modes2volts = ptt2act if supervisor.config_rl.env_rl['use_ptt'] else btt2act
    current_volts2modes = act2ptt if supervisor.config_rl.env_rl['use_ptt'] else act2btt

    modes_filtered = manage_filtered_modes(supervisor,
                                           ptt2act,
                                           btt2act,
                                           supervisor.n_reverse_filtered_from_cmat)

    projector_manager = ProjectorManager(supervisor=supervisor,
                                         m2v=btt2act,
                                         ptt2act=ptt2act,
                                         second_dm_index=1,
                                         parameter_file=supervisor.config_rl.env_rl['parameters_telescope'])

    # TODO remove or simplify if
    if supervisor.config_rl.env_rl['which_modal_basis'] == "Btt" and \
            (supervisor.config_rl.env_rl['use_phase_to_modes_projectors'] or
             supervisor.config_rl.env_rl['s_dm_residual_phase'] or
             supervisor.config_rl.env_rl['s_dm_residual_gan'] or
             supervisor.config_rl.env_rl['s_dm_residual_mix'] or
             "from_gan" in supervisor.config_rl.env_rl['reward_type'] or
             "from_phase" in supervisor.config_rl.env_rl['reward_type'] or
            "cnn" in supervisor.config_rl.env_rl['reward_type'] or
            supervisor.config_rl.env_rl['reward_type'] == "TT_correction_2d_actuators" ):
        projector_targetphase2modes =\
            projector_manager.get_projector_targetphase2modesphase(basis="ptt" if supervisor.config_rl.env_rl['use_ptt']
                                                                   else "btt")
    else:
        projector_targetphase2modes = None

    projector_volts1d_to_volts2d, projector_volts2d_to_volts1d, mask_valid_actuators = \
        create_projectors_between_1d_2d(supervisor, projector_manager)

    zero_response = calculate_zero_response_sensor(supervisor)

    if supervisor.config_rl.env_rl['basis'] == "slope_space_correction" or \
            supervisor.config_rl.env_rl['reward_type'] == "zernike_from_psf":
        projector_phase2zernike, projector_zernike2phase, gerberch_saxton,\
            pupil_guide_star = get_zernike_projector(s_pupil=supervisor.get_s_pupil(),
                                                     i_pupil=supervisor.get_i_pupil(),
                                                     lambda_target_0=supervisor.config.p_targets[0].get_Lambda())
    else:
        projector_phase2zernike, projector_zernike2phase, gerberch_saxton, pupil_guide_star = None, None, None, None

    mask_valid_pixels_pyramid = None  # create_mask_valid_pixels_pyramid(supervisor)

    previous_reference_slopes, reference_slope_agent, memory_reference_slope_agent, total_update_slope_agent = \
        reference_slope_sac_creation(supervisor)

    tt_mirror_to_pzt_mirror, pzt_mirror_to_tt_mirror =\
        projector_manager.get_projector_tt_mirror_to_pzt_mirror(supervisor, basis="ptt" if supervisor.config_rl.env_rl['use_ptt']
                                                                   else "btt")

    return btt2act, act2btt, \
           ptt2act, act2ptt, \
           ptt2act_twott, act2ptt_twott, \
           current_modes2volts, current_volts2modes, \
           modes_filtered, \
           projector_targetphase2modes, \
           projector_volts1d_to_volts2d, projector_volts2d_to_volts1d, mask_valid_actuators, \
           zero_response, \
           projector_phase2zernike, projector_zernike2phase, gerberch_saxton, pupil_guide_star,\
           mask_valid_pixels_pyramid, \
           previous_reference_slopes, reference_slope_agent, memory_reference_slope_agent, total_update_slope_agent, \
           tt_mirror_to_pzt_mirror, pzt_mirror_to_tt_mirror

def create_btt(supervisor):
    # Create different projectors
    dms = []
    p_dms = []
    if type(supervisor.config.p_controllers) == list:
        for config_p_controller in supervisor.config.p_controllers:
            if config_p_controller.get_type() != "geo":
                for dm_idx in config_p_controller.get_ndm():
                    dms.append(supervisor.dms._dms.d_dms[dm_idx])
                    p_dms.append(supervisor.config.p_dms[dm_idx])

    else:
        dms = supervisor.dms._dms.d_dms
        p_dms = supervisor.config.p_dms

    # if deformable mirrors are not only GEO do btt
    if dms:
        """
        Btt : (np.ndarray(ndim=2,dtype=np.float32)) : Btt to Volts matrix (volts x modes shape)
    
        P (projector) : (np.ndarray(ndim=2,dtype=np.float32)) : Volts to Btt matrix (modes x volts shape)
        """
        modes2volts, volts2modes = \
            supervisor.basis.compute_modes_to_volts_basis(dms=dms, p_dms=p_dms,
                                                          modal_basis_type=
                                                          supervisor.config_rl.env_rl['which_modal_basis'])
    else:
        modes2volts, volts2modes = None, None

    return modes2volts, volts2modes


def create_ptt(supervisor, btt2act, act2btt):
    if supervisor.config_rl.env_rl['use_ptt']:
        from src.reinforcement_learning.shesha_modifications.compute_petal_basis_functions import \
            compute_petal_basis

        ptt2act_raw, act2ptt_raw = compute_petal_basis(supervisor, supervisor.config_rl.env_rl['selection_ptt'])

        ptt2act_twott = np.zeros((ptt2act_raw.shape[0] + 2, ptt2act_raw.shape[1] + 2), dtype=np.float32)
        act2ptt_twott = np.zeros((act2ptt_raw.shape[0] + 2, act2ptt_raw.shape[1] + 2), dtype=np.float32)

        if len(supervisor.config.p_dms) > 1:
            ptt2act_twott[:-2, :-2] = ptt2act_raw
            ptt2act_twott[-2:, -2:] = btt2act[-2:, -2:]
            act2ptt_twott[:-2, :-2] = act2ptt_raw
            act2ptt_twott[-2:, -2:] = act2btt[-2:, -2:]

            ptt2act, act2ptt = ptt2act_twott[:, 2:], act2ptt_twott[2:, :]
        else:
            ptt2act, act2ptt = ptt2act_raw.copy(), act2ptt_raw.copy()
            ptt2act_twott = None
            act2ptt_twott = None
        # Original
        # ptt2act = np.zeros((ptt2act_raw.shape[0] + 2, ptt2act_raw.shape[1]))
        # act x modes, in the modes we remove first two modes (TT from PZT mirror) and add two modes (TT FROM TT mirror)
        # ptt2act = np.zeros((ptt2act_raw.shape[0] + 2, ptt2act_raw.shape[1]))
        # A bit of visual explanation below for ptt2act
        # (rows actuators pzt + 2, columns modes pzt - 2 from removed tt modes + 2 from tt mirror)
        #                  Pzt column . <- TT mirror column
        #               (               )
        # Pzt row       (               )
        #               (               )
        #               (               )
        # TT mirror row (               )
        # ptt2act[:-2, :ptt2act_raw.shape[1] - 2] = ptt2act_raw[:, 2:]
        # ptt2act[:, -2:] = btt2act[:, -2:]
        # act2ptt = np.zeros((act2ptt_raw.shape[0], act2ptt_raw.shape[1] + 2))
        # act2ptt[:act2ptt_raw.shape[0] - 2, :-2] = act2ptt_raw[2:, :]
        # act2ptt[-2:, :] = act2btt[-2:, :]
    else:
        ptt2act, act2ptt = None, None
        ptt2act_twott, act2ptt_twott = None, None

    return ptt2act, act2ptt, ptt2act_twott, act2ptt_twott


def manage_filtered_modes(supervisor, ptt2act, modes2volts, n_reverse_filtered_from_cmat):
    if supervisor.config_rl.env_rl['use_ptt']:
        if len(supervisor.config.p_dms) > 1:
            modes_filtered = np.arange(ptt2act.shape[1] - n_reverse_filtered_from_cmat - 2,
                                            ptt2act.shape[1] - 2)
        else:
            modes_filtered = np.arange(ptt2act.shape[1] - n_reverse_filtered_from_cmat,
                                            ptt2act.shape[1])

    else:
        if len(supervisor.config.p_dms) > 1:
            modes_filtered = np.arange(modes2volts.shape[1] - n_reverse_filtered_from_cmat - 2,
                                            modes2volts.shape[1] - 2)
        else:
            modes_filtered = np.arange(modes2volts.shape[1] - n_reverse_filtered_from_cmat,
                                       modes2volts.shape[1])

    return modes_filtered


def create_projectors_between_1d_2d(supervisor, projector_manager):
    if supervisor.config_rl.env_rl['state_cnn_actuators'] or supervisor.config_rl.env_rl['state_cnn_wfs2actuators']:
        # To create a state as a 2D matrix from actuators instead of 1D

        projector_volts1d_to_volts2d, projector_volts2d_to_volts1d =\
            projector_manager.create_projector_volts1d_to_volts2d(set_values=True)

        if len(supervisor.config.p_dms) > 1:
            if supervisor.config_rl.env_rl['include_tip_tilt']:
                command_shape = supervisor.rtc.get_command(0).shape
            else:
                command_shape = supervisor.rtc.get_command(0)[:-2].shape
        else:
            command_shape = supervisor.rtc.get_command(0).shape

        mask_valid_actuators = projector_manager.apply_projector_volts1d_to_volts2d(np.ones(command_shape))
    else:
        projector_volts1d_to_volts2d, projector_volts2d_to_volts1d, mask_valid_actuators = None, None, None

    return projector_volts1d_to_volts2d, projector_volts2d_to_volts1d, mask_valid_actuators


def calculate_zero_response_sensor(supervisor):
    if supervisor.config_rl.env_rl['correct_zero_response']:
        supervisor.reset()
        supervisor.rtc.set_command(controller_index=0, com=np.zeros(supervisor.rtc.get_command(0).shape))
        supervisor.rtc.apply_control(0, comp_voltage=False)
        ncontrols = [0]
        wfs_trace = range(len(supervisor.config.p_wfss))

        if wfs_trace is not None:
            for w in wfs_trace:
                supervisor.wfs.raytrace(w, tel=supervisor.tel, reset=True)

                if not supervisor.config.p_wfss[w].open_loop and supervisor.dms is not None:
                    supervisor.wfs.raytrace(w, dms=supervisor.dms, ncpa=False, reset=False)
                supervisor.wfs.compute_wfs_image(w)

        for ncontrol in ncontrols:
            supervisor.rtc.do_centroids(ncontrol)
            supervisor.rtc.do_control(ncontrol)

        response = supervisor.rtc.get_err(0)

        supervisor.reset()
    else:
        response = None

    return response

def get_zernike_projector(s_pupil, i_pupil, lambda_target_0):
    import pripy
    from aotools import zernike
    # convenience function for trimmed target image

    # init Gerchberg Saxton:
    def rebin(a, shape):
        sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    im_width = 20
    pup_rebin = 4
    pup_width_gs = s_pupil.shape[0] // pup_rebin
    pup_gs = (rebin(s_pupil, [pup_width_gs, pup_width_gs]) > 0.5) * 1.0
    gerberch_saxton = pripy.GerchbergSaxton(pup_gs, lambda_target_0,
                                            i_pupil.shape[0] // pup_rebin, im_width, offset=None)

    # build zernike projector
    max_zernike = 20
    zern_array = zernike.zernikeArray(max_zernike, pup_width_gs, norm="rms")
    z_proj = zern_array[:, pup_gs == 1].T
    z_inv = np.linalg.solve(z_proj.T @ z_proj, z_proj.T)
    phase2zernike = z_inv[1:, :]  # we don't care about piston
    zernike2phase = z_proj[:, 1:]  # we don't care about piston

    print(phase2zernike.shape, zernike2phase.shape, gerberch_saxton, pup_gs.shape)

    return phase2zernike, zernike2phase, gerberch_saxton, pup_gs


def create_mask_valid_pixels_pyramid(supervisor):
    if supervisor.config_rl.env_rl['state_cnn_wfs_pyramid']:
        mask_valid_pixels_pyramid = np.zeros(supervisor.wfs.get_wfs_image(0).shape)
        x_val = np.array(supervisor.wfs._wfs.d_wfs[0].d_validsubsx)
        y_val = np.array(supervisor.wfs._wfs.d_wfs[0].d_validsubsy)

        mask_valid_pixels_pyramid[x_val, y_val] = 1
    else:
        mask_valid_pixels_pyramid = None
    return mask_valid_pixels_pyramid


def reference_slope_sac_creation(supervisor):
    if supervisor.config_rl.sac['reference_slope_sac']:
        # Reference slope agent
        previous_reference_slopes = np.array(supervisor.rtc.get_ref_slopes(0))
        reference_slope_agent, memory_reference_slope_agent = create_reference_slope_agent(supervisor)
        total_update_slope_agent = 0
    else:
        previous_reference_slopes = None
        reference_slope_agent, memory_reference_slope_agent = None, None
        total_update_slope_agent = None
    return previous_reference_slopes, reference_slope_agent, memory_reference_slope_agent, total_update_slope_agent


def create_reference_slope_agent(supervisor):
    from src.reinforcement_learning.single_agent_training.algorithms_single_agent.sac_single_agent import SAC as \
        AGENT
    from src.reinforcement_learning.single_agent_training.algorithms_single_agent.replay_memory_single_agent import\
        ReplayMemory
    num_inputs = supervisor.rtc.get_slopes(0)
    reference_slope_agent = AGENT(state_space=num_inputs.shape[0],
                                  action_space=num_inputs,
                                  config_rl=supervisor.config_rl,
                                  mask_valid_actuators=None)

    memory_reference_slope_agent = ReplayMemory(supervisor.config_rl.sac['memory_size'])

    return reference_slope_agent, memory_reference_slope_agent
