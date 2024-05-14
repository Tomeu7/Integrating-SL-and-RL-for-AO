import argparse
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from src.config import obtain_config_env_rl_default
from src.env_methods.env import AoEnv
import h5py
from src.global_cte import FOLDER_SAVE_DATA_UNET


def loguniform(low=0., high=1., size=None):
    return np.exp(np.random.uniform(low, high, size))

def next_phase_from_mirror(sup_, current_voltage_):
    """
    Raytrace through the atmosphere and apply the specified mirror shape (in volts) to the DM
    """

    sup_.rtc.set_command(controller_index=0, com=current_voltage_)
    sup_.rtc.apply_control(0, comp_voltage=False)

    sup_.target.raytrace(0, tel=sup_.tel, dms=sup_.dms)

    sup_.wfs.raytrace(0, tel=sup_.tel)
    if not sup_.config.p_wfss[0].open_loop and sup_.dms is not None:
        sup_.wfs.raytrace(0, dms=sup_.dms, ncpa=False, reset=False)
    sup_.wfs.compute_wfs_image(0)

    sup_.rtc.compute_slopes(0)
    sup_.rtc.do_control(0)

    sup_.target.comp_tar_image(0)
    sup_.target.comp_strehl(0)

    sup_.iter += 1

def poke_uniform_actuators(dm_shape_, args_):
    """
    Generates random uniform aberration and sets in the DM channel.
    :param var: If var is not None we generate a random normal command using that variance.
    Otherwise we use a log_uniform distribution."""

    new_var = loguniform(low=args_.min_loguniform,
                         high=args_.max_loguniform,
                         size=1)

    new_commands = np.random.normal(loc=0.0, scale=new_var, size=dm_shape_)

    new_commands[:-2] = np.clip(new_commands[:-2], a_min=-args_.clip, a_max=args_.clip).astype(np.float32)

    # TT multiplied by a factor
    new_commands[-2:] *= args_.multiplicative_factor_tt
    new_commands[-2:] = np.clip(new_commands[-2:], a_min=-args_.clip * args_.multiplicative_factor_tt, a_max=args_.clip * args_.multiplicative_factor_tt).astype(np.float32)

    return new_commands

def initialize_folders_and_compass():
    # Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--parameter_file', default="pyr_40x40_8m_gs_9.py")
    parser.add_argument('--data_size', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--train_dataset_percentage', help='Train/Evaluation split', default=0.8, type=float)
    parser.add_argument('--min_loguniform', type=float, help="e.g. 0.0001", default=0.5)
    parser.add_argument('--max_loguniform', type=float, help="e.g. 0.5", default=6)
    parser.add_argument('--clip', type=float, help="e.g. 1.0", default=1000.0)
    parser.add_argument("--dont_get_the_phase", action="store_true")
    parser.add_argument("--pyr_gpu_id", type=int, nargs='+', default=[0])
    parser.add_argument("--max_uniform", default=1.5, type=float)
    parser.add_argument("--min_uniform", default=400, type=float)
    parser.add_argument("--use_uniform_instead_of_loguniform", action="store_true")
    parser.add_argument("--multiplicative_factor_tt", default=1.0, type=float)
    parser.add_argument("--n_modes_filtered", default=100, type=int)
    args_ = parser.parse_args()
    conf_env = obtain_config_env_rl_default(args_.parameter_file, args_.n_modes_filtered)
    conf_env['control_tt'] = True  # ill need the projector
    env_ = AoEnv(config_env_rl=conf_env,
                 parameter_file=args_.parameter_file,
                 seed=args_.seed,
                 pyr_gpu_ids=args_.pyr_gpu_id)
    sup_ = env_.supervisor

    # Create directories for data
    path_to_data_ = FOLDER_SAVE_DATA_UNET + args_.dataset_name + "/"
    path_to_data_train_ = path_to_data_ + "train/"
    path_to_data_eval_ = path_to_data_ + "evaluation/"
    if not os.path.exists(path_to_data_train_):
        os.makedirs(path_to_data_train_)
        os.mkdir(path_to_data_eval_)

    np.random.seed(args_.seed)
    random.seed(args_.seed)

    log_ = pd.DataFrame(columns=['Max wfs', 'Min wfs', 'Max phase', 'Min phase', 'Max voltage', 'Min voltage', 'Max voltage linear subtracted', 'Min voltage linear subtracted'])
    log_normalize_ = pd.DataFrame(columns=['Max wfs', 'Min wfs', 'Max phase', 'Min phase', 'Max voltage', 'Min voltage'])
    wfs_max_, wfs_min_, phase_max_, phase_min_, voltage_max_, voltage_min_ =\
        0, float("inf"), 0, float("inf"), 0, float("inf")
    wfs_norm_max_, wfs_norm_min_ = 0, float("inf")
    voltage_linear_subtracted_max_, voltage_linear_subtracted_min_ = 0, float("inf")
    
    print("--INFO--")
    print("S PUPIL SHAPE ", sup_.get_s_pupil().shape)
    print("M PUPIL SHAPE ", sup_.get_m_pupil().shape)
    print("WFS IMAGE SHAPE", sup_.wfs.get_wfs_image(0).shape)
    if "40x40" in args_.parameter_file:
        print("PSF SHAPE", sup_.target.get_tar_image(0)[384:-384, 384:-384].shape)
    print("PHASE SHAPE", sup_.target.get_tar_phase(0).shape)

    return sup_, env_, env_.mask_valid_actuators, args_, path_to_data_, path_to_data_train_, path_to_data_eval_, log_, log_normalize_,\
           wfs_max_, wfs_min_, phase_max_, phase_min_, voltage_max_, voltage_min_,\
           wfs_norm_max_, wfs_norm_min_, voltage_linear_subtracted_max_, voltage_linear_subtracted_min_

def manage_norm_params(frame_, dont_get_the_phase_,
                       wfs_image_, wfs_max_, wfs_min_,
                       current_voltage_, voltage_max_, voltage_min_,
                       current_voltage_linear_subtracted_, voltage_linear_subtracted_max_, voltage_linear_subtracted_min_,
                       phase_, phase_max_, phase_min_,
                       wfs_norm_max_, wfs_norm_min_):
    wfs_image_norm_ = wfs_image_ / (wfs_image_.sum())
    # d) Statistics, only for training
    wfs_max_ = max(wfs_max_, wfs_image_.max())
    wfs_min_ = min(wfs_min_, wfs_image_.min())
    if dont_get_the_phase_:
        phase_max_ = None
        phase_min_ = None
    else:
        phase_max_ = max(phase_max_, phase_.max())
        phase_min_ = min(phase_min_, phase_.min())
    voltage_max_ = max(voltage_max_, current_voltage_.max())
    voltage_min_ = min(voltage_min_, current_voltage_.min())
    
    voltage_linear_subtracted_max_ = max(voltage_linear_subtracted_max_, current_voltage_linear_subtracted_.max())
    voltage_linear_subtracted_min_ = min(voltage_linear_subtracted_min_, current_voltage_linear_subtracted_.min())
    wfs_norm_max_ = max(wfs_norm_max_, wfs_image_norm_.max())
    wfs_norm_min_ = min(wfs_norm_min_, wfs_image_norm_.min())

    if frame % 500 == 0:
        if dont_get_the_phase_:
            phase_to_print = None
        else:
            phase_to_print = phase_.std()
        print(
            "Frame {} Phase rms {} "
            "Wfs max {} Wfs min {} "
            "Wfs norm max {} Wfs norm min {} "
            "voltage max {} voltage min {} "
            "voltage linear subtracted max {} voltage linear subtracted min {} "
            "Phase max {} Phase min {}".format(frame_,
                                               phase_to_print,
                                               wfs_max_, wfs_min_,
                                               wfs_norm_max_, wfs_norm_min_,
                                               voltage_max_, voltage_min_,
                                               voltage_linear_subtracted_max_, voltage_linear_subtracted_min_,
                                               phase_max_, phase_min_))

    return wfs_max_, wfs_min_, voltage_max_, voltage_min_, phase_max_, phase_min_, wfs_norm_max_, wfs_norm_min_, voltage_linear_subtracted_max_, voltage_linear_subtracted_min_

def save_data(path_, wfs_image_, phase_, commands_to_save_, current_voltage_linear_subtracted_):
    file = h5py.File(path_, 'w')
    file.create_dataset('arr_0', data=wfs_image_)
    file.create_dataset('arr_1', data=phase_)
    file.create_dataset('arr_2', data=commands_to_save_)
    file.create_dataset('arr_3', data=current_voltage_linear_subtracted_)
    file.close()

if __name__ == "__main__":
    # Assert we are in hardware mode
    sup, env, mask_valid_actuators, args, path_to_data, path_to_data_train, path_to_data_eval, log, log_normalize,\
        wfs_max, wfs_min, phase_max, phase_min, voltage_max, voltage_min,\
        wfs_norm_max, wfs_norm_min, voltage_linear_subtracted_max, voltage_linear_subtracted_min = initialize_folders_and_compass()

    dm_shape = sup.rtc.get_command(0).shape
    pupil_mask = sup.get_s_pupil()
    np.save(path_to_data + "mask_valid_commands.npy",mask_valid_actuators)
    np.save(path_to_data + "pupil.npy", pupil_mask)
    linear_reconstructor = sup.rtc.get_command_matrix(0)
    wfs_xpos, wfs_ypos = sup.config.p_wfs0._validsubsx, sup.config.p_wfs0._validsubsy
    np.save(path_to_data + "linear_reconstructor.npy", sup.rtc.get_command_matrix(0))
    np.save(path_to_data + "wfs_xpos.npy", wfs_xpos)
    np.save(path_to_data + "wfs_ypos.npy", wfs_ypos)
    np.save(path_to_data + "wfs_mask.npy", env.wfs_mask)

    for frame in tqdm(range(args.data_size)):

        # a) Input random perturbation
        current_voltage = poke_uniform_actuators(dm_shape, args)

        # b) Move mirror
        next_phase_from_mirror(sup, current_voltage_=current_voltage)

        # c) Save it in training or evaluation folder
        wfs_image = sup.wfs.get_wfs_image(0)

        commands_pzt = current_voltage[:-2]
        commands_tt = env.tt2pzt(current_voltage[-2:])
        commands_to_save = sup.apply_projector_volts1d_to_volts2d(commands_pzt + commands_tt)

        slopes = sup.rtc.get_slopes(0)
        linear_reconstruction = linear_reconstructor.dot(slopes)
        linear_reconstruction_tt = sup.apply_projector_volts1d_to_volts2d(env.tt2pzt(linear_reconstruction[-2:]))
        linear_reconstruction_pzt = sup.apply_projector_volts1d_to_volts2d(linear_reconstruction[:-2])
        current_voltage_linear_subtracted = commands_to_save - (linear_reconstruction_pzt + linear_reconstruction_tt)
        
        if args.dont_get_the_phase:
            phase = 0.
        else:
            phase = sup.target.get_tar_phase(0)
            phase = np.multiply(phase, pupil_mask)
            phase -= phase[pupil_mask == 1].mean()
        if frame < args.train_dataset_percentage * args.data_size:
            save_data(path_to_data_train + "0_" + str(frame) + '.hdf5', wfs_image, phase, commands_to_save, current_voltage_linear_subtracted)
            wfs_max, wfs_min, voltage_max, voltage_min, phase_max, phase_min, wfs_norm_max, wfs_norm_min, voltage_linear_subtracted_max, voltage_linear_subtracted_min =\
                manage_norm_params(frame,
                                   args.dont_get_the_phase,
                                   wfs_image,
                                   wfs_max,
                                   wfs_min,
                                   current_voltage,
                                   voltage_max,
                                   voltage_min,
                                   current_voltage_linear_subtracted,
                                   voltage_linear_subtracted_max,
                                   voltage_linear_subtracted_min,
                                   phase,
                                   phase_max,
                                   phase_min,
                                   wfs_norm_max,
                                   wfs_norm_min)
        else:
            save_data(path_to_data_eval + "0_" + str(frame) + '.hdf5', wfs_image, phase, commands_to_save, current_voltage_linear_subtracted)


    log = log.append({'Max wfs': wfs_max, 'Min wfs': wfs_min,
                      'Max phase': phase_max, 'Min phase': phase_min,
                      'Max voltage': voltage_max, 'Min voltage': voltage_min,
                      'Max voltage linear subtracted':voltage_linear_subtracted_max, 'Min voltage linear subtracted':voltage_linear_subtracted_min, 
                      'Max loguniform': args.max_loguniform, 'Min loguniform': args.min_loguniform},
                     ignore_index=True)
    log.to_csv(path_to_data + "info.csv", index=False)

    log_normalize = log_normalize.append({'Max wfs': wfs_norm_max, 'Min wfs': wfs_norm_min,
                                          'Max phase': phase_max, 'Min phase': phase_min,
                                          'Max voltage': voltage_max, 'Min voltage': voltage_min,
                                          'Max loguniform': args.max_loguniform, 'Min loguniform': args.min_loguniform},
                                         ignore_index=True)

    log_normalize.to_csv(path_to_data + "info_normalize_flux.csv", index=False)