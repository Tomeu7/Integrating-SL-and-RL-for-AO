import os
import json
import pandas as pd
import numpy as np

"""
Helper methods
a) set_r0_and_gain -> changes r0 and gain of lin and non-lin rec based on args
b) manage_atmospheric_conditions_3_layers -> changes atmos params based on args
c) save_configs_to_directory -> saves config to directory of results
d) manage_metrics -> updates metrics
e) add_metrics_to_dataset_and_print -> prints and adds metrics to dataset
"""

def set_r0_and_gain(r0: float, parameter_file:str, env_, gains_linear:float, gains_non_linear:float, path_to_gains_json:str="data/gains_results/gains_per_parameter_files_and_unet.json", normalization_noise_value_linear:float=-1):
    """
    From a dict of previous obtained gains we set gains for linear and non_linear reconstructor
    Args:
        r0: Current value of r0
        parameter_file: Current parameter file
        env_: The environment object
        gains_linear: gain linear reconstruction
        gains_non_linear: gain non-linear reconstruction
        path_to_gains_json: path to previously optimised gains
    Returns: None
    """

    # Key for linear in gain will depend
    if normalization_noise_value_linear >= 0:
        print("normalization_noise_value_linear: ", normalization_noise_value_linear,
              ", greater than 0, changing keys of dict for gain")
        linear_dict_key = "Linear_norm_" + str(normalization_noise_value_linear)
        combination_with_linear_lin_dict_key = "gain_linear_combination_" + str(normalization_noise_value_linear)
        combination_with_linear_nonlin_dict_key = "gain_non_linear_combination_" + str(normalization_noise_value_linear)
    else:
        linear_dict_key = "Linear"
        combination_with_linear_lin_dict_key = "gain_linear_combination"
        combination_with_linear_nonlin_dict_key = "gain_non_linear_combination"

    if r0 is not None:
        env_.supervisor.atmos.set_r0(r0)

    # Override step if first controller in param file is "geo":
    if env_.supervisor.config.p_controllers[0].get_type() != "geo":
        with open(path_to_gains_json, 'r') as file:
            dict_gains = json.load(file)[parameter_file[:-3]]

        print(dict_gains.keys())
        gain_linear = dict_gains[linear_dict_key][str(r0)]
        env_.supervisor.set_gain(gain_linear)
        if hasattr(env_, "model"):
            if env_.model is not None:
                env_.gain_factor_linear = dict_gains[env_.unet_name[:-4]][str(r0)][combination_with_linear_lin_dict_key],
                env_.gain_factor_unet = dict_gains[env_.unet_name[:-4]][str(r0)][combination_with_linear_nonlin_dict_key]
            if gains_non_linear > -1:
                env_.gain_factor_unet = gains_non_linear
        if gains_linear > -1:
            env_.gain_factor_linear = gains_linear
        
        print("-Gains; linear_comb {} non_linear_comb {} linear only {}".format(env_.gain_factor_linear,
                                                                                env_.gain_factor_unet,
                                                                                gain_linear))
    else:
        print("Using geometric controlling, gains not set")

def manage_atmospheric_conditions_3_layers(args, total_step: int, env, agent: int, controller_type: str) -> None:
    """
        Manage atmospheric conditions
        Args:
            args: arguments object
            step: current step
            total_step: current TOTAL step
            env: environment object
            agent: RL model
            controller_type: which controller are we using
        """

    if total_step == args.change_atmospheric_conditions_1_at_step:
        print("Changing atmos conditions 1 v2")
        env.supervisor.atmos.set_wind(0, winddir=90)
        env.supervisor.atmos.set_wind(1, winddir=110)
        env.supervisor.atmos.set_wind(2, winddir=270)
        if args.reset_replay_buffer_when_change_atmos:
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos:
            agent.reset_optimizer()
        if args.reset_replay_buffer_when_change_atmos_10k and controller_type == "RL":
            agent.replay_buffer.reset_to_10k()
            print("Len replay buffer", len(agent.replay_buffer))
    elif total_step == args.change_atmospheric_conditions_2_at_step:
        print("Changing atmos conditions 2 v2")
        env.supervisor.atmos.set_r0(r0=0.08)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()
        if args.reset_replay_buffer_when_change_atmos_10k and controller_type == "RL":
            agent.replay_buffer.reset_to_10k()
            print("Len replay buffer", len(agent.replay_buffer))
    elif total_step == args.change_atmospheric_conditions_3_at_step:  # error at 3
        print("Changing atmos conditions 3 v2")
        env.supervisor.atmos.set_r0(r0=0.16)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()
        if args.reset_replay_buffer_when_change_atmos_10k and controller_type == "RL":
            agent.replay_buffer.reset_to_10k()
            print("Len replay buffer", len(agent.replay_buffer))
    elif total_step == args.change_atmospheric_conditions_4_at_step:
        print("Changing atmos conditions 4 v2")
        env.supervisor.atmos.set_wind(0, windspeed=30)
        env.supervisor.atmos.set_wind(1, windspeed=30)
        env.supervisor.atmos.set_wind(2, windspeed=40)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()
        if args.reset_replay_buffer_when_change_atmos_10k and controller_type == "RL":
            agent.replay_buffer.reset_to_10k()
            print("Len replay buffer", len(agent.replay_buffer))
    elif total_step == args.change_atmospheric_conditions_5_at_step:
        print("Changing atmos conditions 5 v2")
        env.supervisor.atmos.set_wind(0, winddir=90)
        env.supervisor.atmos.set_wind(1, winddir=110)
        env.supervisor.atmos.set_wind(2, winddir=270)
        env.supervisor.atmos.set_r0(r0=0.08)
        env.supervisor.atmos.set_wind(0, windspeed=30)
        env.supervisor.atmos.set_wind(1, windspeed=30)
        env.supervisor.atmos.set_wind(2, windspeed=40)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()
        if args.reset_replay_buffer_when_change_atmos_10k and controller_type == "RL":
            agent.replay_buffer.reset_to_10k()
            print("Len replay buffer", len(agent.replay_buffer))
    elif total_step == args.change_atmospheric_conditions_6_at_step:
        print("Changing atmos conditions 6")
        env.supervisor.atmos.set_wind(0, winddir=180)
        env.supervisor.atmos.set_wind(1, winddir=180)
        env.supervisor.atmos.set_wind(2, winddir=180)
        if args.reset_replay_buffer_when_change_atmos and controller_type == "RL":
            agent.replay_buffer.reset()
        if args.reset_adam_when_change_atmos and controller_type == "RL":
            agent.reset_optimizer()
        if args.reset_replay_buffer_when_change_atmos_10k and controller_type == "RL":
            agent.replay_buffer.reset_to_10k()
            print("Len replay buffer", len(agent.replay_buffer))

def save_configs_to_directory(dir_path_: dict, config_env_rl_: dict, config_agent_: dict) -> None:
    """
    Saves config of env_rl and agent
    :param dir_path_: path to save
    :param config_env_rl_: current config_env_rl
    :param config_agent_: current_config_agent
    """
    if not os.path.exists(dir_path_):
        os.makedirs(dir_path_)

    with open(os.path.join(dir_path_, "config_env_rl.json"), "w") as f:
        json.dump(config_env_rl_, f, indent=4)

    with open(os.path.join(dir_path_, "config_agent.json"), "w") as f:
        json.dump(config_agent_, f, indent=4)


def add_metrics_to_dataset_and_print(df: pd.DataFrame,
                                     r_total_train: float,
                                     r_pzt_total_train: float,
                                     r_tt_total_train: float,
                                     sr_se_total: float,
                                     step: int,
                                     num_episode: int,
                                     total_step: int,
                                     delta_time: float,
                                     seed: int,
                                     sr_le: float,
                                     s_dict: dict,
                                     max_tt_value: float,
                                     min_tt_value: float,
                                     max_pzt_value: float,
                                     min_pzt_value: float,
                                     count_pzt_surpass: int,
                                     count_tt_surpass: int,
                                     a_pzt_total: np.ndarray,
                                     a_tt_total: np.ndarray,
                                     number_clipped_actuators: int,
                                     com_pen: np.ndarray,
                                     ) -> pd.DataFrame:
    """
    Adds metrics to pd.DataFrame and prints
    :param df: current metrics dataframe
    :param r_total_train: totalreward obtained during period of observation
    :param r_pzt_total_train: total reward PZT component
    :param r_tt_total_train: total reward TT component
    :param sr_se_total: total sr se obtaiend during period of observation
    :param step: current step in episode
    :param num_episode: current episode
    :param total_step: current step
    :param delta_time: time it takes per episode
    :param seed: current seed in environment
    :param sr_le: sr le during the period of observation
    :param s_dict: dictionary of state
    :param max_tt_value: max value of tt
    :param min_tt_value: min value of tt
    :param max_pzt_value: max value of pzt
    :param min_pzt_value: min value of pzt
    :param count_pzt_surpass: count of pzt clipped
    :param count_tt_surpass: count of tt clipped
    :param a_pzt_total: current action pzt
    :param a_tt_total: current action tt
    :param reward_to_print: reward
    """
    r_total_train /= step
    r_pzt_total_train /= step
    r_tt_total_train /= step
    a_pzt_total /= step
    a_tt_total /= step
    sr_se_total /= step

    print("Episode:", num_episode,
          "Total steps:", total_step,
          "Episode steps:", step,
          "Seed:", seed,
          "R total:", round(r_total_train, 5),
          "Rec pzt:", round(r_pzt_total_train, 5),
          "Rec tt:", round(r_tt_total_train, 5),
          "Time:", round(delta_time, 5),
          "SR LE:", round(sr_le, 5),
          "SR SE:", round(sr_se_total, 5),
          "Command Pzt Min", min_pzt_value,
          "Command Pzt Max", max_pzt_value,
          "Count PZT surpassing clip", count_pzt_surpass,
          "Command TT Min", min_tt_value,
          "Command TT Max", max_tt_value,
          "Count TT surpassing clip", count_tt_surpass,
          "a pzt",  round(a_pzt_total, 5),
          "a tt",  round(a_tt_total, 5),
          "number_clipped_actuators:", number_clipped_actuators,
          "com_pen: ", round(com_pen, 7)
          )

    new_row = {"Episode": num_episode,
               "Total steps": total_step,
               "Episode steps": step,
               "Seed": seed,
               "R total": round(r_total_train, 5),
               "Rec pzt": round(r_pzt_total_train, 5),
               "Rec tt": round(r_tt_total_train, 5),
               "Time:": round(delta_time, 5),
               "SR LE": round(sr_le, 5),
               "SR SE": round(sr_se_total, 5),
               "Command Pzt Min": min_pzt_value,
               "Command Pzt Max": max_pzt_value,
               "Count PZT surpassing clip": count_pzt_surpass,
               "Command TT Min": min_tt_value,
               "Command TT Max": max_tt_value,
               "Count TT surpassing clip": count_tt_surpass,
               "a pzt":  round(a_pzt_total, 5),
               "a tt":  round(a_tt_total, 5)}
    for key, item in s_dict.items():
        new_row[key + "_min"] = item.min()
        new_row[key + "_max"] = item.max()
    df = pd.concat([df, pd.DataFrame([new_row])]).reset_index(drop=True)

    return df


def manage_reward_and_metrics(r,
                              r_total_train,
                              r_pzt_total_train,
                              r_tt_total_train,
                              rec_for_reward,
                              sr_se,
                              sr_se_total,
                              command,
                              clip_value,
                              max_tt_value,
                              min_tt_value,
                              max_pzt_value,
                              min_pzt_value,
                              count_tt_surpass,
                              count_pzt_surpass,
                              a_pzt,
                              a_tt,
                              a_pzt_total,
                              a_tt_total):
    """
    Updates metrics
    :param r: current reward
    :param r_total_train: totalreward obtained during period of observation
    :param r_pzt_total_train: total reward PZT component
    :param r_tt_total_train: total reward TT component
    :param rec_for_reward: reconstruction of phase
    :param sr_se: current sr_se
    :param sr_se_total: current metric of sr_se
    :param command: current command
    :param clip_value: value of clipping
    :param max_tt_value: max value of tt
    :param min_tt_value: min value of tt
    :param max_pzt_value: max value of pzt
    :param min_pzt_value: min value of pzt
    :param count_pzt_surpass: count of pzt clipped
    :param count_tt_surpass: count of tt clipped
    :param a_pzt: current action in PZT space
    :param a_tt: current action in TT space
    :param a_pzt_total: current action pzt
    :param a_tt_total: current action tt
    :return:
    """
    if isinstance(r, np.ndarray):
        r_total_train += r.mean()
        rec_pzt = -np.square(rec_for_reward[:-2]).mean()
        rec_tt = -np.square(rec_for_reward[-2:]).mean()
        r_pzt_total_train += rec_pzt
        r_tt_total_train += rec_tt
        if a_pzt is not None:
            a_pzt_total += np.abs(a_pzt).mean()
        if a_tt is not None:
            a_tt_total += np.abs(a_tt).mean()
    else:
        r_total_train += r

    sr_se_total += sr_se

    tt_command_abs = np.abs(command[-2:])
    pzt_command_abs = np.abs(command[:-2])
    max_tt_value = max(max_tt_value, tt_command_abs.max())
    min_tt_value =  min(min_tt_value, tt_command_abs.min())
    max_pzt_value = max(max_pzt_value, pzt_command_abs.max())
    min_pzt_value =  min(min_pzt_value, pzt_command_abs.min())
    if (tt_command_abs > clip_value).any():
        count_tt_surpass += 1
    if (pzt_command_abs > clip_value).any():
        count_pzt_surpass += 1

    return r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, max_tt_value, min_tt_value,\
           max_pzt_value, min_pzt_value, count_tt_surpass, count_pzt_surpass, a_pzt_total, a_tt_total
