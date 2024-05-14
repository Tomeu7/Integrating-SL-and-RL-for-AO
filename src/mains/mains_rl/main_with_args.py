from src.env_methods.env import AoEnv
from src.agent.utils import DelayedMDP
import numpy as np
import time
import torch
import random
from src.config import obtain_config_env_rl
from src.config import obtain_config_agent
import argparse
import pandas as pd
import os
from src.env_methods.env_non_linear import AoEnvNonLinear
from src.env_methods.env_with_phase import AoEnvWithPhase
from src.mains.mains_rl.helper import manage_atmospheric_conditions_3_layers, set_r0_and_gain, save_configs_to_directory, add_metrics_to_dataset_and_print, manage_reward_and_metrics

def create_args():
    parser = argparse.ArgumentParser(description="Your program description here.")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--number_of_modes_filtered", default=100, type=int)
    parser.add_argument("--use_mask_influence_function", action="store_true")
    parser.add_argument("--device_rl", type=int, default=0)
    parser.add_argument("--device_compass", type=int, nargs='+', default=[0])
    parser.add_argument("--agent_type", type=str, default="sac", choices=["sac", "td3"])
    parser.add_argument("--controller_type", type=str, default="RL", choices=["RL", "Integrator", "UNet+Linear",
                                                                              "Phase", "Phasev2", "UNet+Linearv2"])
    parser.add_argument("--path_to_gains_json", type=str, default="data/gains_results/gains_per_parameter_files_and_unet.json")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="correction", choices=["only_rl", "correction"])
    parser.add_argument("--evaluation_after", type=int, default=30)
    parser.add_argument("--evaluation_after_steps", type=int, default=9999999)
    parser.add_argument("--total_episodes", type=int, default=1)
    parser.add_argument("--continuous_task", action="store_true")
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--replay_buffer_size", default=50000, type=int)
    parser.add_argument("--entropy_factor", default=1, type=float)
    parser.add_argument("--no_automatic_entropy_tuning", action="store_true")
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--no_s_dm", action="store_true")
    parser.add_argument("--s_dm_residual", action="store_true")
    parser.add_argument("--s_dm_residual_rl", action="store_true")
    parser.add_argument("--number_of_previous_s_dm", type=int, default=3)
    parser.add_argument("--number_of_previous_s_dm_residual_rl", type=int, default=0)
    parser.add_argument("--filter_commands", action="store_true")
    parser.add_argument("--command_clip_value", type=float, default=1000.0)
    parser.add_argument("--integrator_exploration_with_only_rl_for", default=-1, type=int)
    parser.add_argument("--noise_for_exploration", type=float, default=-1)
    parser.add_argument("--action_scale", type=float, default=10.0)
    parser.add_argument("--no_pure_deterministic", action="store_false")
    parser.add_argument("--running_norm", action="store_true")
    parser.add_argument("--running_norm_mode", type=str, default="array", choices=["scalar", "array", "arrayv2"])
    parser.add_argument('--normalise_for_reward', action="store_true")
    parser.add_argument("--correction_pzt_only_rl_tt", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--delayed_assignment", type=int, default=1)
    parser.add_argument("--use_contrastive_replay_memory", action="store_true")
    parser.add_argument("--latest_transitions_count", type=int, default=128)
    # for crossqstyle
    parser.add_argument("--crossqstyle", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr_alpha", type=float, default=0.0003)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--beta1_alpha", type=float, default=0.9)
    parser.add_argument("--beta2_alpha", type=float, default=0.999)
    parser.add_argument("--update_critic_to_policy_ratio", type=int, default=1)
    parser.add_argument("--hidden_dim_critic", type=int, default=64)
    parser.add_argument("--hidden_dim_actor", type=int, default=64)
    parser.add_argument("--num_layers_critic", type=int, default=2)
    parser.add_argument("--num_layers_actor", type=int, default=3)
    parser.add_argument("--use_batch_norm_policy", action="store_true")
    parser.add_argument("--use_batch_norm_critic", action="store_true")
    parser.add_argument("--activation_critic", type=str, default="relu")
    parser.add_argument("--activation_actor", type=str, default="relu")
    parser.add_argument("--bn_momentum", type=float, default=0.01)
    parser.add_argument("--bn_mode", type=str, default="bn")
    parser.add_argument("--tau", type=float, default=0.005)
    # for state prediction in critic
    parser.add_argument("--state_pred_layer", type=int, default=-1)
    parser.add_argument("--state_pred_lambda", type=float, default=0.1)
    # for skip connections
    parser.add_argument("--include_skip_connections_critic", action="store_true")
    parser.add_argument("--include_skip_connections_actor", action="store_true")

    # gains
    parser.add_argument("--gain_linear", type=float, default=-1)
    parser.add_argument("--gain_non_linear", type=float, default=-1)
    parser.add_argument("--gain_phase", type=float, default=-1)

    parser.add_argument("--use_second_version_of_modal_basis", action="store_true")

    # Parameter file parameters
    parser.add_argument("--parameter_file", default="pyr_40x40_8m.py", type=str)
    parser.add_argument("--r0", default=0.16, type=float)
    parser.add_argument("--change_atmospheric_conditions_1_at_step", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_2_at_step", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_3_at_step", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_4_at_step", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_5_at_step", type=int, default=-1)
    parser.add_argument("--change_atmospheric_conditions_6_at_step", type=int, default=-1)
    parser.add_argument("--reset_replay_buffer_when_change_atmos", action="store_true")
    parser.add_argument("--reset_adam_when_change_atmos", action="store_true")
    parser.add_argument("--reset_replay_buffer_when_change_atmos_10k", action="store_true")
    parser.add_argument("--back_to_train_at", type=int, default=-1)
    parser.add_argument("--back_to_eval_at", type=int, default=-1)
    parser.add_argument("--back_to_train_at_step", type=int, default=-1)
    parser.add_argument("--back_to_eval_at_step", type=int, default=-1)
    parser.add_argument("--stop_training_after", type=int, default=100000)
    parser.add_argument("--filter_everywhere", action="store_true")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--normalization_noise_value_linear", default=-1, type=float)
    parser.add_argument("--value_commands_deltapos", default=-1 , type=float)
    
    # Unet
    parser.add_argument("--unet_name", default=None)
    parser.add_argument("--unet_dir", default=None)
    parser.add_argument("--device_unet", default=0)
    parser.add_argument("--s_dm_residual_non_linear", action="store_true")
    parser.add_argument("--number_of_previous_s_dm_residual_non_linear", default=0, type=int)
    parser.add_argument("--s_dm_residual_non_linear_tt", action="store_true")
    parser.add_argument("--number_of_previous_s_dm_residual_non_linear_tt", default=0, type=int)
    parser.add_argument("--normalization_noise_unet", action="store_true")
    parser.add_argument("--subtract_mean_from_phase", action="store_true")
    parser.add_argument("--normalization_noise_value_unet", default=0, type=float)
    parser.add_argument("--use_wfs_mask", action="store_true")
    # TT
    parser.add_argument("--control_tt", action="store_true")
    parser.add_argument("--s_dm_tt", action="store_true")
    parser.add_argument("--s_dm_residual_tt", action="store_true")
    parser.add_argument("--number_of_previous_s_dm_residual_tt", default=0, type=int)
    parser.add_argument("--number_of_previous_s_dm_tt", default=0, type=int)
    parser.add_argument("--joint_tt_into_s_dm", action="store_true")
    parser.add_argument("--joint_tt_into_s_dm_residual", action="store_true")
    parser.add_argument("--scaling_for_residual_tt", type=float, default=1.0)
    # Phase
    parser.add_argument("--s_dm_residual_phase", action="store_true")
    parser.add_argument("--s_dm_residual_phase_tt", action="store_true")
    parser.add_argument("--number_of_previous_s_dm_residual_phase", default=0, type=int)
    parser.add_argument("--number_of_previous_s_dm_residual_phase_tt", default=0, type=int)
    # Integrator
    parser.add_argument("--leak", type=float, default=-1)
    parser.add_argument("--leaky_integrator_for_rl", action="store_true")
    parser.add_argument("--reset_when_clip", action="store_true")
    parser.add_argument("--reduce_gain_tt_to", type=float, default=-1)
    # Load policy
    parser.add_argument("--starting_policy_path", type=str, default=None)
    args_ = parser.parse_args()
    return args_


def choose_evaluation(args_, total_step_, step_, episode_, evaluation_):
    if total_step_ == args_.evaluation_after_steps:
        evaluation_ = True
        print("0. Changing evaluation to:", evaluation_)
    elif episode_ == args_.evaluation_after and step_ == 0:
        evaluation_ = True
        print("1. Changing evaluation to:", evaluation_)
    elif (episode_ == args_.back_to_train_at and step_ == 0) or total_step_ == args_.back_to_train_at_step:
        evaluation_ = False
        print("2. Changing evaluation to:", evaluation_)
    elif (episode_ == args_.back_to_eval_at and step_ == 0) or total_step_ == args_.back_to_eval_at_step:
        evaluation_ = True
        print("3. Changing evaluation to:", evaluation_)
    else:
        evaluation_ = evaluation_
    return evaluation_


def initialise(args_):
    # Variables
    seed_ = args_.seed
    controller_type_ = args_.controller_type
    # sh_10x10_2m.py or sh_40x40_8m.py
    parameter_file_ = args_.parameter_file
    steps_per_episode_ = args_.steps_per_episode
    only_reset_dm_ = args_.continuous_task
    pure_deterministic_ = args_.no_pure_deterministic
    device_sac_ = args_.device_rl
    device_compass_ = args_.device_compass

    if args_.integrator_exploration_with_only_rl_for > -1:
        assert args_.noise_for_exploration > -1, "If exploration activated you must provide noise value"
    if any([args_.s_dm_residual_non_linear, args_.s_dm_residual_non_linear_tt]):
        assert args_.unet_dir is not None and args_.unet_name is not None
    # Configs
    config_env_rl_ = obtain_config_env_rl(parameter_file=parameter_file_,
                                            number_of_modes_filtered=args_.number_of_modes_filtered,
                                            use_mask_influence_function=args_.use_mask_influence_function,
                                            mode=args_.mode,
                                            control_tt=args_.control_tt,
                                            s_dm_residual=args_.s_dm_residual,
                                            s_dm_tt=args_.s_dm_tt,
                                            s_dm_residual_tt=args_.s_dm_residual_tt,
                                            s_dm=not args_.no_s_dm,
                                            s_dm_residual_rl=args_.s_dm_residual_rl,
                                            number_of_previous_s_dm=args_.number_of_previous_s_dm,
                                            number_of_previous_s_dm_residual_rl=args_.number_of_previous_s_dm_residual_rl,
                                            number_of_previous_s_dm_residual_tt=args_.number_of_previous_s_dm_residual_tt,
                                            number_of_previous_s_dm_tt=args_.number_of_previous_s_dm_tt,
                                            s_dm_residual_non_linear=args_.s_dm_residual_non_linear,
                                            number_of_previous_s_dm_residual_non_linear=args_.number_of_previous_s_dm_residual_non_linear,
                                            s_dm_residual_non_linear_tt=args_.s_dm_residual_non_linear_tt,
                                            number_of_previous_s_dm_residual_non_linear_tt=args_.number_of_previous_s_dm_residual_non_linear_tt,
                                            filter_commands=args_.filter_commands,
                                            command_clip_value=args_.command_clip_value,
                                            joint_tt_into_s_dm=args_.joint_tt_into_s_dm,
                                            joint_tt_into_s_dm_residual=args_.joint_tt_into_s_dm_residual,
                                            scaling_for_residual_tt=args_.scaling_for_residual_tt,
                                            action_scale=args_.action_scale,
                                            correction_pzt_only_rl_tt=args_.correction_pzt_only_rl_tt,
                                            delayed_assignment=args_.delayed_assignment,
                                            value_commands_deltapos=args_.value_commands_deltapos,
                                            use_second_version_of_modal_basis=args_.use_second_version_of_modal_basis,
                                            leaky_integrator_for_rl=args_.leaky_integrator_for_rl,
                                            leak=args_.leak,
                                            reset_when_clip=args_.reset_when_clip,
                                            reduce_gain_tt_to=args_.reduce_gain_tt_to,
                                            no_subtract_mean_from_phase=not args_.subtract_mean_from_phase,
                                            reward_scale=args_.reward_scale,
                                            running_norm=args_.running_norm,
                                            running_norm_mode=args_.running_norm_mode,
                                            normalise_for_reward=args_.normalise_for_reward)
    # For environment
    config_agent_ = obtain_config_agent(agent_type=args_.agent_type,
                                        replay_buffer_size=args_.replay_buffer_size,
                                        entropy_factor=args_.entropy_factor,
                                        automatic_entropy_tuning=not args_.no_automatic_entropy_tuning,
                                        alpha=args_.alpha,
                                        gamma=args_.gamma,
                                        filter_everywhere=args_.filter_everywhere,
                                        crossqstyle=args_.crossqstyle,
                                        lr=args_.lr,
                                        lr_alpha=args_.lr_alpha,
                                        beta1=args_.beta1,
                                        beta2=args_.beta2,
                                        beta1_alpha=args_.beta1_alpha,
                                        beta2_alpha=args_.beta2_alpha,
                                        update_critic_to_policy_ratio=args_.update_critic_to_policy_ratio,
                                        hidden_dim_critic=args_.hidden_dim_critic,
                                        hidden_dim_actor=args_.hidden_dim_actor,
                                        use_batch_norm_policy=args_.use_batch_norm_policy,
                                        use_batch_norm_critic=args_.use_batch_norm_critic,
                                        activation_critic=args_.activation_critic,
                                        activation_policy=args_.activation_actor,
                                        bn_momentum=args_.bn_momentum,
                                        bn_mode=args_.bn_mode,
                                        state_pred_layer=args_.state_pred_layer,
                                        state_pred_lambda=args_.state_pred_lambda,
                                        include_skip_connections_critic=args_.include_skip_connections_critic,
                                        include_skip_connections_actor=args_.include_skip_connections_actor,
                                        num_layers_critic=args_.num_layers_critic,
                                        num_layers_actor=args_.num_layers_actor,
                                        tau=args_.tau
                                        )


    # Setting seed in libraries - seed in the env will be set up when we create the object
    torch.manual_seed(seed_)
    np.random.seed(seed_)
    random.seed(seed_)
    if pure_deterministic_:  # maybe this make it slower, only needed if we use convolutions
        torch.backends.cudnn.deterministic = True

    # Environment
    if args_.unet_dir is not None and args_.unet_name is not None:
        # Environment
        env_ = AoEnvNonLinear(unet_dir=args_.unet_dir,
                              unet_name=args_.unet_name,
                              unet_type="volts",
                              only_predict_non_linear=False,
                              device_unet="cuda:" + str(args_.device_unet),
                              gain_factor_unet=None,  # we set it up later
                              gain_factor_linear=None,  # we set it up later
                              normalize_flux=False,
                              normalization_095_005=True,
                              config_env_rl=config_env_rl_,
                              parameter_file=parameter_file_,
                              seed=seed_,
                              device_compass=device_compass_,
                              normalization_noise_unet=args_.normalization_noise_unet,
                              normalization_noise_value_unet=args_.normalization_noise_value_unet,
                              use_wfs_mask_unet=args_.use_wfs_mask,
                              normalization_noise_value_linear=args_.normalization_noise_value_linear)
    elif args_.s_dm_residual_phase or controller_type_ == "Phase":
        config_env_rl_['s_dm_residual_phase'] = True
        config_env_rl_['number_of_previous_s_dm_residual_phase'] = args_.number_of_previous_s_dm_residual_phase
        config_env_rl_['s_dm_residual_phase_tt'] = args_.s_dm_residual_phase_tt
        config_env_rl_['number_of_previous_s_dm_residual_phase_tt'] = args_.number_of_previous_s_dm_residual_phase_tt

        # Environment
        gain_phase = 0.9 if args_.gain_phase == -1 else args_.gain_phase
        env_ = AoEnvWithPhase(gain_factor_phase=gain_phase,  # we set it up later
                              config_env_rl=config_env_rl_,
                              parameter_file=parameter_file_,
                              seed=seed_,
                              device=device_compass_)
    else:
        env_ = AoEnv(config_env_rl=config_env_rl_,
                     parameter_file=parameter_file_,
                     seed=seed_,
                     pyr_gpu_ids=device_compass_,
                     normalization_noise_value_linear=args_.normalization_noise_value_linear)

    set_r0_and_gain(args_.r0, args_.parameter_file, env_, args_.gain_linear, args_.gain_non_linear, args_.path_to_gains_json, args_.normalization_noise_value_linear)

    # Agent
    if controller_type_ == "RL":
        if config_agent_['agent_type'] == "td3":
            if config_agent_['shared_layers']:
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            if config_agent_['shared_layers']:
                raise NotImplementedError
            else:
                from src.agent.sac import Sac as Agent
        agent_ = Agent(config_agent=config_agent_,
                       state_channels=env_.state_size_channel_0,
                       action_shape=env_.action_2d_shape,
                       device=device_sac_,
                       mask_valid_actuators=env_.mask_valid_actuators,
                       two_output_critic=not config_env_rl_['joint_tt_into_reward'] and config_env_rl_['control_tt'],
                       two_output_actor=config_env_rl_['separate_tt_into_two_actions'] and config_env_rl_['control_tt'],
                       use_contrastive_replay=args_.use_contrastive_replay_memory,
                       latest_transitions_count=args_.latest_transitions_count,
                       modal_basis_2d=env_.supervisor.modal_basis_2d)
    else:
        agent_ = None

    return env_, agent_, config_env_rl_, config_agent_, controller_type_, device_compass_, only_reset_dm_, steps_per_episode_, seed_


if __name__ == "__main__":

    args = create_args()
    env, agent, config_env_rl, config_agent, controller_type, device_compass, only_reset_dm, steps_per_episode, seed =\
        initialise(args)
    # Define the directory structure
    dir_path = os.path.join("outputs", "results", "rl", args.experiment_name)
    save_configs_to_directory(dir_path, config_env_rl, config_agent)
    # Starting values
    num_episode = 0
    total_step = 0
    pen_thres_com, command_last = 0, np.zeros(env.supervisor.rtc.get_command(0).shape, np.float32) # for writing
    r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, step_counter_for_metrics, a_pzt_total, a_tt_total = 0, 0, 0, 0, 0, 0, 0
    max_tt_value, min_tt_value, max_pzt_value, min_pzt_value, count_pzt_surpass, count_tt_surpass = -10000, 10000, -10000, 10000, 0, 0
    start_time = time.time()
    columns = ["Episode", "Total steps", "Episode steps", "Seed",
               "R total", "Rec pzt", "Rec tt", "Time", "SR LE", "SR SE",
               "Command Pzt Min", "Command Pzt Max", "Count PZT surpassing clip",
               "Command TT Min", "Command TT Max", "Count TT surpassing clip",
               "a tt", "a pzt"]

    evaluation = False
    df = pd.DataFrame(columns=columns)
    for episode in range(args.total_episodes):
        statistics_loss, statistics_keys = None, None
        list_losses = []
        s = env.reset(only_reset_dm=only_reset_dm if num_episode > 0 else False)
        if episode == 0:
            s_dict = env.get_next_state(return_dict=True)
            for key, _ in s_dict.items():
                columns.append(key + "_min")
                columns.append(key + "_max")
        step = 0
        delayed_mdp_object = DelayedMDP(config_env_rl['delayed_assignment'])
        while True:
            manage_atmospheric_conditions_3_layers(args, total_step, env, agent, controller_type)

            start_time = time.time()
            # 1. Choose action
            if args.integrator_exploration_with_only_rl_for > total_step:
                if isinstance(env, AoEnvNonLinear):
                    a = env.calculate_non_linear_residual()
                elif isinstance(env, AoEnv):
                    a = env.calculate_linear_residual()
                else:
                    raise NotImplementedError
                a /= config_env_rl['action_scale']  # divide by action scale
                noise = np.random.normal(loc=0, scale=args.noise_for_exploration, size=a.shape)
                a = a + noise
                a = np.clip(a, a_min=-1, a_max=1) # clip
                a = env.filter_actions(a, exploratory=True)  # filter
            elif controller_type == "RL":
                evaluation = choose_evaluation(args, total_step, step, episode, evaluation)
                a = agent.select_action(s, evaluation=evaluation)
                a = a.cpu().numpy()
                a = env.filter_actions(a)
            else:
                a = None
            
            # 2. Env step
            s_next, r, done, info = env.step(a, controller_type=controller_type)

            r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, max_tt_value, min_tt_value,\
                max_pzt_value, min_pzt_value, count_tt_surpass, count_pzt_surpass, a_pzt_total, a_tt_total =\
                manage_reward_and_metrics(r, r_total_train, r_pzt_total_train, r_tt_total_train,
                                          env.reconstruction_for_reward,
                                          info["sr_se"],
                                          sr_se_total,
                                          command=env.supervisor.rtc.get_command(0),
                                          clip_value=config_env_rl['command_clip_value'],
                                          max_tt_value=max_tt_value,
                                          min_tt_value=min_tt_value,
                                          max_pzt_value=max_pzt_value,
                                          min_pzt_value=min_pzt_value,
                                          count_tt_surpass=count_tt_surpass,
                                          count_pzt_surpass=count_pzt_surpass,
                                          a_pzt=env.a_pzt,
                                          a_tt=env.a_pzt_from_tt,
                                          a_pzt_total=a_pzt_total,
                                          a_tt_total=a_tt_total)

            # 3. if the delayed_mdp is ready
            # Save on replay (s, a, r, s_next) which comes from delayed_mdp and the s_next and reward this timestep
            # Depending on some configs it may vary a little
            if controller_type == "RL" and total_step < args.stop_training_after:
                if delayed_mdp_object.check_update_possibility():
                    state_assigned, action_assigned, state_next_assigned, command_assigned, command_last_assigned, mask_clipped_actuators_assigned = delayed_mdp_object.credit_assignment()
                     
                    if config_env_rl['value_commands_deltapos'] > 0:
                        delta_com_penaliser = np.zeros(env.pzt_shape, np.float32)
                        for key, item in env.supervisor.look_up_table.items():
                            # 0 is current actuator
                            # we get closest with distance_matrix_sorted with distance of 1
                            indices_with_distance_one = item[env.supervisor.distance_matrix_sorted[key] == 1]
                            neighbours_coms = env.supervisor.past_command[:-2][indices_with_distance_one]
                            current_com = env.supervisor.past_command[:-2][key]
                            argmax = np.argmax(np.square(current_com - neighbours_coms))
                            if  (np.abs(current_com) > np.abs(neighbours_coms[argmax])):
                                delta_com_penaliser[key] = np.max(np.square(current_com - neighbours_coms))
                        r = r - config_env_rl['value_commands_deltapos'] * env.supervisor.apply_projector_volts1d_to_volts2d(delta_com_penaliser)
                    
                    # c) Push to memory for different configs
                    agent.update_replay_buffer(state_assigned, action_assigned, r, state_next_assigned)

                    # d) Update
                    if step % agent.config_agent['train_every_steps'] == 0:
                        statistics_loss = agent.train()

                # 4. Save s, a, s_next, r, a_next to do the correct credit assignment in replay memory later
                # We use this object because we have delay
                delayed_mdp_object.save(s, a, s_next, env.supervisor.past_command, command_last, env.supervisor.mask_clipped_actuators)
                command_last = env.supervisor.past_command.copy()

            step += 1
            total_step += 1
            step_counter_for_metrics += 1
            # 5. s = s_next
            s = s_next.copy()
            

            if statistics_loss is not None:
                statistics_keys = ["num_updates"] + list(statistics_loss.keys())
                additional_metrics = [agent.num_updates]
                this_list_losses = additional_metrics + [value for _, value in statistics_loss.items()]
                list_losses.append(this_list_losses)

            if total_step % 1000 == 0:
                df.to_csv(dir_path + "/results.csv", index=False)
                if len(list_losses) > 0:
                    df_losses = pd.DataFrame(list_losses, columns=statistics_keys)
                    df_losses.to_csv(dir_path + "/results_losses.csv", index=False)

            if step >= steps_per_episode or total_step % env.config_env_rl['reset_strehl_every_and_print'] == 0:
                if config_env_rl['value_commands_deltapos'] > -1:
                    com_pen = - delta_com_penaliser.sum()
                else:
                    com_pen = 0
                df = add_metrics_to_dataset_and_print(df,
                                                      r_total_train,
                                                      r_pzt_total_train,
                                                      r_tt_total_train,
                                                      sr_se_total,
                                                      step_counter_for_metrics,
                                                      num_episode, 
                                                      total_step,
                                                      time.time() - start_time,
                                                      seed=env.supervisor.current_seed,
                                                      sr_le=info["sr_le"],
                                                      s_dict=env.get_next_state(return_dict=True),
                                                      max_tt_value=max_tt_value,
                                                      min_tt_value=min_tt_value,
                                                      max_pzt_value=max_pzt_value,
                                                      min_pzt_value=min_pzt_value,
                                                      count_pzt_surpass=count_pzt_surpass,
                                                      count_tt_surpass=count_tt_surpass,
                                                      a_pzt_total=a_pzt_total,
                                                      a_tt_total=a_tt_total,
                                                      number_clipped_actuators=env.supervisor.number_clipped_actuators,
                                                      com_pen=com_pen)
                r_total_train, r_pzt_total_train, r_tt_total_train, sr_se_total, step_counter_for_metrics, a_pzt_total, a_tt_total  =\
                    0, 0, 0, 0, 0, 0, 0
                max_tt_value, min_tt_value, max_pzt_value, min_pzt_value, count_pzt_surpass, count_tt_surpass =\
                    -10000, 10000, -10000, 10000, 0, 0
                start_time = time.time()

            if step >= steps_per_episode:
                break

        num_episode += 1

    if args.save_policy:
        agent.save_policy(os.path.join(dir_path, "policy_epoch_" + str(episode) + ".pth"))