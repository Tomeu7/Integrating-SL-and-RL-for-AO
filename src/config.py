

def obtain_config_env_rl_default(parameter_file: str,
                                 n_reverse_filtered_from_cmat: int,
                                 no_subtract_mean_from_phase: bool = False,
                                 use_second_version_of_modal_basis: bool=False) -> dict:
    """
    Default value for env_rl
    :param parameter_file: current parameter file
    :param n_reverse_filtered_from_cmat: number of modes filtered
    :param no_subtract_mean_from_phase: if for UNet mean is subtracted from phase
    :param use_second_version_of_modal_basis: if we use second version of modal basis
    """
    config_env_rl = {}
    # Parameter file
    config_env_rl['parameter_file'] = parameter_file
    # State
    config_env_rl['normalization_bool'] = True
    config_env_rl['s_dm_residual_rl'] = False
    config_env_rl['s_dm'] = True
    config_env_rl['s_dm_residual'] = False
    config_env_rl['number_of_previous_s_dm'] = 0
    config_env_rl['number_of_previous_s_dm_residual'] = 0
    config_env_rl['number_of_previous_s_dm_residual_rl'] = 0
    config_env_rl['normalization_bool'] = True
    config_env_rl['dm_std'] = 1.0
    config_env_rl['dm_residual_std'] = 1.0
    # Reward
    config_env_rl['reward_type'] = "2d_actuators"
    config_env_rl['number_of_previous_a_for_reward'] = 0
    config_env_rl['reward_scale'] = 1.0
    config_env_rl['normalise_for_reward'] = False
    # Action
    config_env_rl['action_scale'] = 1.0
    # Other
    config_env_rl['use_second_version_of_modal_basis'] = use_second_version_of_modal_basis
    config_env_rl['filter_state_with_btt'] = True
    config_env_rl['n_reverse_filtered_from_cmat'] = n_reverse_filtered_from_cmat
    config_env_rl['filter_state_actuator_space_with_btt'] = True
    config_env_rl['use_mask_influence_function'] = False
    config_env_rl['filter_commands'] = False
    config_env_rl['command_clip_value'] = 1000
    # Mode: correction or only_rl
    config_env_rl['mode'] = "only_rl"
    # Delayed assignment
    config_env_rl['delayed_assignment'] = 0
    # Reset strehl LE every
    config_env_rl['reset_strehl_every_and_print'] = 999999999999999
    # Control tip tilt?
    config_env_rl['control_tt'] = False
    config_env_rl['number_of_previous_s_dm_residual_tt'] = 0
    config_env_rl['s_dm_residual_tt'] = False
    config_env_rl['number_of_previous_s_dm_tt'] = 0
    config_env_rl['s_dm_tt'] = False
    config_env_rl['joint_tt_into_s_dm'] = False
    config_env_rl['joint_tt_into_s_dm_residual'] = False
    config_env_rl['joint_tt_into_reward'] = False
    config_env_rl['separate_tt_into_two_actions'] = False
    config_env_rl['value_commands_deltapos'] = -1
    # Unet extra
    config_env_rl['s_dm_residual_non_linear'] = False
    config_env_rl['s_dm_residual_non_linear_tt'] = False
    config_env_rl['number_of_previous_s_dm_residual_non_linear'] = 0
    config_env_rl['number_of_previous_s_dm_residual_non_linear_tt'] = 0
    config_env_rl['joint_tt_into_s_dm_residual_non_linear'] = False
    config_env_rl['no_subtract_mean_from_phase'] = no_subtract_mean_from_phase
    # Integrator
    config_env_rl['leaky_integrator_for_rl'] = False
    config_env_rl['leak'] = -1
    # Reset
    config_env_rl['reset_when_clip'] = False
    config_env_rl['reduce_gain_tt_to'] = -1
    config_env_rl['basis_projectors'] = "actuators"

    # Norm
    config_env_rl['running_norm'] = False
    config_env_rl['running_norm_mode'] = "scalar"
    return config_env_rl

def obtain_config_env_rl(parameter_file: str,
                         number_of_modes_filtered: int=100,
                         use_mask_influence_function: bool=False,
                         mode: str="only_rl",
                         s_dm_residual: bool=True,
                         s_dm_residual_rl: bool=False,
                         number_of_previous_s_dm_residual_rl: int=0,
                         number_of_previous_s_dm: int=3,
                         s_dm: bool=True,
                         control_tt: bool=False,
                         s_dm_tt: bool=False,
                         s_dm_residual_tt: bool=False,
                         number_of_previous_s_dm_residual_tt: int=0,
                         number_of_previous_s_dm_tt: int=0,
                         s_dm_residual_non_linear: bool=False,
                         number_of_previous_s_dm_residual_non_linear: int=0,
                         s_dm_residual_non_linear_tt: bool=False,
                         number_of_previous_s_dm_residual_non_linear_tt: int=0,
                         filter_commands: bool=False,
                         command_clip_value: int=1000,
                         joint_tt_into_s_dm: bool=False,
                         joint_tt_into_s_dm_residual: bool=False,
                         scaling_for_residual_tt: float=1.0,
                         action_scale: float=10.0,
                         number_of_previous_a_for_reward: int=0,
                         correction_pzt_only_rl_tt: bool=False,
                         delayed_assignment: int=1,
                         value_commands_deltapos:float=-1,
                         use_second_version_of_modal_basis: bool=False,
                         leaky_integrator_for_rl: bool=False,
                         leak: float=-1,
                         reset_when_clip: bool=False,
                         reduce_gain_tt_to: float=-1,
                         no_subtract_mean_from_phase: bool=False,
                         reward_scale: float=1.0,
                         running_norm: bool = False,
                         running_norm_mode: str ="array",
                         normalise_for_reward: bool = False) -> dict:
    """
    Config for environment for RL experiments
    :param parameter_file: current parameter file
    :param number_of_modes_filtered: number of modes filtered
    :param use_mask_influence_function: if mask is used for influence functions
    :param mode: "correction" or "only_rl" - how to use RL
    :param s_dm_residual: state channel residual
    :param s_dm_residual_rl: state channel residual action
    :param number_of_previous_s_dm_residual_rl: number of history of actions in state
    :param number_of_previous_s_dm: number of history of residuals in state
    :param s_dm: state command
    :param control_tt: if TT mirror is controlled
    :param s_dm_tt: if state command TT into PZT projection is added to state
    :param s_dm_residual_tt: if state residual TT into PZT projection is added to state
    :param number_of_previous_s_dm_residual_tt:
    :param number_of_previous_s_dm_tt:
    :param s_dm_residual_non_linear:
    :param number_of_previous_s_dm_residual_non_linear:
    :param s_dm_residual_non_linear_tt:
    :param number_of_previous_s_dm_residual_non_linear_tt:
    :param filter_commands:
    :param command_clip_value:
    :param joint_tt_into_s_dm:
    :param joint_tt_into_s_dm_residual:
    :param scaling_for_residual_tt:
    :param action_scale:
    :param number_of_previous_a_for_reward:
    :param correction_pzt_only_rl_tt:
    :param delayed_assignment:
    :param use_second_version_of_modal_basis:
    :param leaky_integrator_for_rl:
    :param leak:
    :param reset_when_clip:
    :param reduce_gain_tt_to:
    :param no_subtract_mean_from_phase:
    :param reward_scale:
    :param running_norm:
    :param running_norm_mode:
    :param normalise_for_reward
    :return:
    """
    config_env_rl = {}
    # Parameter file
    config_env_rl['parameter_file'] = parameter_file
    # State
    config_env_rl['normalization_bool'] = True
    config_env_rl['s_dm_residual_rl'] = s_dm_residual_rl
    config_env_rl['s_dm'] = s_dm
    config_env_rl['s_dm_residual'] = s_dm_residual
    config_env_rl['number_of_previous_s_dm'] = number_of_previous_s_dm
    config_env_rl['number_of_previous_s_dm_residual'] = 0
    config_env_rl['number_of_previous_s_dm_residual_rl'] = number_of_previous_s_dm_residual_rl
    config_env_rl['normalization_bool'] = True
    config_env_rl['dm_std'] = 10.0
    config_env_rl['dm_residual_std'] = 10.0
    config_env_rl['correction_pzt_only_rl_tt'] = correction_pzt_only_rl_tt
    # Reward
    config_env_rl['reward_type'] = "2d_actuators"  # 2d_actuators or scalar_actuators
    config_env_rl['number_of_previous_a_for_reward'] = number_of_previous_a_for_reward
    config_env_rl['reward_scale'] = reward_scale
    config_env_rl['normalise_for_reward'] = normalise_for_reward
    # Action
    config_env_rl['action_scale'] = action_scale
    # Other
    config_env_rl['use_second_version_of_modal_basis'] = use_second_version_of_modal_basis
    config_env_rl['filter_state_with_btt'] = True
    config_env_rl['n_reverse_filtered_from_cmat'] = number_of_modes_filtered
    config_env_rl['use_mask_influence_function'] = use_mask_influence_function
    config_env_rl['filter_state_actuator_space_with_btt'] = True
    config_env_rl['filter_commands'] = filter_commands
    config_env_rl['command_clip_value'] = command_clip_value
    # Mode: correction or only_rl
    config_env_rl['mode'] = mode
    # Delayed assignment
    config_env_rl['delayed_assignment'] = delayed_assignment
    # Reset strehl LE every
    config_env_rl['reset_strehl_every_and_print'] = 1000
    # Control tip tilt?
    config_env_rl['control_tt'] = control_tt
    config_env_rl['number_of_previous_s_dm_residual_tt'] = number_of_previous_s_dm_residual_tt
    config_env_rl['s_dm_residual_tt'] = s_dm_residual_tt
    config_env_rl['number_of_previous_s_dm_tt'] = number_of_previous_s_dm_tt
    config_env_rl['s_dm_tt'] = s_dm_tt
    config_env_rl['joint_tt_into_s_dm'] = joint_tt_into_s_dm
    config_env_rl['joint_tt_into_s_dm_residual'] = joint_tt_into_s_dm_residual
    config_env_rl['joint_tt_into_reward'] = True
    config_env_rl['separate_tt_into_two_actions'] = False
    config_env_rl['scaling_for_residual_tt'] = scaling_for_residual_tt
    config_env_rl['basis_projectors'] = "actuators"
    config_env_rl['value_commands_deltapos'] = value_commands_deltapos
    # Unet extra
    config_env_rl['s_dm_residual_non_linear'] = s_dm_residual_non_linear
    config_env_rl['s_dm_residual_non_linear_tt'] = s_dm_residual_non_linear_tt
    config_env_rl['number_of_previous_s_dm_residual_non_linear'] = number_of_previous_s_dm_residual_non_linear
    config_env_rl['number_of_previous_s_dm_residual_non_linear_tt'] = number_of_previous_s_dm_residual_non_linear_tt
    config_env_rl['joint_tt_into_s_dm_residual_non_linear'] = joint_tt_into_s_dm_residual  # same
    config_env_rl['no_subtract_mean_from_phase'] = no_subtract_mean_from_phase
    # Integrator
    config_env_rl['leaky_integrator_for_rl'] = leaky_integrator_for_rl
    config_env_rl['leak'] = leak
    # Reset
    config_env_rl['reset_when_clip'] = reset_when_clip
    config_env_rl['reduce_gain_tt_to'] = reduce_gain_tt_to
    # Norm
    config_env_rl['running_norm'] = running_norm
    config_env_rl['running_norm_mode'] = running_norm_mode
    return config_env_rl


def obtain_config_agent(agent_type:str,
                        replay_buffer_size:int=50000,
                        entropy_factor:float=1.0,
                        automatic_entropy_tuning:bool=False,
                        alpha:float=0.2,
                        gamma:float=0.1,
                        filter_everywhere:bool=False,
                        crossqstyle:bool=False,
                        lr:float = 0.0003,
                        lr_alpha:float = 0.0003,
                        beta1:float=0.9,
                        beta2:float=0.999,
                        beta1_alpha:float=0.9,
                        beta2_alpha:float=0.999,
                        update_critic_to_policy_ratio:int=1,
                        hidden_dim_critic:int=64,
                        hidden_dim_actor:int=64,
                        use_batch_norm_critic:bool=False,
                        use_batch_norm_policy:bool=False,
                        activation_critic:str="relu",
                        activation_policy:str="relu",
                        bn_momentum:float=0.01,
                        bn_mode:str="bn",
                        state_pred_layer:int=-1,
                        state_pred_lambda:int=0.1,
                        include_skip_connections_critic:bool=False,
                        include_skip_connections_actor:bool=False,
                        num_layers_actor: int=2,
                        num_layers_critic: int=3,
                        tau: int = 0.005
                        ) -> dict:
    """
    Configuration for agent
    :param agent_type: "sac" or "td3"
    :param replay_buffer_size: length of replay buffer
    :param entropy_factor: factor for entropy in case we want to reduce exploration (only SAC)
    :param automatic_entropy_tuning: if alpha is automatically adapt (only SAC)
    :param alpha: default value of alpha (only SAC)
    :param gamma: gamma for bellman equation
    :param filter_everywhere: if filter is introduced also in training
    :param crossqstyle: if the agent is upgraded with CrossQ https://openreview.net/pdf?id=PczQtTsTIX
    :param lr: value of learning rate for actor and critic
    :param lr_alpha: value of learnign rate for alpha (only SAC)
    :param beta1: Value of beta1 for ADAM for actor and critic
    :param beta2: Value of beta2 for ADAM for actor and critic
    :param beta1_alpha: Value of beta1 for ADAM for alpha (only SAC)
    :param beta2_alpha: Value of beta2for ADAM for alpha (only SAC)
    :param update_critic_to_policy_ratio: how many times to update the critic more than the policy
    :param hidden_dim_critic: number of filters per hidden layer in critic
    :param hidden_dim_actor: number of filters per hidden layer in actor
    :param use_batch_norm_critic: if critic uses batch norm layers
    :param use_batch_norm_policy: if actor uses batch norm layers
    :param activation_critic: activation for hidden layers in critic
    :param activation_policy: activation for hidden layers in policy
    :param bn_momentum: momentum for batch normalisation
    :param bn_mode: "bn" (batch normalisation) or "brn" (batch renormalisation)
    :param state_pred_layer: if > -1 the critic is augmented with state prediction on layer {value number}
    :param state_pred_lambda: value of importantece of state prediction for critic:
    :param include_skip_connections_critic: if True critic uses skip connections in hidden layers
    :param include_skip_connections_actor: if True actor uses skip connections in hidden layers
    :param num_layers_critic: number of layers critic
    :param num_layers_actor: number of layers actor
    :param tau
    """
    config_agent = {}
    config_agent['use_batch_norm_policy'] = use_batch_norm_policy
    config_agent['use_batch_norm_critic'] = use_batch_norm_critic
    config_agent['hidden_dim_critic'] = hidden_dim_critic
    config_agent['hidden_dim_actor'] = hidden_dim_actor
    config_agent['replay_capacity'] = replay_buffer_size
    config_agent['batch_size'] = 256
    config_agent['lr'] = lr
    config_agent['target_update_interval'] = 1
    config_agent['tau'] = tau
    config_agent['lr_alpha'] = lr_alpha
    config_agent['gamma'] = gamma
    config_agent['update_simplified'] = False
    config_agent['train_for_steps'] = 1
    config_agent['train_every_steps'] = 1
    config_agent['print_every'] = 1001
    config_agent['num_layers_critic'] = num_layers_critic
    config_agent['num_layers_actor'] = num_layers_actor
    # Weight initialization policy
    config_agent['initialize_last_layer_zero'] = True
    config_agent['initialise_last_layer_near_zero'] = False
    config_agent['shared_layers'] = False
    config_agent['agent_type'] = agent_type
    config_agent['filter_everywhere'] = filter_everywhere
    config_agent['crossqstyle'] = crossqstyle
    config_agent['beta1'] = beta1
    config_agent['beta2'] = beta2
    config_agent['beta1_alpha'] = beta1_alpha
    config_agent['beta2_alpha'] = beta2_alpha
    config_agent['update_critic_to_policy_ratio'] = update_critic_to_policy_ratio
    config_agent['activation_critic'] = activation_critic
    config_agent['activation_policy'] = activation_policy
    config_agent['bn_momentum'] = bn_momentum
    config_agent['bn_mode'] = bn_mode
    config_agent['state_pred_layer'] = state_pred_layer
    config_agent['state_pred_lambda'] = state_pred_lambda
    config_agent['include_skip_connections_critic'] = include_skip_connections_critic
    config_agent['include_skip_connections_actor'] = include_skip_connections_actor
    if agent_type == "sac":
        config_agent['alpha'] = alpha
        config_agent['entroy_factor'] = entropy_factor
        config_agent['automatic_entropy_tuning'] = automatic_entropy_tuning
    elif agent_type == "td3":
        config_agent['alpha'] = None
        config_agent['entroy_factor'] = None
        config_agent['automatic_entropy_tuning'] = None
    else:
        raise NotImplementedError

    # TD3 only
    config_agent['update_policy_td3_every'] = 2

    return config_agent


