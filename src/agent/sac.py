import torch
import torch.nn as nn
import numpy as np
import time
from src.agent.utils import soft_update, hard_update, initialise
from typing import Tuple

class Sac:
    def __init__(self,
                 config_agent,
                 state_channels,
                 action_shape,
                 device,
                 mask_valid_actuators,
                 two_output_critic,
                 two_output_actor,
                 use_contrastive_replay=False,
                 latest_transitions_count=128,
                 modal_basis_2d=None):

        self.device = "cuda:" + str(device)
        # Parameters state-action
        self.state_channels = state_channels
        self.action_shape = action_shape
        # Transpose is necessary for filtering, as batch is in first dim
        # we usually do modal_basis.dot(command)
        # here we will do command.dot(modal_basis)
        if modal_basis_2d is not None:
            self.modal_basis_2d = torch.FloatTensor(modal_basis_2d.T).to(self.device) 
        else:
            self.modal_basis_2d = None
        self.config_agent = config_agent
        self.alpha = config_agent['alpha']

        # Initializing SAC
        self.target_entropy, self.log_alpha, self.alpha_optim, \
            self.policy, _, self.critic, self.critic_target, \
            self.policy_optim, self.critic_optim, self.feature_extractor =\
            initialise(self.device,
                       config_agent['lr_alpha'],
                       config_agent['automatic_entropy_tuning'],
                       self.alpha,
                       self.state_channels,
                       config_agent['lr'],
                       mask_valid_actuators,
                       config_agent['initialise_last_layer_near_zero'],
                       config_agent['initialize_last_layer_zero'],
                       config_agent['num_layers_actor'],
                       config_agent['num_layers_critic'],
                       two_output_critic=two_output_critic,
                       two_output_actor=two_output_actor,
                       shared_layers=config_agent['shared_layers'],
                       agent_type=config_agent['agent_type'],
                       entropy_factor=config_agent['entroy_factor'],
                       crossqstyle=config_agent['crossqstyle'],
                       beta1=config_agent['beta1'],
                       beta2=config_agent['beta2'],
                       beta1_alpha=config_agent['beta1_alpha'],
                       beta2_alpha=config_agent['beta2_alpha'],
                       hidden_dim_critic=config_agent['hidden_dim_critic'],
                       use_batch_norm_critic=config_agent['use_batch_norm_critic'],
                       use_batch_norm_policy=config_agent['use_batch_norm_policy'],
                       activation_policy=config_agent['activation_policy'],
                       activation_critic=config_agent['activation_critic'],
                       bn_momentum=config_agent['bn_momentum'],
                       bn_mode=config_agent['bn_mode'],
                       state_pred_layer=config_agent['state_pred_layer'],
                       include_skip_connections_critic=config_agent['include_skip_connections_critic'],
                       include_skip_connections_actor=config_agent['include_skip_connections_actor']
                       )
        # for bn
        if not self.config_agent['crossqstyle']:
            from src.agent.utils import get_parameters_by_name
            self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
            self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        else:
            self.batch_norm_stats = None
            self.batch_norm_stats_target = None


        # Replay buffer
        if use_contrastive_replay:
            from src.agent.utils import ContrastiveReplayMemory
            self.replay_buffer = ContrastiveReplayMemory(capacity=config_agent['replay_capacity'],
                                                         latest_transitions_count=latest_transitions_count)
        else:
            from src.agent.utils import ReplayMemory
            self.replay_buffer = ReplayMemory(capacity=config_agent['replay_capacity'])
        # Counters
        self.num_updates = 0
        self.print_counter = 0
        self.print_update_every = config_agent['print_every']
        # Loss MSE
        self.mse_loss = nn.MSELoss()
        # For TT training
        self.two_output_critic = two_output_critic
        self.two_output_actor = two_output_actor

        print("----------------------------------------------")
        if self.feature_extractor is not None:
            print(self.feature_extractor)
        print(self.critic)
        print(self.policy)
        print("Policy to eval")
        self.policy.eval()
        print("Critic to eval")
        self.critic.eval()
        if self.critic_target is not None:
            print("Critic target to eval - it should always be in evaluation mode")
            self.critic_target.eval()
        print("----------------------------------------------")

    @torch.no_grad()
    def get_tensors_from_memory(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Gets tensors from replay buffer
        """
        state_batch, action_batch, reward_batch, next_state_batch = \
            self.replay_buffer.sample(batch_size=self.config_agent['batch_size'])
        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        if self.two_output_actor:
            action_batch = action_batch.to(self.device)
        else:
            action_batch = action_batch.to(self.device).unsqueeze(1)
        if self.two_output_critic:
            reward_batch = reward_batch.to(self.device)
        else:
            reward_batch = reward_batch.to(self.device).unsqueeze(1)
        return state_batch, action_batch, reward_batch, next_state_batch

    ###################################################################################################################
    @torch.no_grad()
    def get_bellman_backup(self,
                           reward_batch: torch.tensor,
                           next_state_batch: torch.tensor,
                           next_state_action: torch.tensor=None,
                           next_state_log_pi: torch.tensor=None
                           ) -> torch.tensor:
        """
        Computes bellman backup with target
        :param reward_batch: batch of rewards
        :param next_state_batch: batch of next states
        :param next_state_action: (optional) batch of next actions
        :param next_state_log_pi: (optional) batch of logprob for next actions
        """

        if next_state_action is None and next_state_log_pi is None:
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, False, False)
            if self.config_agent['filter_everywhere']:
                next_state_action = self.filter_batch(next_state_action)

        qf1_next_target, qf2_next_target, _, _ = self.critic_target(next_state_batch, next_state_action)

        if self.two_output_actor and not self.two_output_critic:
            next_state_log_pi = next_state_log_pi.mean(1, keepdim=True)
        elif self.two_output_critic and not self.two_output_actor:
            qf1_next_target = qf1_next_target.mean(1, keepdim=True)
            qf2_next_target = qf2_next_target.mean(1, keepdim=True)

        min_qf_next_target = torch.min(qf1_next_target,
                                       qf2_next_target) - self.alpha * next_state_log_pi

        next_q_value = reward_batch + self.config_agent['gamma'] * min_qf_next_target

        return next_q_value

    def calculate_q_loss(self, qf1: torch.tensor, qf2: torch.tensor, next_q_value: torch.tensor):
        """
        Calculates loss with Soft Bellman equation
        :param qf1: Q_1(s,a)
        :param qf2: Q_2(s,a)
        :param next_q_value: Bellman error
        :return: float total loss, float q1 loss, float q2 loss
        """
        qf1_loss = self.mse_loss(qf1, next_q_value)
        qf2_loss = self.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        return qf_loss, qf1_loss, qf2_loss

    def update_critic(self,
                      state_batch: torch.tensor,
                      action_batch: torch.tensor,
                      reward_batch: torch.tensor,
                      next_state_batch: torch.tensor,
                      next_state_action: torch.tensor=None,
                      next_state_log_pi: torch.tensor=None,
                      time_to_print: bool=False
                      )  -> Tuple[torch.tensor, torch.tensor, float, float, float]:
        """
        Updates critic with Soft Bellman equation
        :param state_batch: the state batch extracted from memory
        :param action_batch: the action batch extracted from memory
        :param reward_batch: the reward batch extracted from memory
        :param next_state_batch: the next state batch batch extracted from memory
        :param next_state_action: if we sample it from outside the function
        :param next_state_log_pi: if we sample it from outside the function
        :param time_to_print: bool used for metrics
        :return: float loss for Q1, float loss for Q2
        """

        if self.config_agent['crossqstyle']:
            with torch.no_grad():
                next_state_action_batch, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                # bsz x 2, nstate, 50, 50
                xstate = torch.cat([state_batch, next_state_batch], 0)
                # bsz x 2, 1, 50, 50
                xact = torch.cat([action_batch, next_state_action_batch], 0)

            # bsz x 2, 1, 50, 50
            self.critic.train()  # switch to training - to update BN statistics if any
            qfull1, qfull2, nextstatefullpred1, nextstatefullpred2 = self.critic(xstate, xact)
            self.critic.eval()  # switch to eval
            # Separating losses
            qf1, qf1next = torch.chunk(qfull1, chunks=2, dim=0)
            qf2, qf2next = torch.chunk(qfull2, chunks=2, dim=0)
            # (Optional) state prediction
            if nextstatefullpred1 is not None:
                nextstatepred1, _ = torch.chunk(nextstatefullpred1, chunks=2, dim=0)
                nextstatepred2, _ = torch.chunk(nextstatefullpred2, chunks=2, dim=0)
            else:
                nextstatepred1, nextstatepred2 = None, None
            min_qnext = torch.min(qf1next, qf2next) - self.alpha * next_state_log_pi
            next_q_value = (reward_batch + self.config_agent['gamma'] * min_qnext).detach()
        else:
            if self.config_agent['gamma'] > 0:
                next_q_value = self.get_bellman_backup(reward_batch,
                                                    next_state_batch,
                                                    next_state_action,
                                                    next_state_log_pi
                                                    )
            else:
                next_q_value = reward_batch
            self.critic.train()  # switch to training - to update BN statistics if any
            qf1, qf2, nextstatepred1, nextstatepred2 = self.critic(state_batch, action_batch)
            self.critic.eval()  # switch to eval

        # (Optional) Two output critic
        if self.two_output_critic and not self.two_output_actor:
            qf1 = qf1.mean(1, keepdim=True)
            qf2 = qf2.mean(1, keepdim=True)

        qf_loss, qf1_loss, qf2_loss = self.calculate_q_loss(qf1,
                                                            qf2,
                                                            next_q_value)
        # (Optional) state prediction
        if nextstatepred1 is not None:
            nextstate1_loss = self.config_agent['state_pred_lambda'] * 0.5 * self.mse_loss(nextstatepred1, next_state_batch)
            nextstate2_loss = self.config_agent['state_pred_lambda'] * 0.5 * self.mse_loss(nextstatepred2, next_state_batch)
            qf_loss = qf_loss + nextstate1_loss + nextstate2_loss
        else:
            nextstate1_loss, nextstate2_loss = None, None

        # Default
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if time_to_print:
            track_q1 = qf1.mean().detach().cpu().item()
            track_q2 = qf2.mean().detach().cpu().item()
            track_next_qvalue = next_q_value.mean().detach().cpu().item()
        else:
            track_q1 = None
            track_q2 = None
            track_next_qvalue = None

        return qf1_loss, qf2_loss, track_q1, track_q2, track_next_qvalue, nextstate1_loss, nextstate2_loss

    ###################################################################################################################
    def calculate_policy_loss(self, log_pi: torch.tensor, qf1_pi: torch.tensor, qf2_pi: torch.tensor) -> torch.tensor:
        """
        Calculates loss for policy
        :param log_pi: current logprob
        :param qf1_pi: prediction for q1
        :param qf2_pi: prediction for q2
        """
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = torch.mean((self.alpha * log_pi) - min_qf_pi)
        return policy_loss

    def filter_batch(self, pi: torch.tensor) -> torch.tensor:
        """
        Filters a batch of actions
        :param pi: action to filter
        :return: action filtered
        """
        pi_filtered = pi.reshape(pi.shape[0], -1).mm(self.modal_basis_2d) # matrixmatrix multiply
        pi_filtered_final = pi_filtered.reshape(pi.shape[0], 1, pi.shape[-2], pi.shape[-1])

        return pi_filtered_final


    def update_actor(self, state_batch:torch.tensor, pi:torch.tensor=None, log_pi:torch.tensor=None) -> Tuple[torch.tensor, torch.tensor]:
        """
        Updates actor
        :param state_batch: input state batch
        :param pi: (optional) input action
        :param log_pi: (optional) input lobprob from pi
        """
        # policy accepts two bools, first if we are in evaluation mode, second if we are in loop
        self.policy.train()  # switch to training - to update BN statistics if any
        if pi is None or log_pi is None:
            pi, log_pi, _ = self.policy.sample(state_batch, False, False)
            if self.config_agent['filter_everywhere']:
                pi = self.filter_batch(pi)
        self.policy.eval()

        qf1_pi, qf2_pi, _, _ = self.critic(state_batch, pi)

        if self.two_output_actor and not self.two_output_critic:
            log_pi = log_pi.mean(1, keepdim=True)
        elif self.two_output_critic and not self.two_output_actor:
            qf1_pi = qf1_pi.mean(1, keepdim=True)
            qf2_pi = qf2_pi.mean(1, keepdim=True)

        policy_loss = self.calculate_policy_loss(log_pi, qf1_pi, qf2_pi)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        af_loss = self.update_alpha(log_pi)

        return af_loss, policy_loss


    def update_alpha(self, log_pi: torch.tensor) -> torch.tensor:
        """
        Updates alpha
        :param log_pi: array, log probabilities of the current action
        + Notes:
        log_pi with CNN model -> (Bsz, 1, Nact, Nact)
        :return: float alpha loss
        """

        if self.config_agent['automatic_entropy_tuning']:
            log_pi = log_pi.reshape(self.config_agent['batch_size'], -1).sum(dim=-1, keepdim=True)
            multiplier = (log_pi + self.target_entropy).detach()
            alpha_loss = -(self.log_alpha * multiplier).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        return alpha_loss

    def update_less_operations(self) -> dict:
        """
        Agent update step with less operations
        """
        time_to_print = self.print_counter > self.print_update_every and self.num_updates % self.config_agent['update_critic_to_policy_ratio'] == 0


        state_batch, action_batch, reward_batch, next_state_batch = self.get_tensors_from_memory()

        next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, False, False)
        if self.config_agent['filter_everywhere']:
            next_state_action = self.filter_batch(next_state_action)

        # Policy
        if self.num_updates % self.config_agent['update_critic_to_policy_ratio'] == 0:
            alpha_loss, pf_loss = self.update_actor(next_state_batch, next_state_action, next_state_log_pi)
        else:
            alpha_loss, pf_loss = 0, 0

        # Critic
        qf1_loss, qf2_loss, track_q1, track_q2, track_next_qvalue, nextstate1_loss, nextstate2_loss = self.update_critic(state_batch,
                                                                                                                         action_batch,
                                                                                                                         reward_batch,
                                                                                                                         next_state_batch,
                                                                                                                         next_state_action,
                                                                                                                         next_state_log_pi,
                                                                                                                         time_to_print=time_to_print)

        # Update target
        if self.num_updates % self.config_agent['target_update_interval'] == 0 and not self.config_agent['crossqstyle']:
            soft_update(self.critic_target, self.critic, self.config_agent['tau'])
            if len(self.batch_norm_stats) > 0:
                hard_update(self.batch_norm_stats, self.batch_norm_stats_target)

        self.print_counter += 1
        if time_to_print:
            if self.config_agent['automatic_entropy_tuning']:
                alpha_log = self.alpha.clone().item()
            else:
                alpha_log = torch.tensor(self.alpha).item()

            self.print_counter = 0

            statistics = {
                "q1_loss": qf1_loss.detach().cpu().numpy(),
                "q2_loss": qf2_loss.detach().cpu().numpy(),
                "pi_loss": pf_loss.detach().cpu().numpy(),
                "alpha_loss": alpha_loss.detach().cpu().numpy(),
                "alpha_log": alpha_log,
                "track_q1": track_q1,
                "track_q2": track_q2,
                "track_next_qvalue": track_next_qvalue,
                "nextstate1_loss": nextstate1_loss.detach().cpu().numpy() if nextstate1_loss is not None else 0,
                "nextstate2_loss": nextstate2_loss.detach().cpu().numpy() if nextstate2_loss is not None else 0
            }

            return statistics
        else:
            return None

    def update(self) -> dict:
        """
        Agent update step
        """
        time_to_print = self.print_counter > self.print_update_every and self.num_updates % self.config_agent['update_critic_to_policy_ratio'] == 0

        # Extract tensors
        state_batch, action_batch, reward_batch, next_state_batch = self.get_tensors_from_memory()

        # Update critic
        qf1_loss, qf2_loss, track_q1, track_q2, track_next_qvalue, nextstate1_loss, nextstate2_loss = self.update_critic(state_batch,
                                                                                                                         action_batch,
                                                                                                                         reward_batch,
                                                                                                                         next_state_batch,
                                                                                                                         time_to_print = time_to_print)

        # Update actor
        if self.num_updates % self.config_agent['update_critic_to_policy_ratio'] == 0:
            alpha_loss, pf_loss = self.update_actor(state_batch)
        else:
            alpha_loss, pf_loss = 0, 0

        # Update target
        if self.num_updates % self.config_agent['target_update_interval'] == 0 and not self.config_agent['crossqstyle']:
            soft_update(self.critic_target, self.critic, self.config_agent['tau'])
            if len(self.batch_norm_stats) > 0:
                hard_update(self.batch_norm_stats, self.batch_norm_stats_target)

        self.print_counter += 1  # counter for metrics
        if time_to_print:
            if self.config_agent['automatic_entropy_tuning']:
                alpha_log = self.alpha.clone().item()
            else:
                alpha_log = torch.tensor(self.alpha).item()

            self.print_counter = 0

            statistics = {
                "q1_loss": qf1_loss.detach().cpu().numpy(),
                "q2_loss": qf2_loss.detach().cpu().numpy(),
                "pi_loss": pf_loss.detach().cpu().numpy(),
                "alpha_loss": alpha_loss.detach().cpu().numpy(),
                "alpha_log": alpha_log,
                "track_q1": track_q1,
                "track_q2": track_q2,
                "track_next_qvalue": track_next_qvalue,
                "nextstate1_loss": nextstate1_loss.detach().cpu().numpy() if nextstate1_loss is not None else 0,
                "nextstate2_loss": nextstate2_loss.detach().cpu().numpy() if nextstate2_loss is not None else 0
            }

            return statistics
        else:
            return None

    def select_action(self, s: np.ndarray, evaluation: bool) -> torch.tensor:
        """
        Samples action based on state
        :param s: current state
        :param evaluation: if exploitation or exploration mode
        """
        state = torch.FloatTensor(s).unsqueeze(0).to(device=self.device)
        action = self.policy.sample(state, evaluation, sample_for_loop=True)
        return action

    def train(self) -> dict:
        """
        Trains agent
        """

        statistics = None
        if len(self.replay_buffer) > self.config_agent['batch_size']:
            for i in range(self.config_agent['train_for_steps']):
                time_start_iter = time.time()

                if self.config_agent['update_simplified']:
                    statistics = self.update_less_operations()
                else:
                    statistics = self.update()

                self.num_updates += 1

            if statistics is not None:
                print(statistics)
                time_update = time.time() - time_start_iter
                statistics['time_update'] = time_update
                additional_info = f"Updater metrics: send policy weights, number of updates: {self.num_updates} - iter_time {time_update:.4f} - Len replay {len(self.replay_buffer)}"
                dynamic_info = " - ".join([f"{key}: {value:.2f}" for key, value in statistics.items()])
                print(f"{additional_info} - {dynamic_info}")

        return statistics

    def update_replay_buffer(self, s, a, r, s_next):
        self.replay_buffer.push(s, a, r, s_next)

    def save_policy(self, file_name='policy_model_weights.pth'):
        print("Saving policy to: ", file_name)
        torch.save(self.policy.state_dict(), file_name)

    def load_policy(self, file_name='policy_model_weights.pth'):
        print("Loading policy from: ", file_name)
        self.policy.load_state_dict(torch.load(file_name))

    def reset_optimizer(self):
        from torch.optim.adam import Adam
        self.policy_optim = Adam(self.policy.parameters(), lr=self.config_agent['lr'])

