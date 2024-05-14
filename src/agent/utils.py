import torch
import numpy as np
import random
from torch.optim.adam import Adam
from collections import deque
import matplotlib.pyplot as plt
import pickle
# from https://github.com/ludvb/batchrenorm

__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-3,
            momentum: float = 0.01,
            affine: bool = True,
            warmup_steps: int = 10000
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        """
        Scales standard deviation
        """
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        """
        Scales mean
        """
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        '''
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        '''
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            # x=r(x^−μ)/σ+d # changing input with running mean, std, dynamic upper limit r, dynamic shift limit d
            # μ, σ, r, d updated as:
            # -> μ = μ + momentum * (input.mean(0))
            # -> σ = σ + momentum * (input.std(0) + eps)
            # -> r = clip(input.std(0)/σ, !/rmax, rmax)
            # -> d = clip((input.mean(0) - μ)/σ, -dmax, dmax)
            # Also: optional masking
            # Also: counter "num_batches_tracked"
            # Note: The introduction of r and d mitigates some of the issues of BN, especially with small BZ or significant shifts in the input distribution.
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0)
                batch_var = z.var(0, unbiased=False)
            else:
                batch_mean = x.mean(dims)
                batch_var = x.var(dims, unbiased=False)

            # Adding warm up
            warmed_up_factor = (self.num_batches_tracked >= self.warmup_steps).float()

            running_std = torch.sqrt(self.running_var.view_as(batch_var) + self.eps)
            r = ((batch_var / running_std).clamp_(1 / self.rmax, self.rmax)).detach()
            d = (((batch_mean - self.running_mean.view_as(batch_mean)) / running_std).clamp_(-self.dmax,
                                                                                             self.dmax)).detach()
            if warmed_up_factor:
                x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            else:
                x = r * ((x - batch_mean) / torch.sqrt(batch_var + self.eps)) + d
            # Pytorch convention (1-beta)*estimated + beta*observed
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
            self.num_batches_tracked += 1
        else:  # x=r(x^−μpop​ )/σpop​ +d # running mean and std
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        if self.affine:  # Step 3 affine transform: y=γx+β
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")


def get_parameters_by_name(model: torch.nn.Module, included_names):
    """
    Extract parameters from the state dict of ``model``
    if the name contains one of the strings in ``included_names``.

    :param model: the model where the parameters come from.
    :param included_names: substrings of names to include.
    :return: List of parameters values (Pytorch tensors)
        that matches the queried names.
    """
    return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]

class DelayedMDP:
    def __init__(self, delay):
        self.delay = delay

        # for reward assignment
        self.action_list = deque(maxlen=delay+1)
        self.state_list = deque(maxlen=delay+1)
        self.next_state_list = deque(maxlen=delay+1)
        self.next_action_list = deque(maxlen=delay+1)

        # additional
        self.command_list = deque(maxlen=delay+1)
        self.command_last_list = deque(maxlen=delay+1)
        self.mask_clipped_actuators_list = deque(maxlen=delay+1)

    def check_update_possibility(self):
        """
        Checks that action list (and all the lists by the same rule)
        have enough information to take into account the delay
        """
        return len(self.action_list) >= (self.delay + 1)

    def save(self, s, a, s_next, command, command_last, mask_clipped_actuators):

        self.action_list.append(a)
        self.state_list.append(s)
        self.next_state_list.append(s_next)
        self.command_list.append(command)
        self.command_last_list.append(command_last)
        self.mask_clipped_actuators_list.append(mask_clipped_actuators)

    def credit_assignment(self):
        return self.state_list[0], self.action_list[0], self.next_state_list[-1], self.command_list[0], self.command_last_list[0], self.mask_clipped_actuators_list[0]

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.precision = torch.float32

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (torch.tensor(state,  dtype=self.precision),
                                      torch.tensor(action,  dtype=self.precision),
                                      torch.tensor(reward,  dtype=self.precision),
                                      torch.tensor(next_state, dtype=self.precision))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.position = 0

    def reset_to_10k(self):
        self.buffer = self.buffer[:10000]
        self.position = len(self.buffer)

    def save(self, folder_replay_buffer, filename):
        with open(folder_replay_buffer + filename, 'wb') as f:
            pickle.dump({
                'capacity': self.capacity,
                'buffer': self.buffer,
                'position': self.position,
                'precision': self.precision
            }, f)

    def load(self, filename, fd):
        with open(fd + filename, 'rb') as f:
            data = pickle.load(f)
            self.capacity = data['capacity']
            self.buffer = data['buffer']
            self.position = data['position']
            self.precision = data['precision']

class ContrastiveReplayMemory(ReplayMemory):
    def __init__(self, capacity, latest_transitions_count=1):
        super().__init__(capacity)
        self.latest_transitions = []  # To store the most recent N transitions
        self.latest_transitions_count = latest_transitions_count

    def push(self, state, action, reward, next_state):
        transition = (torch.tensor(state,  dtype=self.precision),
                      torch.tensor(action,  dtype=self.precision),
                      torch.tensor(reward,  dtype=self.precision),
                      torch.tensor(next_state, dtype=self.precision))

        if len(self.latest_transitions) >= self.latest_transitions_count:
            self.latest_transitions.pop(0)
        self.latest_transitions.append(transition)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random_sample_count = batch_size - len(self.latest_transitions)
        # Exclude the most recent N transitions from random sampling
        batch = random.sample(self.buffer[:-len(self.latest_transitions)], random_sample_count)
        batch.extend(self.latest_transitions)  # Append the most recent N transitions
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state, action, reward, next_state

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def initialise(device: int,
               lr_alpha: float,
               automatic_entropy_tuning: bool,
               alpha: float,
               state_channels: int,
               lr: float,
               mask_valid_actuators: np.ndarray,
               initialise_last_layer_near_zero: bool,
               initialize_last_layer_zero: bool,
               num_layers_actor: int,
               num_layers_critic: int,
               two_output_critic: bool,
               two_output_actor: bool,
               shared_layers: bool,
               agent_type: str,
               entropy_factor: float,
               crossqstyle:bool,
               beta1:float,
               beta2:float,
               beta1_alpha:float,
               beta2_alpha:float,
               hidden_dim_critic:int,
               use_batch_norm_policy:bool,
               use_batch_norm_critic:bool,
               activation_policy: str,
               activation_critic: str,
               bn_momentum: float,
               bn_mode: str,
               state_pred_layer: bool,
               include_skip_connections_critic: bool,
               include_skip_connections_actor: bool
               ):
    """
    Initialises models
    Usually we have an actor which predicts action that maximises return
    And a critic which predicts return given state and action and current policy
    :param device: device used for the models
    :param lr_alpha: learning rate alpha
    :param automatic_entropy_tuning: if True alpha is adjusted based on target entropy
    :param alpha: starting value for alpha
    :param state_channels: state channels (usually 1+pastcom+pastrec)
    :param lr: learning rate for critic and actor
    :param mask_valid_actuators: mask of valid actuators
    :param initialise_last_layer_near_zero: for the policy if we initialise close to 0
    :param initialize_last_layer_zero: for the policy if we initialise at 0
    :param num_layers_actor: number of layers for actor
    :param num_layers_critic: number of layers for critic
    :param two_output_critic: if the critic outputs two values (one for pzt and one for TT)
    :param two_output_actor: if the actor outputs two values (one for pzt and one for TT)
    :param shared_layers: if there is a shared feature extractor between actor and critic
    :param agent_type: "SAC" or "TD3"
    :param entropy_factor: if we multiply target entropy by a factor
    :param crossQstyle: if we use the upgraded version of SAC -> crossQ
    :param beta1: beta1 for ADAM (actor+critic)
    :param beta2: beta2 for ADAM (actor+critic)
    :param beta1_alpha: beta1 for ADAM (alpha)
    :param beta2_alpha: beta2 for ADAM (alpha)
    :hidden_dim_critic number of kernels for critic
    :use_batch_norm_policy: if we use batch norm for policy
    :use_batch_norm_critic: if we use batch norm for critic
    :param activation_policy: activation for actor hidden layers.
    :param activation_critic: activation for critic hidden layers.
    :param bn_momentum: momentum for batch normalization. The value is 1-value of the paper due to pytorch implmentation.
    :param bn_mode: "bn" (batch normalization) or "brn" (batch renormalization)
    :param state_pred_layer: if > -1 specify after which layer the next state is predicted
    :param include_skip_connections_critic: if True critic uses skip connections in hidden layers
    :param include_skip_connections_actor: if True actor uses skip connections in hidden layers
    """

    policy_target = None
    target_entropy, log_alpha, alpha_optim = None, None, None
    feature_extractor = None
    critic_target = None
    if agent_type == "td3":
        if shared_layers:
            raise NotImplementedError
        else:
            from src.agent.models import Td3PolicyCNNActuators, QNetworkCNNActuators
            critic = QNetworkCNNActuators(num_state_channels=state_channels,
                                          num_layers=num_layers_critic,
                                          mask_valid_actuators=mask_valid_actuators,
                                          two_output_critic=two_output_critic,
                                          two_output_actor=two_output_actor,
                                          activation=activation_critic).to(device)
            critic_target = QNetworkCNNActuators(num_state_channels=state_channels,
                                                 num_layers=num_layers_critic,
                                                 mask_valid_actuators=mask_valid_actuators,
                                                 two_output_critic=two_output_critic,
                                                 two_output_actor=two_output_actor,
                                                 activation=activation_critic).to(device)
            policy = Td3PolicyCNNActuators(num_state_channels=state_channels,
                                           mask_valid_actuators=mask_valid_actuators,
                                           num_layers=num_layers_actor,
                                           initialise_last_layer_near_zero=initialise_last_layer_near_zero,
                                           initialize_last_layer_zero=initialize_last_layer_zero,
                                           two_output_actor=two_output_actor,
                                           activation=activation_policy).to(device)
            policy_target = Td3PolicyCNNActuators(num_state_channels=state_channels,
                                                  mask_valid_actuators=mask_valid_actuators,
                                                  num_layers=num_layers_actor,
                                                  initialise_last_layer_near_zero=initialise_last_layer_near_zero,
                                                  initialize_last_layer_zero=initialize_last_layer_zero,
                                                  two_output_actor=two_output_actor,
                                                  activation=activation_policy).to(device)

            hard_update(critic, critic_target)
            hard_update(policy, policy_target)
            optim = Adam

            # actor optim
            policy_optim = optim(policy.parameters(),
                                 lr=lr,
                                 betas=(beta1, beta2))

            critic_optim = optim(critic.parameters(),
                                 lr=lr,
                                 betas=(beta1, beta2))

    elif agent_type == "sac":
        from src.agent.models import StateFeatureExtractor, GaussianPolicyCNNActuatorsSharedLayer, QNetworkCNNActuatorsSharedLayer
        if shared_layers:
            if crossQstyle:
                raise NotImplementedError
            hidden_dim_feature_extractor = 64
            # StateFeatureExtractor, GaussianPolicyCNNActuatorsSharedLayer, QNetworkCNNActuatorsSharedLayer
            feature_extractor = StateFeatureExtractor(hidden_dim_feature_extractor=hidden_dim_feature_extractor,
                                                      num_layers=1,
                                                      num_state_channels=state_channels).to(device)
            policy = GaussianPolicyCNNActuatorsSharedLayer(hidden_dim_feature_extractor=hidden_dim_feature_extractor,
                                                           mask_valid_actuators=mask_valid_actuators,
                                                           num_layers=num_layers_actor,
                                                           initialise_last_layer_near_zero=initialise_last_layer_near_zero,
                                                           initialize_last_layer_zero=initialize_last_layer_zero,
                                                           activation=activation_policy).to(device)
            critic = QNetworkCNNActuatorsSharedLayer(hidden_dim_feature_extractor=hidden_dim_feature_extractor,
                                                     num_layers=num_layers_critic,
                                                     mask_valid_actuators=mask_valid_actuators,
                                                     hidden_dim=hidden_dim_critic,
                                                     activation=activation_critic
                                                     ).to(device)
            critic_target = QNetworkCNNActuatorsSharedLayer(hidden_dim_feature_extractor=hidden_dim_feature_extractor,
                                                            num_layers=num_layers_critic,
                                                            mask_valid_actuators=mask_valid_actuators,
                                                            hidden_dim=hidden_dim_critic,
                                                            activation=activation_critic).to(device)

            hard_update(critic, critic_target)

            # actor optim
            policy_optim = Adam(list(feature_extractor.parameters()) +
                                list(policy.parameters()),
                                lr=lr,
                                betas=(beta1, beta2))

            # critic optim
            critic_optim = Adam(list(feature_extractor.parameters()) +
                                list(critic.parameters()),
                                lr=lr,
                                betas=(beta1, beta2))
        else:
            from src.agent.models import GaussianPolicyCNNActuators, QNetworkCNNActuators

            # Models
            policy = GaussianPolicyCNNActuators(num_state_channels=state_channels,
                                                mask_valid_actuators=mask_valid_actuators,
                                                num_layers=num_layers_actor,
                                                initialise_last_layer_near_zero=initialise_last_layer_near_zero,
                                                initialize_last_layer_zero=initialize_last_layer_zero,
                                                two_output_actor=two_output_actor,
                                                bn_momentum=bn_momentum,
                                                use_batch_norm=use_batch_norm_policy,
                                                bn_mode=bn_mode,
                                                include_skip_connections=include_skip_connections_actor,
                                                activation=activation_policy).to(device)
            critic = QNetworkCNNActuators(num_state_channels=state_channels,
                                          num_layers=num_layers_critic,
                                          mask_valid_actuators=mask_valid_actuators,
                                          two_output_critic=two_output_critic,
                                          two_output_actor=two_output_actor,
                                          hidden_dim=hidden_dim_critic,
                                          bn_momentum=bn_momentum,
                                          use_batch_norm=use_batch_norm_critic,
                                          bn_mode=bn_mode,
                                          state_pred_layer=state_pred_layer,
                                          include_skip_connections=include_skip_connections_critic,
                                          activation=activation_critic).to(device)
            if not crossqstyle:
                critic_target = QNetworkCNNActuators(num_state_channels=state_channels,
                                                     num_layers=num_layers_critic,
                                                     mask_valid_actuators=mask_valid_actuators,
                                                     two_output_critic=two_output_critic,
                                                     two_output_actor=two_output_actor,
                                                     hidden_dim=hidden_dim_critic,
                                                     bn_momentum=bn_momentum,
                                                     use_batch_norm=use_batch_norm_critic,
                                                     bn_mode=bn_mode,
                                                     state_pred_layer=state_pred_layer,
                                                     include_skip_connections=include_skip_connections_critic,
                                                     activation=activation_critic).to(device)
                hard_update(critic, critic_target)
            optim = Adam

            # actor optim
            policy_optim = optim(policy.parameters(),
                                 lr=lr,
                                 betas=(beta1, beta2))

            critic_optim = optim(critic.parameters(),
                                 lr=lr,
                                 betas=(beta1, beta2))

        # Alpha
        if automatic_entropy_tuning:
            target_entropy = -torch.prod(
                torch.FloatTensor(mask_valid_actuators[mask_valid_actuators == 1].shape).to(device)).item() * entropy_factor
            log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device, dtype=torch.float32)

            alpha_optim = Adam([log_alpha], lr=lr_alpha,
                                 betas=(beta1_alpha, beta2_alpha))
        else:
            target_entropy, log_alpha, alpha_optim = None, None, None
    else:
        raise NotImplementedError

    return target_entropy, log_alpha, alpha_optim,\
           policy, policy_target, critic, critic_target,\
           policy_optim, critic_optim, feature_extractor


def visualize_filters(layer, name):
    """Method to visualise filters for conv2d"""
    weight_tensor = layer.weight.data.cpu().numpy()

    num_filters = weight_tensor.shape[0]

    rows = int(num_filters ** 0.5)
    cols = num_filters // rows

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows):
        for j in range(cols):
            filter_image = weight_tensor[i * rows + j, 0, :, :]

            filter_image = (filter_image - filter_image.min()) / (filter_image.max() - filter_image.min())

            axes[i, j].imshow(filter_image, cmap="gray")
            axes[i, j].axis('off')

    plt.savefig("results/filters/" + name + ".png")
