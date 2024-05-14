import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

def weights_init_last_layer_policy_(mean_out,
                                    log_std_out=None,
                                    initialize_last_layer_zero=False,
                                    initialise_last_layer_near_zero=False):
    """
    Initialises weights of policy
    :param mean_out: layer of mean for SAC
    :param log_std_out: layer of std for SAC
    :param initialize_last_layer_zero: weights 0
    :param initialise_last_layer_near_zero: weights close to 0
    """

    if initialize_last_layer_zero:

        with torch.no_grad():
            mean_out.weight = torch.nn.Parameter(torch.zeros_like(mean_out.weight),
                                                      requires_grad=True)
            if log_std_out is not None:
                log_std_out.weight = torch.nn.Parameter(torch.zeros_like(log_std_out.weight),
                                                             requires_grad=True)
                torch.nn.init.constant_(log_std_out.bias, -1)
    elif initialise_last_layer_near_zero:

        with torch.no_grad():
            mean_out.weight = torch.nn.Parameter(torch.zeros_like(mean_out.weight),
                                                      requires_grad=True)
            torch.nn.init.xavier_uniform_(mean_out.weight,
                                          gain=1e-4)
            if log_std_out is not None:
                log_std_out.weight = torch.nn.Parameter(torch.zeros_like(log_std_out.weight),
                                                        requires_grad=True)

                torch.nn.init.xavier_uniform_(log_std_out.weight,
                                              gain=1e-4)
                torch.nn.init.constant_(log_std_out.bias, -1)

    return mean_out, log_std_out


def weights_init_(m, gain=1):
    if isinstance(m, StateFeatureExtractor):
        pass  # Skip the shared feature extractor
    elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class StatePredictionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StatePredictionModule, self).__init__()
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.layer(x)

class QNetworkCNNActuators(nn.Module):
    def __init__(self,
                 num_state_channels=4,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 num_layers=2,
                 kernel_size=3,
                 activation="relu",
                 two_output_critic=False,
                 two_output_actor=False,
                 bn_momentum: float = 0.9,
                 use_batch_norm: bool = False,
                 bn_mode: str = "bn",
                 state_pred_layer: int=-1,
                 include_skip_connections: bool= False
                 ):
        """
        QNetwork for SAC/CrossQ
        :param num_state_channels: number of inputs channels
        :param hidden_dim: number of kernels for cnn
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param kernel_size: kernel size, default 3x3
        :param mask_valid_actuators: mask of valid actions
        :param two_output_critic: if output in critic becomes two channels (one for PZT and another one for TT)
        :param two_output_actor: if output in actor becomes two channels (one for PZT and another one for TT)
        :param use_batch_norm: if True batch norm is used between layers
        :param bn_momentum: the momentum value for batch norm
        :param bn_mode: "bn" or "brn"
        :param state_pred_layer: specify after which layer to predict the next state
        :param include_skip_connections: if True skipconnections are included from input to each layer
        """
        super(QNetworkCNNActuators, self).__init__()

        self.include_skip_connections = include_skip_connections
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        # Activations
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU()
        elif activation == "tanh":
            activation = nn.Tanh()
        else:
            raise NotImplementedError

        if bn_mode == "bn":
            BN = nn.BatchNorm2d
        elif bn_mode == "brn":
            from src.agent.utils import BatchRenorm2d
            BN = BatchRenorm2d
        else:
            raise NotImplementedError

        output_channels = 2 if two_output_critic else 1
        actor_channels = 2 if two_output_actor else 1
        input_dim_hidden = hidden_dim
        if self.include_skip_connections:
            input_dim_hidden = hidden_dim + num_state_channels + actor_channels
        if use_batch_norm:
            self.last_layer_per_block = BN
        else:
            self.last_layer_per_block = (nn.ReLU, nn.LeakyReLU, nn.Tanh)
        # Layers
        self.q1_list = nn.ModuleList()
        self.q2_list = nn.ModuleList()
        # BN layer 0 - according to the code of crossQ
        if use_batch_norm:
            self.q1_list.append(BN(num_state_channels + actor_channels, momentum=bn_momentum))
            self.q2_list.append(BN(num_state_channels + actor_channels, momentum=bn_momentum))

        self.q1_list.append(nn.Conv2d(num_state_channels + actor_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
        self.q1_list.append(activation)
        self.q2_list.append(nn.Conv2d(num_state_channels + actor_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
        self.q2_list.append(activation)

        # BN layer 0 - according to the code of crossQ
        if use_batch_norm:
            self.q1_list.append(BN(hidden_dim, momentum=bn_momentum))
            self.q2_list.append(BN(hidden_dim, momentum=bn_momentum))

        for i in range(num_layers - 1):
            self.q1_list.append(nn.Conv2d(input_dim_hidden, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
            self.q1_list.append(activation)
            self.q2_list.append(nn.Conv2d(input_dim_hidden, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
            self.q2_list.append(activation)

            if use_batch_norm:
                self.q1_list.append(BN(hidden_dim, momentum=bn_momentum))
                self.q2_list.append(BN(hidden_dim, momentum=bn_momentum))

        self.q1_list.append(nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1))
        self.q2_list.append(nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1))

        # (Optional) State prediction
        self.state_pred_layer = state_pred_layer
        if self.state_pred_layer > -1 and self.state_pred_layer is not None:
            assert self.state_pred_layer <= num_layers
            # Prediction module
            self.state_pred_module = StatePredictionModule(input_dim=hidden_dim,
                                                           output_dim=num_state_channels)

        # Weight initialization
        self.apply(weights_init_)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(
            torch.tensor(mask_valid_actuators.reshape(-1, mask_valid_actuators.shape[0], mask_valid_actuators.shape[1]),
                         dtype=torch.float32), requires_grad=False)

    def forward(self, state, action) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Recieve as input state and action in 2D format.
        :param state: (Bsz, N, Na, Na).
        :param action: (Bsz, 1, Na, Na).
        :return: Tuple containing Q_1(s,a), Q_2(s,a), next_state_pred1, next_state_pred2.
             Q_1(s,a) and Q_2(s,a) are of shape (Bsz, 1, Na, Na).
             next_state_pred1 and next_state_pred2 are the predicted next states after the specified layer.
        """

        x = torch.cat([state, action], 1)
        x1 = x.clone()
        x2 = x.clone()
        next_state_pred1, next_state_pred2 = None, None
        idx_layer = 0
        for i, layerq1 in enumerate(self.q1_list):
            x1 = layerq1(x1)
            x2 = self.q2_list[i](x2)

            if isinstance(layerq1, self.last_layer_per_block):
                idx_layer +=1
                # (Optional) Code for state prediction
                if idx_layer == self.state_pred_layer:
                    next_state_pred1 = self.state_pred_module(x1)
                    next_state_pred2 = self.state_pred_module(x2)

                # (Optional) Code for skip connection
                if self.include_skip_connections and (idx_layer < self.num_layers):
                    x1 = torch.cat([x1, x], dim=1)
                    x2 = torch.cat([x2, x], dim=1)

        x1 = torch.mul(x1, self.mask_valid_actuators)
        x2 = torch.mul(x2, self.mask_valid_actuators)

        return x1, x2, next_state_pred1, next_state_pred2

    def to(self, device):
        return super(QNetworkCNNActuators, self).to(device)


class GaussianPolicyCNNActuators(nn.Module):
    def __init__(self,
                 num_state_channels=4,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 num_layers=2,
                 activation="relu",
                 kernel_size=3,
                 initialize_last_layer_zero=False,
                 initialise_last_layer_near_zero=False,
                 eps=1e-5,
                 log_sig_max=2,
                 log_sig_min=-20,
                 two_output_actor=False,
                 bn_momentum:float=0.9,
                 use_batch_norm:bool=False,
                 bn_mode: str="bn",
                 include_skip_connections: bool = False
                 ) -> None:
        """
        Gaussian policy actuators based on SAC/CrossQ
        https://arxiv.org/pdf/1801.01290.pdf
        :param num_state_channels: number of inputs channels
        :param hidden_dim: number of kernels for cnn
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param kernel_size: kernel size, default 3x3
        :param initialize_last_layer_zero: if last layer is initialised to 0 weights
        :param initialise_last_layer_near_zero: if last layer is initialised to almost 0 weights
        :param log_sig_min: min for normal
        :param log_sig_min: max for normal
        :param eps: small value for stability
        :param mask_valid_actuators: mask of valid actions
        :param two_output_actor: if output becomes two channels (one for PZT and another one for TT)
        :param use_batch_norm: if True batch norm is used between layers
        :param bn_momentum: the momentum value for batch norm
        :param bn_mode: "bn" or "brn"
        :param include_skip_connections: if True skipconnections are included from input to each layer
        """

        super(GaussianPolicyCNNActuators, self).__init__()

        # Activations
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU()
        elif activation == "tanh":
            activation = nn.Tanh()
        else:
            raise NotImplementedError

        if bn_mode == "bn":
            BN = nn.BatchNorm2d
        elif bn_mode == "brn":
            from src.agent.utils import BatchRenorm2d
            BN = BatchRenorm2d
        else:
            raise NotImplementedError

        self.include_skip_connections = include_skip_connections
        if use_batch_norm:
            self.last_layer_per_block = BN
        else:
            self.last_layer_per_block = (nn.ReLU, nn.LeakyReLU, nn.Tanh)
        input_dim_hidden = hidden_dim
        if self.include_skip_connections:
            input_dim_hidden = hidden_dim + num_state_channels

        # Layers
        self.pi_list = nn.ModuleList()

        if use_batch_norm:
            self.pi_list.append(BN(num_state_channels, momentum=bn_momentum))
        self.pi_list.append(torch.nn.Conv2d(num_state_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
        self.pi_list.append(activation)
        if use_batch_norm:
            self.pi_list.append(BN(hidden_dim, momentum=bn_momentum))

        for i in range(num_layers - 1):
            self.pi_list.append(nn.Conv2d(input_dim_hidden, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
            self.pi_list.append(activation)
            if use_batch_norm:
                self.pi_list.append(BN(hidden_dim, momentum=bn_momentum))

        output_channels = 2 if two_output_actor else 1

        self.mean_out = torch.nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.log_std_out = torch.nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1)

        # Weight initialization
        self.apply(weights_init_)

        self.mean_out, self.log_std_out = weights_init_last_layer_policy_(self.mean_out, self.log_std_out,
                                                                          initialize_last_layer_zero,
                                                                          initialise_last_layer_near_zero)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(
            torch.tensor(mask_valid_actuators.reshape(-1, mask_valid_actuators.shape[0], mask_valid_actuators.shape[1]),
                         dtype=torch.float32), requires_grad=False)
        self.LOG_SIG_MAX = log_sig_max
        self.LOG_SIG_MIN = log_sig_min
        self.eps = eps

    def forward(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        state_original = state.clone()
        for i in range(len(self.pi_list)):
            state = self.pi_list[i](state)
            # (Optional) Code for skip connection
            if isinstance(self.pi_list[i], self.last_layer_per_block)  and self.include_skip_connections and (i < len(self.pi_list) - 1):
                state = torch.cat([state, state_original], dim=1)

        mean, log_std = self.mean_out(state), self.log_std_out(state)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, evaluation=False, sample_for_loop=False):
        """
        Recieve as input state in 2D format.
        :param state: (Bsz, N, Na, Na).
        :param evaluation: if we are exploring or exploiting.
        :param sample_for_loop: if we are in the loop or updating.
        :return: mean (Bsz, 1, Na, Na), log_std (Bsz, 1, Na, Na).
        """

        if sample_for_loop:
            # For the loop we do less operation and we do not track the gradients.
            with torch.no_grad():
                mean, log_std = self.forward(state)
                if evaluation:
                    mean = torch.tanh(mean)
                    mean = torch.mul(mean, self.mask_valid_actuators)
                    return mean.squeeze(0)
                else:
                    std = log_std.exp()
                    normal = Normal(mean, std)
                    x_t = normal.rsample()
                    y_t = torch.tanh(x_t)
                    a = torch.mul(y_t, self.mask_valid_actuators)
                    return a.squeeze(0)
        else:
            mean, log_std = self.forward(state)
            std = log_std.exp()

            normal = Normal(mean, std)

            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            mean = torch.tanh(mean)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log((1 - y_t.pow(2).clamp(min=0, max=1)) + self.eps)
            log_prob = log_prob.sum(1, keepdim=True)

            # with prob 1 we will have 0 on the unvalid actuators
            log_prob = torch.mul(log_prob, self.mask_valid_actuators)
            action = torch.mul(y_t, self.mask_valid_actuators)

            return action, log_prob, mean

    def to(self, device):
        return super(GaussianPolicyCNNActuators, self).to(device)

### Shared layers

class StateFeatureExtractor(nn.Module):
    def __init__(self, num_state_channels=4, hidden_dim_feature_extractor=16, num_layers=1, activation="relu", kernel_size=3):
        super(StateFeatureExtractor, self).__init__()

        # Weight initialization
        self.apply(weights_init_)
        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(num_state_channels, hidden_dim_feature_extractor, kernel_size=kernel_size, stride=1, padding=1))

        for i in range(num_layers - 1):
            self.layers.append(nn.Conv2d(hidden_dim_feature_extractor, hidden_dim_feature_extractor, kernel_size=kernel_size, stride=1, padding=1))

        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # Weight initialization
        self.apply(weights_init_)

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))

        x = torch.tanh(self.layers[-1](x))

        return x


class QNetworkCNNActuatorsSharedLayer(nn.Module):
    def __init__(self,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 hidden_dim_feature_extractor=16,
                 num_layers=2,
                 kernel_size=3,
                 activation="relu"):
        super(QNetworkCNNActuatorsSharedLayer, self).__init__()

        self.q1_list = nn.ModuleList()
        self.q2_list = nn.ModuleList()
        for i in range(num_layers - 1):
            self.q1_list.append(nn.Conv2d(hidden_dim_feature_extractor + 1, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
            self.q2_list.append(nn.Conv2d(hidden_dim_feature_extractor + 1, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        if num_layers-1 == 0:
            self.q1_list.append(nn.Conv2d(hidden_dim_feature_extractor + 1, 1, kernel_size=kernel_size, stride=1, padding=1))
            self.q2_list.append(nn.Conv2d(hidden_dim_feature_extractor + 1, 1, kernel_size=kernel_size, stride=1, padding=1))
        else:
            self.q1_list.append(nn.Conv2d(hidden_dim, 1, kernel_size=kernel_size, stride=1, padding=1))
            self.q2_list.append(nn.Conv2d(hidden_dim, 1, kernel_size=kernel_size, stride=1, padding=1))
        # Activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # Weight initialization
        self.apply(weights_init_)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(torch.tensor(mask_valid_actuators, dtype=torch.float32),
                                                 requires_grad=False)

    def forward(self, state_features, action):
        x = torch.cat([state_features, action], 1)
        x1, x2 = x, x
        # Layers
        for i in range(len(self.q1_list)-1):
            x1 = self.q1_list[i](x)
            x2 = self.q2_list[i](x)

            x1 = self.activation(x1)
            x2 = self.activation(x2)

        x1 = self.q1_list[-1](x1)
        x2 = self.q2_list[-1](x2)

        x1 = torch.mul(x1, self.mask_valid_actuators)
        x2 = torch.mul(x2, self.mask_valid_actuators)

        return x1, x2

    def to(self, device):
        return super(QNetworkCNNActuatorsSharedLayer, self).to(device)


class GaussianPolicyCNNActuatorsSharedLayer(nn.Module):
    def __init__(self,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 hidden_dim_feature_extractor=16,
                 num_layers=2,
                 activation="relu",
                 kernel_size=3,
                 initialize_last_layer_zero=False,
                 initialise_last_layer_near_zero=True,
                 eps=1e-7,
                 log_sig_max=2,
                 log_sig_min=-20
                 ):
        """
        Gaussian policy actuators based on SAC
        https://arxiv.org/pdf/1801.01290.pdf
        This version shares an initial layer with critic. A feature extractor.
        :param hidden_dim: number of kernels for cnn
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param kernel_size: kernel size, default 3x3
        :param initialize_last_layer_zero: if last layer is initialised to 0 weights
        :param initialise_last_layer_near_zero: if last layer is initialised to almost 0 weights
        :param log_sig_min: min for normal
        :param log_sig_min: max for normal
        :param eps: small value for stability
        :param mask_valid_actuators: mask of valid actions
        """

        super(GaussianPolicyCNNActuatorsSharedLayer, self).__init__()
        self.pi_list = nn.ModuleList()
        self.pi_list.append(
            nn.Conv2d(hidden_dim_feature_extractor, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))
        for i in range(num_layers - 1):
            self.pi_list.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        if num_layers-1 == 0:
            self.mean_out = torch.nn.Conv2d(hidden_dim_feature_extractor, 1, kernel_size=kernel_size, stride=1, padding=1)
        else:
            self.mean_out = torch.nn.Conv2d(hidden_dim, 1, kernel_size=kernel_size, stride=1, padding=1)

        # Activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # Weight initialization
        self.apply(weights_init_)

        self.mean_out, self.log_std_out = weights_init_last_layer_policy_(self.mean_out, self.log_std_out,
                                                                          initialize_last_layer_zero,
                                                                          initialise_last_layer_near_zero)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(torch.tensor(mask_valid_actuators, dtype=torch.float32),
                                                 requires_grad=False)
        self.LOG_SIG_MAX = log_sig_max
        self.LOG_SIG_MIN = log_sig_min
        self.eps = eps

    def forward(self, state_features, evaluation=False, sample_for_loop=False):
        """
        Recieve as input state in 2D format.
        :param state: (Bsz, N, Na, Na).
        :param evaluation: if we are exploring or exploiting.
        :param sample_for_loop: if we are in the loop or updating.
        :return: mean (Bsz, 1, Na, Na), log_std (Bsz, 1, Na, Na).
        """

        if sample_for_loop:

            # For the loop we do less operation and we do not track the gradients.
            with torch.no_grad():
                # Layers
                for i in range(len(self.pi_list)):
                    # print(i, state.dtype)
                    state_features = self.activation(self.pi_list[i](state_features))

                mean, log_std = self.mean_out(state_features), self.log_std_out(state_features)
                # log_std = (self.LOG_SIG_MIN + 0.5 * (self.LOG_SIG_MAX - self.LOG_SIG_MIN) * (log_std + 1))
                log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)

                if evaluation:
                    mean = torch.tanh(mean)
                    mean = torch.mul(mean, self.mask_valid_actuators)
                    return mean.squeeze(0)
                else:
                    std = log_std.exp()
                    normal = Normal(mean, std)
                    x_t = normal.rsample()
                    y_t = torch.tanh(x_t)
                    a = torch.mul(y_t, self.mask_valid_actuators)
                    return a.squeeze(0)
        else:
            for i in range(len(self.pi_list)):
                state_features = self.activation(self.pi_list[i](state_features))

            mean, log_std = self.mean_out(state_features), self.log_std_out(state_features)
            log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
            std = log_std.exp()

            normal = Normal(mean, std)

            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            mean = torch.tanh(mean)
            log_prob = normal.log_prob(x_t)

            log_prob -= torch.log((1 - y_t.pow(2).clamp(min=0, max=1)) + self.eps)

            log_prob = torch.mul(log_prob, self.mask_valid_actuators)
            action = torch.mul(y_t, self.mask_valid_actuators)

            return action, log_prob, mean

    def to(self, device):
        return super(GaussianPolicyCNNActuatorsSharedLayer, self).to(device)


class Td3PolicyCNNActuators(nn.Module):
    def __init__(self,
                 num_state_channels=4,
                 mask_valid_actuators=None,
                 hidden_dim=64,
                 num_layers=2,
                 activation="relu",
                 kernel_size=3,
                 initialize_last_layer_zero=False,
                 initialise_last_layer_near_zero=False,
                 eps=1e-5,
                 std=0.3,
                 two_output_actor=False
                 ):

        super(Td3PolicyCNNActuators, self).__init__()

        # Layers
        self.pi_list = nn.ModuleList()
        self.pi_list.append(
            torch.nn.Conv2d(num_state_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        for i in range(num_layers - 1):
            self.pi_list.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=1))

        output_channels = 2 if two_output_actor else 1

        self.mean_out = torch.nn.Conv2d(hidden_dim, output_channels, kernel_size=kernel_size, stride=1, padding=1)

        # Activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise NotImplementedError

        # Weight initialization
        self.apply(weights_init_)

        self.mean_out, _ = weights_init_last_layer_policy_(self.mean_out,
                                                           initialize_last_layer_zero=initialize_last_layer_zero,
                                                           initialise_last_layer_near_zero=initialise_last_layer_near_zero)

        # Mask valid actuators
        self.mask_valid_actuators = nn.Parameter(
            torch.tensor(mask_valid_actuators.reshape(-1, mask_valid_actuators.shape[0], mask_valid_actuators.shape[1]),
                         dtype=torch.float32), requires_grad=False)
        self.ACTION_MIN = -1.0
        self.ACTION_MAX = 1.0
        self.STD = std
        self.eps = eps

    def forward(self, state, evaluation=False, sample_for_loop=False):
        """
        Recieve as input state in 2D format.
        :param state: (Bsz, N, Na, Na).
        :param evaluation: if we are exploring or exploiting.
        :param sample_for_loop: if we are in the loop or updating.
        :return: mean (Bsz, 1, Na, Na), log_std (Bsz, 1, Na, Na).
        """

        if sample_for_loop:

            # For the loop we do less operation and we do not track the gradients.
            with torch.no_grad():
                # Layers
                for i in range(len(self.pi_list)):
                    state = self.activation(self.pi_list[i](state))
                mean = self.mean_out(state)

                if evaluation:
                    # TODO check
                    mean = torch.clamp(mean, min=self.ACTION_MIN, max=self.ACTION_MAX)
                    mean = torch.mul(mean, self.mask_valid_actuators)
                    return mean.squeeze(0)
                else:

                    normal = Normal(mean, self.STD)

                    x_t = normal.rsample()
                    y_t = torch.clamp(x_t, min=self.ACTION_MIN, max=self.ACTION_MAX)
                    a = torch.mul(y_t, self.mask_valid_actuators)
                    return a.squeeze(0)
        else:
            for i in range(len(self.pi_list)):
                state = self.activation(self.pi_list[i](state))
            mean = self.mean_out(state)

            normal = Normal(mean, self.STD)
            x_t = normal.rsample()
            y_t = torch.clamp(x_t, min=self.ACTION_MIN, max=self.ACTION_MAX)
            action = torch.mul(y_t, self.mask_valid_actuators)

            return action, mean

    def to(self, device):
        return super(Td3PolicyCNNActuators, self).to(device)