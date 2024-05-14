from src.env_methods.env_with_phase import AoEnvWithPhase
import numpy as np
import torch
import random
from src.config import obtain_config_env_rl_default
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

seed = 1234
# sh_10x10_2m.py, sh_40x40_8m.py, "pyr_40x40_8m.py"
parameter_file = "pyr_40x40_8m_gs_0_n3.py"
device_compass = 4

# Configs
config_env_rl = obtain_config_env_rl_default(parameter_file, n_reverse_filtered_from_cmat=100)  # for environment
config_env_rl['reset_strehl_every_and_print'] = 999999999999999

# Setting seed in libraries - seed in the env will be set up when we create the object
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

config_env_rl['s_dm_residual_phase'] = True
config_env_rl['number_of_previous_s_dm_residual_phase'] = 0
config_env_rl['s_dm_residual_phase_tt'] = False
config_env_rl['number_of_previous_s_dm_residual_phase_tt'] = 0

# Environment
env = AoEnvWithPhase(gain_factor_phase=0.9,  # we set it up later
                     config_env_rl=config_env_rl,
                     parameter_file=parameter_file,
                     seed=seed,
                     device=device_compass)


def integrator_with_phase(gain_factor_phase, no_filter, len_loop):
    env.gain_factor_phase = gain_factor_phase
    env.reset_without_rl(False)
    sr_se_tot = 0
    for _ in range(len_loop):
        env.step_only_phase(no_filter=no_filter)
        sr_se_tot += env.supervisor.target.get_strehl(0)[0]
    sr_se_tot /= float(len_loop)
    sr_le = env.supervisor.target.get_strehl(0)[1]
    print("----------")
    print("gain_factor_phase, no_filter, len_loop", gain_factor_phase, no_filter, len_loop)
    print("SR SE:", sr_se_tot, "SR LE: ", sr_le)


integrator_with_phase(0.9, True, 1000)
integrator_with_phase(0.9, False, 1000)