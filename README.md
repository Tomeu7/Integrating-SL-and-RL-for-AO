# Implementation of "Integrating supervised and reinforcement learning for predictive control with an unmodulated pyramid wavefront sensor for adaptive optics"

### Visualization of the RL agent working with SR, action, reward and two channels of the state

<div align="center">
    <img src="https://gitlab.com/Tomeu/test-repository-for-paper/-/raw/main/img/visualisation.gif" width="600">
</div>

## Requirements

+ Anaconda installation with
    - Gym
    - Pytorch
    - Compass (https://github.com/ANR-COMPASS/shesha).
+ CUDA 11.6
    - We did the experiments COMPASS 5.2.1 and CUDA 11.6.

## Installation

For installation we provide two options:

### Option 1: install via docker (requirements: docker)

We provide an installation tutorial with docker. First build the docker image with:

```
sudo docker build -t integrating-sl-rl-for-ao .
```

Now run a container with:

```
sudo docker run -it --gpus all --rm -v $HOME/test-repository-for-paper:/test-repository-for-paper integrating-sl-rl-for-ao /bin/bash
```

And then inside the container activate the conda environment created to run the tests:
```
conda activate integrating-sl-rl-for-ao
```

### Option 2: install via anaconda (requirements:anaconda)

After anaconda installation you can create a new environment with python 3.8.13

```
conda env create -f environment.yml
```

Ensure CUDA version is 11.6. Otherwise, you might need to change CUDA version or COMPASS version. Moreover, you need to set up COMPASS variables as requested in their repository.


## Directory structure

```
project
│   README.md
└───src
│   └───config.py # Configuration for environment and agent
│   └───agent # RL folder
│       └───models.py # Helper functions and preprocessing
│       └───sac.py # Soft Actor Critic implementation
│       └───utils.py # Helper functions for the RL agent
│   └───unet # U-Net folder
│       └───dataset.py # dataset functionalities that will be used when training the U-Net
│       └───unet.py # U-Net code
│   └───mains # Code to run
│       └───mains_rl # RL folder that you can execute
│                  └───main.py # Basic agent with default functionalities
│                  └───helper.py # Helper functions for the RL closed loop
│       └───mains_unet # U-Net folder that you can execute
│                  └───main_closed_loop_unet.py # Closed loop with the U-Net
│                  └───main_recollect_data.py # Recollect data for the U-Net
│                  └───main_train.py # Train the U-Net
└───shesha # Shesha package to execute COMPASS with python
│       └───Supervisor # Scripts to communicate with COMPASS.
│       │       │MlSupervisor.py  Supervisor modification for a ML problem (only running UNet).
│       │       │RlSupervisor.py  Supervisor modification for a RL problem.
│       │   ...
└───data # Folder of parameter files for different simulations and U-Net models
└───global_cte # constants values for paths
```

## Usage

### 1. Training Unet

First, collect data:

```
python src/mains/mains_unet/main_recollect_data.py --parameter_file "pyr_40x40_8m_gs_9.py" --data_size 200000 --dataset_name "path_to_save_dir/"
```

Then train with the code below.

```
python src/mains/mains_unet/main_train.py --experiment_name "test" --data_name "path_to_save_dir" --use_voltage_as_phase
```

### 2. Testing Unet

Once trained, you can test the U-Net either in closed loop:

```
# Closed loop
python src/mains/mains_unet/main_closed_loop_unet.py --parameter_file "pyr_40x40_8m_gs_9.py" --device 7 --unet_dir "m9_example" --unet_name "80_net_reb_nosubtractmean_L1_relative_n3_M9_clip_0_L1_relative.pth" --normalization_noise_unet --normalization_noise_value_unet 0
```

### 3. Training RL


```
# Testing the UNet+Linear in the same setting as RL
python src/mains/mains_rl/main_with_args.py --seed 1234 --parameter_file "pyr_40x40_8m_gs_9.py" --r0 "0.12" --experiment_name "test_unet" --number_of_modes_filtered 100 --device_compass 0 --device_unet 0 --unet_name "80_net_reb_nosubtractmean_L1_relative_n3_M9_clip_0_L1_relative.pth" --unet_dir "m9_example" --s_dm_residual_non_linear --normalization_noise_unet --steps_per_episode 60000 --total_episodes 1 --normalization_noise_value_unet 0 --control_tt --controller_type "UNet+Linear"

# Training RL
python src/mains/mains_rl/main_with_args.py --seed 1234 --parameter_file "pyr_40x40_8m_gs_9.py" --r0 "0.12" --experiment_name "test_rl" --delayed_assignment 1 --gamma 0.1 --number_of_modes_filtered 100 --device_rl 2 --device_compass 2 --device_unet 2 --number_of_previous_s_dm=3  --mode "correction" --unet_name "80_net_reb_nosubtractmean_L1_relative_n3_M9_clip_0_L1_relative.pth" --unet_dir "m9_example" --s_dm_residual_non_linear --s_dm_residual_non_linear_tt --normalization_noise_unet --steps_per_episode 60000 --total_episodes 1 --normalization_noise_value_linear -1 --normalization_noise_value_unet 0 --reward_scale 10 --s_dm_tt --number_of_previous_s_dm_tt 3 --control_tt --value_commands_deltapos 0.00001 --command_clip_value 1000 --filter_commands --evaluation_after_steps 30000
```

This work has been a collaboration of Barcelona Supercomputing Center, Paris Observatory and Universitat Politècnica de Catalunya for the RisingSTARS project.

<div align="center">
  <img src="https://gitlab.com/Tomeu/test-repository-for-paper/-/raw/main/img/Image1.png" width="200" />
  <img src="https://gitlab.com/Tomeu/test-repository-for-paper/-/raw/main/img/Image2.png" width="200" />
  <img src="https://gitlab.com/Tomeu/test-repository-for-paper/-/raw/main/img/Image3.jpg" width="200" />
  <img src="https://gitlab.com/Tomeu/test-repository-for-paper/-/raw/main/img/Image4.png" width="200" />
</div>

## Acknowledgments

We would like to thank user pranz24 for providing a working version of Soft Actor Critic in Pytorch in https://github.com/pranz24/pytorch-soft-actor-critic. Also the U-Net code shares structure with https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.