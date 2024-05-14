# main_closed_loop_unet.py
from src.env_methods.env_non_linear import AoEnvNonLinear
import numpy as np
import torch
import random
from src.config import obtain_config_env_rl_default
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import json
from src.global_cte import FOLDER_RESULTS_UNET
plt.style.use("ggplot")

class closedLoopUnetTester:
    def __init__(self,
                 parameter_file: str,
                 unet_dir:str,
                 unet_name:str ="40_net_Final_dataset_g9_normalization_noise_L1_relative.pth",
                 seed:int=1234,
                 device:int=0,
                 nfilt:int=100,
                 subtract_mean_from_phase:bool=False,
                 normalization_noise_unet:bool=True,
                 normalization_noise_value_unet:float=3,
                 use_wfs_mask_unet:bool=False,
                 gains_json_path: str= "",
                 normalization_noise_value_linear: float = -1,
                 only_predict_non_linear: bool = False
                 ) -> None:
        """
        Class to test U-Net in closed-loop
        Args:
        + unet_dir: directory of UNet models
        + unet_name: name of U-Net model
        + seed: seed for the simulation
        + device: device used for the U-Net model
        + nfilt: number of modes filtered
        + subtract_mean_from_phase: if True the mean is subtracted from the phase
        + normalization_noise_unet: if we do remove expected RMS noise
        + normalization_noise_value_unet: value to subtract if we do normalization_noise_unet
        + use_wfs_mask_unet: if valid pixels for wfs is used
        + gains_json_path: path to save gains
        + normalization_noise_value_linear: if >= 0 we clipd values <0 from WFS image for linear rec.
        + only_predict_non_linear: if True, the UNet only predicts non-linear component
        """

        self.parameter_file = parameter_file
        self.unet_dir = unet_dir
        self.gains_json_path = gains_json_path
        self.gains_df_dir = "/".join(self.gains_json_path.split("/")[:2]) + "/"
        self.parameter_file_save_name = args.parameter_file.split("/")[-1][:-3]
        self.unet_save_name = unet_name[:-4]
        self.only_predict_non_linear = only_predict_non_linear
        config_env_rl = obtain_config_env_rl_default(parameter_file,
                                                     n_reverse_filtered_from_cmat=nfilt,
                                                     no_subtract_mean_from_phase=not subtract_mean_from_phase)  # for environment
        config_env_rl['reset_strehl_every_and_print'] = 999999999999999

        # Key for linear in gain will depend
        self.normalization_noise_value_linear = normalization_noise_value_linear
        if self.normalization_noise_value_linear >= 0:
            print("normalization_noise_value_linear: ", normalization_noise_value_linear, ", greater than 0, changing keys of dict for gain")
            self.linear_dict_key = "Linear_norm_" + str(self.normalization_noise_value_linear)
            self.combination_with_linear_lin_dict_key = "gain_linear_combination_" + str(self.normalization_noise_value_linear)
            self.combination_with_linear_nonlin_dict_key = "gain_non_linear_combination_" + str(self.normalization_noise_value_linear)
        else:
            self.linear_dict_key = "Linear"
            self.combination_with_linear_lin_dict_key = "gain_linear_combination"
            self.combination_with_linear_nonlin_dict_key = "gain_non_linear_combination"

        self.device = device

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        unet_type = "volts"

        # Environment
        self.env = AoEnvNonLinear(unet_dir,
                                  unet_name,
                                  unet_type,
                                  self.only_predict_non_linear,
                                  "cuda:" + str(device),
                                  gain_factor_unet=0,  # we set it up later
                                  normalize_flux=False,
                                  normalization_095_005=True,
                                  config_env_rl=config_env_rl,
                                  parameter_file=parameter_file,
                                  seed=seed,
                                  device_compass=[device],
                                  normalization_noise_unet=normalization_noise_unet,
                                  normalization_noise_value_unet=normalization_noise_value_unet,
                                  use_wfs_mask_unet=use_wfs_mask_unet,
                                  normalization_noise_value_linear=normalization_noise_value_linear)


    def update_json_with_gains(self, df: pd.DataFrame, r0: float) -> None:
        """
        Function to update the JSON file with new gain values
        :param df: dataframe of results
        :param r0: current r0 value
        """

        best_combination = df.nlargest(1, 'sr_le').iloc[0]
        best_unet = df[df['gain_factor_linear'] == 0].nlargest(1, 'sr_le').iloc[0]

        print("Updating JSON with gains")
        # Read existing JSON file
        with open(self.gains_json_path, 'r') as file:
            json_data = json.load(file)

        # If the parameter file or unet_name does not exist in the JSON, add them
        if self.parameter_file_save_name not in json_data:
            json_data[self.parameter_file_save_name] = {}
        else:
            print("Parameter file already present in json")

        if self.unet_save_name not in json_data[self.parameter_file_save_name]:
            json_data[self.parameter_file_save_name][self.unet_save_name] = {}
            json_data[self.parameter_file_save_name][self.unet_save_name][str(r0)] = {
                self.combination_with_linear_nonlin_dict_key: best_combination['gain_factor_unet'],
                self.combination_with_linear_lin_dict_key: best_combination['gain_factor_linear'],
                "gain_non_linear": best_unet['gain_factor_unet']
            }
        else:
            print("Unet name and parameter file already present in json")

        if str(r0) not in json_data[self.parameter_file_save_name][self.unet_save_name]:
            json_data[self.parameter_file_save_name][self.unet_save_name][str(r0)] = {}
        else:
            print("Unet name, parameter file and r0already present in json")

        if self.linear_dict_key not in json_data[self.parameter_file_save_name]:
            json_data[self.parameter_file_save_name][self.linear_dict_key] = {}

        json_data[self.parameter_file_save_name][self.unet_save_name][str(r0)][self.combination_with_linear_nonlin_dict_key] = best_combination['gain_factor_unet']
        json_data[self.parameter_file_save_name][self.unet_save_name][str(r0)][self.combination_with_linear_lin_dict_key] = best_combination['gain_factor_linear']
        json_data[self.parameter_file_save_name][self.unet_save_name][str(r0)]["gain_non_linear"] = best_unet['gain_factor_unet']

        # only update with normal unet, as the gains for only_predict_non_linear are less
        if not self.only_predict_non_linear:
            best_linear = df[df['gain_factor_unet'] == 0].nlargest(1, 'sr_le').iloc[0]
            json_data[self.parameter_file_save_name][self.linear_dict_key][str(r0)] = best_linear['gain_factor_linear']

        # Write updated data back to JSON
        with open(self.gains_json_path, 'w') as file:
            json.dump(json_data, file, indent=4)

    def unet_plus_linear_scan_gains_with_r0(self, len_loop:int = 100) -> pd.DataFrame:
        """
        Scanning the gains in closed-loop
        len_loop: length of the loop for scanning the gains
        """
        if self.only_predict_non_linear:
            gains_linear = [0, 1]
        else:
            gains_linear = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
        for r0 in (0.08, 0.12, 0.16):
            print("r0: ", r0)
            self.env.supervisor.atmos.set_r0(r0)
            results = []
            for gain_factor_unet in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
                for gain_factor_linear in gains_linear:
                    sr_se_tot, sr_le, _ = self.unet_plus_linear_loop(gain_factor_unet,
                    gain_factor_linear, len_loop ,verbose=False)
                    # Storing the results in the list
                    result = {
                        "gain_factor_unet": gain_factor_unet,
                        "gain_factor_linear": gain_factor_linear,
                        "sr_se_tot": sr_se_tot,
                        "sr_le": sr_le
                    }
                    results.append(result)

            df = pd.DataFrame(results)

            # Sort by sr_le in descending order and pick top 3
            top_combinations = df.nlargest(3, 'sr_le')
            print("Best 3: ---")
            print(top_combinations[['gain_factor_unet', 'gain_factor_linear', 'sr_le']])

            # Sort by sr_se_tot in descending order and pick top 3
            top_combinations = df.nlargest(3, 'sr_se_tot')
            print("Best 3: ---")
            print(top_combinations[['gain_factor_unet', 'gain_factor_linear', 'sr_se_tot']])

            # For linear
            # Sort by sr_le in descending order and pick top 3
            top_combinations = df[df['gain_factor_unet']  == 0].nlargest(3, 'sr_le')
            print("Best 3: ---")
            print(top_combinations[['gain_factor_unet', 'gain_factor_linear', 'sr_le']])

            # Sort by sr_se_tot in descending order and pick top 3
            top_combinations = df[df['gain_factor_unet']  == 0].nlargest(3, 'sr_se_tot')
            print("Best 3: ---")
            print(top_combinations[['gain_factor_unet', 'gain_factor_linear', 'sr_se_tot']])

            # For non-linear

            # Sort by sr_le in descending order and pick top 3
            top_combinations = df[df['gain_factor_linear'] == 0].nlargest(3, 'sr_le')
            print("Best 3: ---")
            print(top_combinations[['gain_factor_unet', 'gain_factor_linear', 'sr_le']])

            # Sort by sr_se_tot in descending order and pick top 3
            top_combinations = df[df['gain_factor_linear'] == 0].nlargest(3, 'sr_se_tot')
            print("Best 3: ---")
            print(top_combinations[['gain_factor_unet', 'gain_factor_linear', 'sr_se_tot']])
            df.to_csv(self.gains_df_dir + self.unet_dir + "_" + str(r0) + ".csv")

            self.update_json_with_gains(df, r0)
        return df

    def unet_plus_linear_loop(self,
                              gain_factor_unet:float,
                              gain_factor_linear:float,
                              len_loop:int=1000,
                              return_list_of_LE:bool=False,
                              seed: int=None,
                              verbose:bool=True) -> (float, float, list):
        """
        A closed-loop experiment with linear/non-linear/linear+non-linear reconstructions
        C_t = C_t-1 + g_lin c_lin + g_non_lin c_non_lin
        Args:
            + gain_factor_unet: g_non_lin
            + gain_factor_linear: g_lin
        """
        if seed is not None:
            self.env.supervisor.current_seed = seed
        self.env.gain_factor_unet = gain_factor_unet
        self.env.gain_factor_linear = gain_factor_linear
        self.env.reset_without_rl(False)
        list_of_LE = []
        sr_se_tot = 0
        for idx in range(len_loop):
            self.env.step_only_combined_with_linear()
            sr_se_tot += self.env.supervisor.target.get_strehl(0)[0]
            if return_list_of_LE:
                list_of_LE.append(self.env.supervisor.target.get_strehl(0)[1])
        sr_se_tot /= float(len_loop)
        sr_le = self.env.supervisor.target.get_strehl(0)[1]
        if verbose:
            print("SR LE: ", sr_le, "SR SE: ", sr_se_tot)
        return sr_le, sr_se_tot, list_of_LE

    def give_me_a_gain_and_will_produce_results(self,
                                                gain_linear:float,
                                                gain_non_linear_combination:float,
                                                gain_linear_combination:float,
                                                gain_non_linear:float,
                                                len_loop:int=1000) -> (float, float, float):
        """
        Closed-loop experiments over a given configuration with all three possibilities: linear/non-linear/combination reconstructions
        C_t = C_t-1 + g_lin c_lin + g_non_lin c_non_lin
        Args:
            + gain_linear: g_lin, gain for linear only using linear rec.
            + gain_non_linear_combination: g_non_lin, gain for non-linear when using it combined with the linear rec.
            + gain_linear_combination: g_lin, gain for linear rec when using it combined with the non-linear rec.
            + gain_non_linear: g_non_lin, gain for non-linear rec. only using non-linear rec.
            + len_loop: length of the closed-loop experiments
        """
        _, _, sr_le_list_linear = self.unet_plus_linear_loop(0, gain_linear, len_loop, return_list_of_LE=True)
        _, _, sr_le_list_non_linear = self.unet_plus_linear_loop(gain_non_linear, 0, len_loop, return_list_of_LE=True)
        _, _, sr_le_list_combination = self.unet_plus_linear_loop(gain_non_linear_combination, gain_linear_combination, len_loop, return_list_of_LE=True)
        
        return sr_le_list_linear, sr_le_list_non_linear, sr_le_list_combination


    def iterate_over_r0(self, dict_of_gains: dict) -> dict:
        """
        Closed-loop experiments over different r0 with different gains
        Args:
            + dict_of_gains: dictionary of gains for each r0 value
        """
        dict_of_results = {'0.08': {}, '0.12': {}, '0.16': {}}
        for r0 in [0.08, 0.12, 0.16]:
            print("r0: ", r0)
            self.env.supervisor.atmos.set_r0(r0)

            sr_le_list_linear, sr_le_list_non_linear, sr_le_list_combination =\
                self.give_me_a_gain_and_will_produce_results(dict_of_gains[self.linear_dict_key][str(r0)],
                                                             dict_of_gains[self.unet_save_name][str(r0)][self.combination_with_linear_nonlin_dict_key],
                                                             dict_of_gains[self.unet_save_name][str(r0)][self.combination_with_linear_lin_dict_key],
                                                             dict_of_gains[self.unet_save_name][str(r0)]['gain_non_linear'])
            dict_of_results[str(r0)]['Lin+U-Net'] = sr_le_list_combination
            dict_of_results[str(r0)]['Lin'] = sr_le_list_linear
            dict_of_results[str(r0)]['U-Net'] = sr_le_list_non_linear

        return dict_of_results

    def iterate_over_r0_different_seeds(self, dict_of_gains: dict) -> list:
        """
        Experiments over different seeds
        Args:
            + dict_of_gains: dictionary of gains for each r0 value
        """
        list_of_dict_of_results = []
        for seed in [1500, 1501, 1502, 1503, 1504]:
            self.env.supervisor.current_seed = seed
            dict_of_results = self.iterate_over_r0(dict_of_gains)
            list_of_dict_of_results.append(dict_of_results)
        
        return list_of_dict_of_results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--parameter_file', default="pyr_40x40_8m_gs_0_n3.py")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--nfilt", default=100, type=int) # ops filtering 50 modes, its ok
    parser.add_argument("--unet_dir", default="nosubtractmean_Final_L1_relative_n3_M0L1_relative",
                        help="nosubtractmean_Final_L1_relative_n3_M0L1_relative, "
                             " Final_dataset_g9_normalization_noise_L1_relative")
    parser.add_argument("--unet_name", default="40_net_Final_dataset_g9_normalization_noise_L1_relative.pth",
                        help="105_net_nosubtractmean_Final_L1_relative_n3_M0L1_relative,"
                             "40_net_Final_dataset_g9_normalization_noise_L1_relative.pth")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--subtract_mean_from_phase", action="store_true")
    parser.add_argument("--test_gains", action="store_true")
    parser.add_argument("--normalization_noise_unet", action="store_true")
    parser.add_argument("--normalization_noise_value_unet", default=3, type=float)
    parser.add_argument("--use_wfs_mask_unet", action="store_true")
    parser.add_argument("--gains_json_path", default="data/gains_results/gains_per_parameter_files_and_unet.json", type=str)
    parser.add_argument("--len_loop_test_gains", default=100, type=int)
    parser.add_argument("--normalization_noise_value_linear", default=-1, type=float)
    parser.add_argument("--only_predict_non_linear", action="store_true")
    parser.add_argument("--do_different_seeds_for_test", action="store_true")
    args = parser.parse_args()

    # Check seed values
    if args.test_gains:
        seed = 200
    else:
        seed = args.seed
        assert seed != 200

    closed_loop_unet_tester = closedLoopUnetTester(args.parameter_file,
                                                   args.unet_dir,
                                                   args.unet_name,
                                                   args.seed,
                                                   args.device,
                                                   args.nfilt,
                                                   args.subtract_mean_from_phase,
                                                   normalization_noise_unet=args.normalization_noise_unet,
                                                   normalization_noise_value_unet=args.normalization_noise_value_unet,
                                                   use_wfs_mask_unet=args.use_wfs_mask_unet,
                                                   gains_json_path=args.gains_json_path,
                                                   normalization_noise_value_linear=args.normalization_noise_value_linear,
                                                   only_predict_non_linear=args.only_predict_non_linear
                                                   )

    if args.test_gains:
        print("Testing gains")
        df = closed_loop_unet_tester.unet_plus_linear_scan_gains_with_r0(args.len_loop_test_gains)
    else:
        print("Testing loop")
        """
        This list of gains is based on a precomputed optimisation
        """
        with open(args.gains_json_path, 'r') as file:
            dict_of_gains_ = json.load(file)[args.parameter_file.split("/")[-1][:-3]]
        if args.do_different_seeds_for_test:
            list_of_dict_of_results = closed_loop_unet_tester.iterate_over_r0_different_seeds(dict_of_gains_)
            with open(FOLDER_RESULTS_UNET + args.unet_name[:-4] + "_closed_loop_test_seeds.pickle", 'wb') as handle:
                pickle.dump(list_of_dict_of_results, handle)
        else:
            df = closed_loop_unet_tester.iterate_over_r0(dict_of_gains_)

            with open(FOLDER_RESULTS_UNET + args.unet_name[:-4] + "_closed_loop_test.pickle", 'wb') as handle:
                pickle.dump(df, handle)
