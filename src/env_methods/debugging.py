import numpy as np
import matplotlib.pyplot as plt

def save_prediction(real_2d_,
                    linear_reconstruction_2d_,
                    non_linear_reconstruction_2d_,
                    mask_valid_actuators,
                    real_phase_on_pupil,
                    non_linear_reconstruction_phase_on_pupil,
                    linear_reconstruction_phase_on_pupil,
                    real_phase_tt_on_pupil,
                    non_linear_reconstruction_phase_tt_on_pupil,
                    linear_reconstruction_phase_tt_on_pupil,
                    pupil,
                    target_shape,
                    i_save_unet):
    real_2d_[mask_valid_actuators == 0] = np.nan
    linear_reconstruction_2d_[mask_valid_actuators == 0] = np.nan
    non_linear_reconstruction_2d_[mask_valid_actuators == 0] = np.nan
    difference_volts_linear = (real_2d_ + linear_reconstruction_2d_)
    difference_volts_non_linear = (real_2d_ + non_linear_reconstruction_2d_)
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"Real: {real_2d_[mask_valid_actuators == 1].std():.3f}", fontsize=14)
    plt.imshow(real_2d_)
    plt.subplot(2, 3, 2)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"L-Rec.: {linear_reconstruction_2d_[mask_valid_actuators == 1].std():.3f}", fontsize=14)
    plt.imshow(linear_reconstruction_2d_)
    plt.subplot(2, 3, 3)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"L-Res.: {difference_volts_linear[mask_valid_actuators == 1].std():.3f}", fontsize=14)
    plt.imshow(difference_volts_linear)
    plt.subplot(2, 3, 4)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"Real: {real_2d_[mask_valid_actuators == 1].std():.3f}", fontsize=14)
    plt.imshow(real_2d_)
    plt.subplot(2, 3, 5)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"NL-Rec.: {non_linear_reconstruction_2d_[mask_valid_actuators == 1].std():.3f}", fontsize=14)
    plt.imshow(non_linear_reconstruction_2d_)
    plt.subplot(2, 3, 6)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"NL-Res.: {difference_volts_non_linear[mask_valid_actuators == 1].std():.3f}", fontsize=14)
    plt.imshow(difference_volts_non_linear)
    path = f"results/unet_predictions/save_unet_{i_save_unet}.png"
    plt.savefig(path)
    plt.close()

    # 2) phase
    real_phase = np.zeros(target_shape)
    non_linear_reconstruction_phase = np.zeros(target_shape)
    linear_reconstruction_phase = np.zeros(target_shape)
    real_phase[pupil == 1] = real_phase_on_pupil
    non_linear_reconstruction_phase[pupil == 1] = non_linear_reconstruction_phase_on_pupil
    linear_reconstruction_phase[pupil == 1] = linear_reconstruction_phase_on_pupil

    min_val_1_3 = min(real_phase[pupil == 1].min(), linear_reconstruction_phase[pupil == 1].min())
    max_val_1_3 = max(real_phase[pupil == 1].max(), linear_reconstruction_phase[pupil == 1].max())
    min_val_4_6 = min(real_phase[pupil == 1].min(), non_linear_reconstruction_phase[pupil == 1].min())
    max_val_4_6 = max(real_phase[pupil == 1].max(), non_linear_reconstruction_phase[pupil == 1].max())

    real_phase[pupil == 0] = np.nan
    linear_reconstruction_phase[pupil == 0] = np.nan
    non_linear_reconstruction_phase[pupil == 0] = np.nan

    fig = plt.figure()
    plt.subplot(2, 3, 1)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"Real: {real_phase[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(real_phase)
    plt.subplot(2, 3, 2)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"L-Rec: {linear_reconstruction_phase[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(linear_reconstruction_phase)
    plt.subplot(2, 3, 3)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    difference_linear = real_phase + linear_reconstruction_phase
    plt.title(f"L-Res: {difference_linear[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(difference_linear)
    plt.subplot(2, 3, 4)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"Real: {real_phase[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(real_phase)
    plt.subplot(2, 3, 5)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"NL-Rec: {non_linear_reconstruction_phase[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(non_linear_reconstruction_phase)
    plt.subplot(2, 3, 6)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    difference_non_linear = real_phase + non_linear_reconstruction_phase
    plt.title(f"NL-Res: {difference_non_linear[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(difference_non_linear)
    # Adjust the layout to make space for the colorbar

    # Save and close
    path = f"results/unet_predictions/save_unet_phase_{i_save_unet}.png"
    plt.savefig(path)
    plt.close()

    # Only TT

    real_phase_tt = np.zeros(target_shape)
    non_linear_reconstruction_phase_tt = np.zeros(target_shape)
    linear_reconstruction_phase_tt = np.zeros(target_shape)
    real_phase_tt[pupil == 1] = real_phase_tt_on_pupil
    non_linear_reconstruction_phase_tt[pupil == 1] = non_linear_reconstruction_phase_tt_on_pupil
    linear_reconstruction_phase_tt[pupil == 1] = linear_reconstruction_phase_tt_on_pupil

    real_phase_tt[pupil == 0] = np.nan
    linear_reconstruction_phase_tt[pupil == 0] = np.nan
    non_linear_reconstruction_phase_tt[pupil == 0] = np.nan

    fig = plt.figure()
    plt.subplot(2, 3, 1)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"Real: {real_phase_tt[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(real_phase_tt)
    plt.subplot(2, 3, 2)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"L-Rec: {linear_reconstruction_phase_tt[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(linear_reconstruction_phase_tt)
    plt.subplot(2, 3, 3)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    difference_linear_tt = real_phase_tt + linear_reconstruction_phase_tt
    plt.title(f"L-Res: {difference_linear_tt[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(difference_linear_tt)
    plt.subplot(2, 3, 4)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"Real: {real_phase_tt[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(real_phase_tt)
    plt.subplot(2, 3, 5)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.title(f"NL-Rec: {non_linear_reconstruction_phase_tt[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(non_linear_reconstruction_phase_tt)
    plt.subplot(2, 3, 6)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    difference_non_linear_tt = real_phase_tt + non_linear_reconstruction_phase_tt
    plt.title(f"NL-Res: {difference_non_linear_tt[pupil == 1].std():.3f} um", fontsize=13)
    plt.imshow(difference_non_linear_tt)
    # Adjust the layout to make space for the colorbar

    # Save and close
    path = f"results/unet_predictions/save_unet_phase_tt_{i_save_unet}.png"
    plt.savefig(path)
    plt.close()