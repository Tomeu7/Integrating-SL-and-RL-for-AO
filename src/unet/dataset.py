# dataset.py
import os
import numpy as np
import copy
import pandas as pd
import h5py

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.npz', '.hdf5'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class Dataset:
    def __init__(self,
                 dataroot,
                 save_dir,
                 mode,
                 use_voltage_as_phase=True,
                 max_dataset_size=float("inf"),
                 input_nc=4,
                 output_nc=1,
                 normalization_noise=False,
                 no_subtract_mean_from_phase=False):

        self.dataroot = dataroot  # get the image directory
        self.dataset_paths = sorted(make_dataset(os.path.join(dataroot,  mode) + "/", max_dataset_size))  # get image paths
        self.use_voltage_as_phase = use_voltage_as_phase
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.normalization_noise = normalization_noise
        self.no_subtract_mean_from_phase = no_subtract_mean_from_phase
        if normalization_noise:
            self.readout_noise = 3
        else:
            self.readout_noise = 0
        self.normalization_095_005 = True

        self.max_wfs_image, self.min_wfs_image, self.max_phase, self.min_phase, \
            self.scale_wfs, self.scale_phase, self.scale_phase_modified, self.min_phase_modified = \
            self.obtain_norm_values(dataroot, use_voltage_as_phase, save_dir)

        self.total_wfs_size, \
            self.value_left_up_1, self.value_left_up_2, self.value_right_down_1, self.value_right_down_2, \
            self.pupil_mask_padded, self.wfs_image_pad_value, self.unet_size, \
            self.num_pix_wfs, self.num_pix_phase, self.pupil_mask, self.pupil_mask_padded, \
            self.phase_pad_value = self.obtain_geometry_values(dataroot, use_voltage_as_phase)

        self.save_dataset_init(save_dir)

    def obtain_norm_values(self, dataroot, use_voltage_as_phase,
                           save_dir):
        df_norm = pd.read_csv(dataroot + "/info.csv")

        if save_dir is not None:  # evaluation will be none
            df_norm.to_csv(save_dir + '/info.csv')

        if use_voltage_as_phase:
            min_phase = df_norm['Min voltage'].values[0]
            max_phase = df_norm['Max voltage'].values[0]
        else:
            min_phase = df_norm['Min phase'].values[0]
            max_phase = df_norm['Max phase'].values[0]

        min_wfs_image = df_norm['Min wfs'].values[0]
        max_wfs_image = df_norm['Max wfs'].values[0]
        if self.normalization_noise:
            min_wfs_image = 0

        scale_phase_modified = None
        min_phase_modified = None
        scale_wfs = (max_wfs_image - min_wfs_image)
        scale_phase = (max_phase - min_phase)
        if self.normalization_095_005:
            scale_phase_modified = (max_phase - min_phase) / 0.9
            min_phase_modified = min_phase - 0.05 * scale_phase_modified

        return max_wfs_image, min_wfs_image, max_phase, min_phase, \
               scale_wfs, scale_phase, scale_phase_modified, min_phase_modified

    def obtain_geometry_values(self, dataroot,
                               use_voltage_as_phase):

        if use_voltage_as_phase:
            num_pix_phase = 40  # 40 because its commands
            unet_size = 64  # 128 max(wfs,phase)
            pupil_mask = np.load(dataroot + '/mask_valid_commands.npy')
            wfs_image_pad_value = 0
        else:
            num_pix_phase = 448  # 448
            unet_size = 512  # 512
            pupil_mask = np.load(dataroot + '/pupil.npy')
            wfs_image_pad_value = int(unet_size / 2) - 32  # 192

        total_wfs_size = 256
        num_pix_wfs = 56  # full size 256, divide by 2 128, we have 56 pix in the par file
        edge_offset_wfs = 44
        center_offset_wfs = 56
        # for padding wfs
        value_left_up_1 = edge_offset_wfs - 4
        value_left_up_2 = edge_offset_wfs + num_pix_wfs + 4
        value_right_down_1 = edge_offset_wfs + num_pix_wfs + center_offset_wfs - 4
        value_right_down_2 = -edge_offset_wfs + 4

        phase_pad_value = int(unet_size / 2) - int(num_pix_phase / 2)  # 33
        if phase_pad_value >= 0:
            pupil_mask_padded = np.pad(pupil_mask,
                                       ((phase_pad_value, phase_pad_value),
                                        (phase_pad_value, phase_pad_value)),
                                        'constant')
        else:
            pupil_mask_padded = None

        return total_wfs_size, \
               value_left_up_1, value_left_up_2, value_right_down_1, value_right_down_2, \
               pupil_mask_padded, wfs_image_pad_value, unet_size,\
               num_pix_wfs, num_pix_phase, pupil_mask, pupil_mask_padded, phase_pad_value

    def save_dataset_init(self, save_dir):
        if save_dir is not None:
            def log_info():
                info = []
                info.append(
                    "----------------------------------------------------------------------------------------------")
                info.append(f"+ Dataroot: {self.dataroot}")
                info.append(f"+ Dataset length: {len(self.dataset_paths)}")
                info.append(f"+ Input_nc: {self.input_nc}")
                info.append(f"+ Output_nc: {self.output_nc}")
                info.append(f"+ Unet size: {self.unet_size}")
                info.append(f"+ Number of pixels (WFS): {self.num_pix_wfs}")
                info.append(f"+ Number of pixels (Phase): {self.num_pix_phase}")
                info.append(f"+ Total WFS size: {self.total_wfs_size}")
                info.append(f"+ Scale Phase modified: {self.scale_phase_modified}")
                info.append(f"+ Min Phase modified: {self.min_phase_modified}")
                info.append(f"+ Scale WFS: {self.scale_wfs}")
                info.append(f"+ Scale Phase: {self.scale_phase}")
                info.append(f"+ Max Phase: {self.max_phase}")
                info.append(f"+ Max WFS image: {self.max_wfs_image}")
                info.append(f"+ Min Phase: {self.min_phase}")
                info.append(f"+ Min WFS image: {self.min_wfs_image}")
                info.append(f"+ No_subtract_mean_from_phase: {self.no_subtract_mean_from_phase}")
                info.append(
                    "----------------------------------------------------------------------------------------------")
                return "\n".join(info)
            # Generate the log info
            log_text = log_info()
            print(log_text)
            with open(save_dir + '/logfile.txt', 'w') as f:
                f.write(log_text)

    def pad_expand(self, wfs_image):
        if self.wfs_image_pad_value > 0:
            wfs_image = np.pad(wfs_image, ((self.wfs_image_pad_value, self.wfs_image_pad_value),
                                           (self.wfs_image_pad_value, self.wfs_image_pad_value)), 'constant')
        wfs_image = np.expand_dims(wfs_image, axis=0)
        return wfs_image

    def prepare_wfs_image(self, wfs_image):
        """
        Prepare the input for training
        """

        if self.normalization_noise:
            # Things below readout_noise... maybe forget about them?
            wfs_image = wfs_image - self.readout_noise
            wfs_image[wfs_image < 0] = 0

        # Remove extra dimensions
        wfs_image = np.squeeze(wfs_image)

        # Cut each wfs image into its quadrant
        # Upper left
        wfs_channel_1 = wfs_image[self.value_left_up_1:self.value_left_up_2,
                                  self.value_left_up_1:self.value_left_up_2]
        # Lower left
        wfs_channel_2 = wfs_image[self.value_left_up_1:self.value_left_up_2,
                                  self.value_right_down_1:self.value_right_down_2]
        # Upper right
        wfs_channel_3 = wfs_image[self.value_right_down_1:self.value_right_down_2,
                                  self.value_left_up_1:self.value_left_up_2]
        # Lower right
        wfs_channel_4 = wfs_image[self.value_right_down_1:self.value_right_down_2,
                                  self.value_right_down_1:self.value_right_down_2]

        # Pad, important before normalization because we are not centered at 0, we are centered at min_wfs_image
        wfs_channel_1 = self.pad_expand(wfs_channel_1)
        wfs_channel_2 = self.pad_expand(wfs_channel_2)
        wfs_channel_3 = self.pad_expand(wfs_channel_3)
        wfs_channel_4 = self.pad_expand(wfs_channel_4)
        # Concatenate
        wfs_image_multiple_channels = np.concatenate([wfs_channel_1, wfs_channel_2, wfs_channel_3, wfs_channel_4],
                                                      axis=0)

        wfs_image_multiple_channels_norm = (wfs_image_multiple_channels - self.min_wfs_image) / self.scale_wfs

        return wfs_image_multiple_channels_norm

    def prepare_phase(self, phase):
        """
        Prepare the output for training
        """
        phase[self.pupil_mask == 1] -= phase[self.pupil_mask == 1].mean()
        if self.normalization_095_005:
            phase = phase / self.scale_phase_modified  # NOP - self.min_phase_modified
        else:
            phase = phase / self.scale_phase  # 2 * - 1 For now -1 and 1 NOP - self.min_phase

        phase = np.multiply(phase, self.pupil_mask)
        phase = np.pad(phase, ((self.phase_pad_value, self.phase_pad_value),
                               (self.phase_pad_value, self.phase_pad_value)), 'constant')
        phase = np.expand_dims(phase, axis=0)
        return phase

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        data_path = self.dataset_paths[index]
        h5py_bool = True
        if h5py_bool:
            with h5py.File(data_path, 'r') as hdf5_data:
                wfs_image_raw = hdf5_data['arr_0'][:]
                if self.use_voltage_as_phase:
                    phase_raw = hdf5_data['arr_2'][:]
                else:
                    phase_raw = hdf5_data['arr_1'][:]
        else:
            data_array = np.load(data_path)
            wfs_image_raw = data_array['arr_0']
            phase_raw = data_array['arr_2'] if self.use_voltage_as_phase else data_array['arr_1']

        x = self.prepare_wfs_image(wfs_image_raw)
        y = self.prepare_phase(phase_raw)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)

        return x, y

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset_paths)