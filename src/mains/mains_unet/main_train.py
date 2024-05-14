# main_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from src.unet.dataset import Dataset
from src.unet.unet import UnetGenerator, init_net
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from src.global_cte import FOLDER_SAVE_DATA_UNET, FOLDER_CHECKPOINTS_UNET

class TrainManager:
    def __init__(self,
                 experiment_name,
                 dataroot,
                 save_dir,
                 model_,
                 optimizer_,
                 batch_size_,
                 max_num_epochs_,
                 criterion_name_,
                 use_voltage_as_phase,
                 main_device,
                 normalization_noise,
                 max_dataset_size,
                 epsilon=0.002,
                 max_num_eval=20,
                 subtract_mean_from_phase=True,
                 save_frequency_=5):
        self.experiment_name = experiment_name
        self.save_dir = save_dir + self.experiment_name
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.max_num_epochs = max_num_epochs_
        self.model = model_
        self.use_voltage_as_phase = use_voltage_as_phase
        self.optimizer = optimizer_
        if max_dataset_size == math.inf:
            train_datasize = float("inf")
            eval_datasize = float("inf")
        else:
            train_datasize = int(0.8*max_dataset_size)
            eval_datasize= int(0.2*max_dataset_size)
        train_dataset = Dataset(dataroot=dataroot,
                                save_dir=self.save_dir, mode="train", use_voltage_as_phase=use_voltage_as_phase,
                                normalization_noise=normalization_noise, max_dataset_size=train_datasize,
                                no_subtract_mean_from_phase=not subtract_mean_from_phase)
        eval_dataset = Dataset(dataroot=dataroot,
                               save_dir=None, mode="evaluation", use_voltage_as_phase=use_voltage_as_phase,
                               normalization_noise=normalization_noise, max_dataset_size=eval_datasize,
                               no_subtract_mean_from_phase=not subtract_mean_from_phase)
        self.pupil_mask_padded = train_dataset.pupil_mask_padded.copy()
        self.pupil_mask = train_dataset.pupil_mask.copy()
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=batch_size_,
                                                            shuffle=True)
        self.eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                           batch_size=batch_size_,
                                                           shuffle=True)

        self.train_loss_list, self.eval_loss_list = [], []
        self.save_frequency = save_frequency_

        # Early stopping
        self.min_eval_loss = math.inf
        self.max_num_eval = max_num_eval
        self.early_stopping_counter = 0
        self.saved_weights = self.model.state_dict()
        self.device = main_device
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.2)
        self.criterion_name = criterion_name_
        self.epsilon = epsilon
        if self.criterion_name in ["MSE", "MSE_relative"]:
            self.criterion = nn.MSELoss()
        elif self.criterion_name in ["L1", "L1_relative"]:
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError

    def calculate_loss(self, predicted, real):
        norms = None
        if self.criterion_name in ["MSE", "MSE_relative"]:
            absolute_difference = torch.pow(
                predicted[:, :, self.pupil_mask_padded == 1] - real[:, :, self.pupil_mask_padded == 1], 2)
            if self.criterion_name == "MSE_relative":
                with torch.no_grad():
                    norms = torch.norm(real[:, :, self.pupil_mask_padded == 1], dim=(1, 2), p=2).view(-1, 1, 1) / (
                        len(self.pupil_mask_padded[self.pupil_mask_padded == 1])) + self.epsilon
                loss_per_element_in_batch = torch.sum(absolute_difference / norms, dim=(1, 2))
            else:
                loss_per_element_in_batch = torch.sum(absolute_difference, dim=(1, 2))
            loss = torch.mean(loss_per_element_in_batch)
        elif self.criterion_name in ["L1", "L1_relative"]:
            absolute_difference = torch.abs(
                predicted[:, :, self.pupil_mask_padded == 1] - real[:, :, self.pupil_mask_padded == 1])
            if self.criterion_name == "L1_relative":
                with torch.no_grad():
                    norms = torch.norm(real[:, :, self.pupil_mask_padded == 1], dim=(1, 2), p=1).view(-1, 1, 1) \
                            / (len(self.pupil_mask_padded[self.pupil_mask_padded == 1])) + self.epsilon
                loss_per_element_in_batch = torch.sum(absolute_difference / norms, dim=(1, 2))
            else:
                loss_per_element_in_batch = torch.sum(absolute_difference, dim=(1, 2))
            loss = torch.mean(loss_per_element_in_batch)
        else:
            raise NotImplementedError

        if norms is None:
            return loss, None
        else:
            return loss, norms.cpu().numpy()

    def train_epoch(self, epoch):
        self.model.train()
        train_running_loss = 0.0
        norms_max = 0
        norms_min = 100000
        for i, data in tqdm(enumerate(self.train_dataloader)):
            x, real = data
            x = x.to(self.device)
            real = real.to(self.device)

            self.optimizer.zero_grad()
            predicted = self.model(x)

            loss, norms = self.calculate_loss(predicted, real)
            loss.backward()
            self.optimizer.step()
            if norms is not None:
                norms_max = max(np.max(norms), norms_max)
                norms_min = min(np.min(norms), norms_min)
            train_running_loss += loss.item()

        self.plot_prediction_vs_real(real[0, 0, :, :], predicted[0, 0, :, :], epoch, mode="train")

        return train_running_loss/(i+1), norms_min, norms_max

    def save_network(self, epoch):
        save_filename = '%s_net_%s.pth' % (epoch, self.experiment_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.saved_weights, save_path)

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        eval_running_loss = 0.0
        for i, data in tqdm(enumerate(self.eval_dataloader)):
            x, real = data
            x = x.to(self.device)
            real = real.to(self.device)

            predicted = self.model(x)
            loss, _ = self.calculate_loss(predicted, real)

            eval_running_loss += loss.item()

        self.plot_prediction_vs_real(real[0, 0, :, :], predicted[0, 0, :, :], epoch, mode="evaluation")

        return eval_running_loss/(i+1)

    def check_early_stopping(self, eval_loss):
        """
        Check the evaluation loss if we are doing early stopping or not
        """
        done = False
        if eval_loss < self.min_eval_loss:
            self.min_eval_loss = eval_loss
            self.early_stopping_counter = 0
            self.saved_weights = self.model.state_dict()
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter > self.max_num_eval:
                done = True
        return done

    def plot_losses(self):
        """
        Plots the losses
        """
        plt.figure()
        plt.plot(self.train_loss_list, label="train")
        plt.plot(self.eval_loss_list, label="eval")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(self.save_dir + "/training_curves.png")

    def train(self):
        """
        Trains the network
        """
        for epoch in range(self.max_num_epochs):
            train_loss, norms_min, norms_max = self.train_epoch(epoch)
            eval_loss = self.eval_epoch(epoch)
            done = self.check_early_stopping(eval_loss)
            self.scheduler.step(eval_loss, epoch)
            self.train_loss_list.append(train_loss)
            self.eval_loss_list.append(eval_loss)
            print("+ Epoch: {} Train loss: {:.2g} Eval loss: {:.2g} Lr {:7f} Counter early stopping {} Min eval loss: {:.2g} Norms max {} min {}"
                  .format(epoch, train_loss, eval_loss, self.optimizer.param_groups[0]['lr'],
                          self.early_stopping_counter, self.min_eval_loss, norms_max, norms_min))
            if epoch % self.save_frequency == 0:
                self.save_network(epoch)
            if done:
                self.save_network(epoch)
                break

        self.plot_losses()
        return self.train_loss_list, self.eval_loss_list


    def plot_prediction_vs_real(self, real, predicted, epoch, mode="evaluation"):
        """
        Plots prediction of phase vs real phase
        """

        # Save
        image_dir_mode = self.save_dir + '/' + mode + "/"
        if not os.path.exists(image_dir_mode):
            os.mkdir(image_dir_mode)

        if self.train_dataloader.dataset.normalization_095_005:
            delta = self.train_dataloader.dataset.scale_phase_modified
            minim = self.train_dataloader.dataset.min_phase_modified
        else:
            delta = self.train_dataloader.dataset.scale_phase
            minim = self.train_dataloader.dataset.min_phase
        real_b = real.detach().cpu().numpy()
        real_b = (real_b * delta) + minim

        fake_b = predicted.detach().cpu().numpy()
        fake_b = (fake_b * delta) + minim

        difference = fake_b - real_b

        real_b[self.pupil_mask_padded == 0.] = np.nan
        fake_b[self.pupil_mask_padded == 0.] = np.nan
        difference[self.pupil_mask_padded == 0.] = np.nan

        # Plot phases

        fig, ax = plt.subplots(1, 3, figsize=[9, 3])
        plt.suptitle("Epoch: " + str(epoch))
        ax[0].imshow(fake_b)
        ax[1].imshow(real_b)
        ax[2].imshow(difference)
        if self.use_voltage_as_phase:
            ax[0].set_title(f"Inferred: {fake_b[self.pupil_mask_padded == 1].std():0.3f} volts um")
            ax[1].set_title(f"Real: {real_b[self.pupil_mask_padded == 1].std():0.3f} volts um")
            ax[2].set_title(f"Residual: {(difference[self.pupil_mask_padded == 1]).std():0.3f} volts um")
        else:
            ax[0].set_title(f"Inferred: {fake_b[self.pupil_mask_padded == 1].std():0.3f} volts RMS")
            ax[1].set_title(f"Real: {real_b[self.pupil_mask_padded == 1].std():0.3f} volts RMS")
            ax[2].set_title(f"Residual: {(difference[self.pupil_mask_padded == 1]).std():0.3f} volts RMS")

        image_name = '%s_%s' % ("infered_vs_pred", epoch)
        image_name = image_dir_mode + image_name
        plt.savefig(image_name + ".png")
        plt.close("all")


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    parser = argparse.ArgumentParser(description='Process .npz files and write max/min to CSV.')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPUs to use')
    parser.add_argument('--experiment_name', required=True, help='Experiment name.', type=str)
    parser.add_argument('--criterion', type=str, help="MSE | MSE_relative | L1 | L1_relative", default="L1_relative")
    parser.add_argument('--use_voltage_as_phase', action='store_true')
    parser.add_argument('--init_type', help="normal | xavier | kaiming | orthogonal", default='normal')
    parser.add_argument('--init_gain', help="scaling factor for normal, xavier and orthogonal", default=0.02)
    parser.add_argument('--initialize_last_layer_0', action='store_true')
    parser.add_argument('--normalization_noise', action='store_true')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"))
    parser.add_argument('--data_dir', type=str, default=FOLDER_SAVE_DATA_UNET)
    parser.add_argument('--data_name', required=True, help='Path to data files.', type=str)
    parser.add_argument('--save_dir', type=str, default=FOLDER_CHECKPOINTS_UNET)
    parser.add_argument('--max_num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--starting_lr', type=float, default=0.0002)
    parser.add_argument('--subtract_mean_from_phase', action='store_true')
    args = parser.parse_args()

    main_device = "cuda:" + str(args.gpu_ids[0])
    criterion_name = args.criterion
    assert criterion_name in ["MSE", "MSE_relative", "L1", "L1_relative"]

    if args.use_voltage_as_phase:
        model = UnetGenerator(input_nc=4, output_nc=1, num_downs=6, ngf=64)
    else:
        model = UnetGenerator(input_nc=4, output_nc=1, num_downs=9, ngf=64)

    model = init_net(model, args.init_type, args.init_gain, args.gpu_ids, args.initialize_last_layer_0)
    optimizer = optim.Adam(model.parameters(), lr=args.starting_lr)
    train_manager = TrainManager(experiment_name=args.experiment_name + "_" + criterion_name,
                                 dataroot=os.path.join(args.data_dir, args.data_name),
                                 save_dir=args.save_dir,
                                 model_=model,
                                 optimizer_=optimizer,
                                 batch_size_=args.batch_size,
                                 max_num_epochs_=args.max_num_epochs,
                                 criterion_name_=args.criterion,
                                 use_voltage_as_phase=args.use_voltage_as_phase,
                                 main_device=main_device,
                                 normalization_noise=args.normalization_noise,
                                 max_dataset_size=args.max_dataset_size,
                                 subtract_mean_from_phase=args.subtract_mean_from_phase)

    train_manager.train()
