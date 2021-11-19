from __future__ import absolute_import, division, print_function

import numpy as np
import time
import os
from glob import glob
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from utils import *
import datasets
import networks
from networks.LiteFlowNet3 import estimate, read_image, save_flow

from IPython import embed

# Link:
def collate_fn(batch):
    return tuple(zip(*batch))

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32

        # assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # Store all networks that will be used into self.models
        self.models["liteflownet3"] = networks.LiteFlowNet3().to(self.device)
        self.parameters_to_train += list(self.models["liteflownet3"].parameters())
        self.models["liteflownet3"].load_state_dict(torch.load('/home/hu440/LP-MOT/pytorch-liteflownet3/network-sintel.pytorch'))

        self.models["resnet"] = networks.ResNet(self.opt.num_layers,
                                                self.opt.weights_init == "pretrained").to(self.device)
        self.parameters_to_train += list(self.models["resnet"].parameters())

        self.models["attnet"] = networks.AttNet(num_input_ch=10).to(self.device)
        self.parameters_to_train += list(self.models["attnet"].parameters())

        # TODO: R-FCN or something similar

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # Data configuration
        # TODO: configuration inside self.dataset
        datasets_dict = {"visdrone": datasets.VisDroneDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        # TODO: The following are for LiteFlowNet3. Change it fit datasets universally
        visdrone_folders = [name for name in glob(os.path.join(self.opt.data_path, '*'))]
        train_csv, val_csv, test_csv = list(), list(), list()
        for folder in visdrone_folders:
            for txt in glob(os.path.join(folder, 'annotations', '*')):
                if 'train' in folder:
                    train_filenames = open(txt).readlines()
                    train_csv.append(txt)
                if 'val' in folder:
                    val_filenames = open(txt).readlines()
                    val_csv.append(txt)
                if 'test' in folder:
                    test_filenames = open(txt).readlines()

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
                        csv_files=train_csv,
                        root_dir=self.opt.data_path+"/VisDrone2019-MOT-train",
                        transform=transforms.Compose([datasets.Rescale((270, 480)), datasets.ToTensor()]))
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

        val_dataset = self.dataset(
                csv_files=val_csv,
                root_dir=self.opt.data_path+"/VisDrone2019-MOT-val",
                transform=transforms.Compose([datasets.Rescale((270, 480)), datasets.ToTensor()])
            )
        self.val_dataset = val_dataset
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print(" ==== Training ====")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        # Send all inputs to device
        for idx, input in enumerate(inputs):
            for ipt in input:
                if idx == 0: # image
                    ipt = ipt.to(self.device)
                else: # gt
                    for key, val in ipt.items():
                        ipt[key] = val.to(self.device)

        # TODO: implement the workflow as structured below
        # LiteFowNet3: 2 images -> optical flow map
        img1, img2 = inputs[0][0], inputs[0][1]
        img1_path, img2_path = self.train_dataset.index2file(inputs[1][0]['image_id']), self.train_dataset.index2file(inputs[1][1]['image_id'])

        opt_flow = estimate(self.models['liteflownet3'], img1, img2)
        save_flow(opt_flow, '/home/hu440/LP-MOT/LP-MOT/flow/out.flo')

        # ResNet: 1 image -> feature map
        with torch.no_grad():
            # self.models['resnet'] = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(self.device)
            res_output = self.models['resnet'](img2.unsqueeze(0).to(self.device))
        print(f'resnet: {res_output[-1].shape}\toptical flow:{opt_flow.shape}')
        breakpoint()

        # AttNet: OF map + feature map -> result map

        # R-FCN: result map -> ROIs

        # Classify: ROIs -> class + motion info (x, y)

        # if self.opt.pose_model_type == "shared":
        #     # If we are using a shared encoder for both depth and pose (as advocated
        #     # in monodepthv1), then all images are fed separately through the depth encoder.
        #     all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
        #     all_features = self.models["encoder"](all_color_aug)
        #     all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
        #
        #     features = {}
        #     for i, k in enumerate(self.opt.frame_ids):
        #         features[k] = [f[i] for f in all_features]
        #
        #     outputs = self.models["depth"](features[0])
        # else:
        #     # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        #     features = self.models["encoder"](inputs["color_aug", 0, 0])
        #     outputs = self.models["depth"](features)

        # self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        # for scale in self.opt.scales:
        #     loss = 0
        #     reprojection_losses = []
        #
        #     if self.opt.v1_multiscale:
        #         source_scale = scale
        #     else:
        #         source_scale = 0
        #
        #     disp = outputs[("disp", scale)]
        #     color = inputs[("color", 0, scale)]
        #     target = inputs[("color", 0, source_scale)]
        #
        #     for frame_id in self.opt.frame_ids[1:]:
        #         pred = outputs[("color", frame_id, scale)]
        #         reprojection_losses.append(self.compute_reprojection_loss(pred, target))
        #
        #     reprojection_losses = torch.cat(reprojection_losses, 1)
        #
        #     if not self.opt.disable_automasking:
        #         identity_reprojection_losses = []
        #         for frame_id in self.opt.frame_ids[1:]:
        #             pred = inputs[("color", frame_id, source_scale)]
        #             identity_reprojection_losses.append(
        #                 self.compute_reprojection_loss(pred, target))
        #
        #         identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
        #
        #         if self.opt.avg_reprojection:
        #             identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        #         else:
        #             # save both images, and do min all at once below
        #             identity_reprojection_loss = identity_reprojection_losses
        #
        #     elif self.opt.predictive_mask:
        #         # use the predicted mask
        #         mask = outputs["predictive_mask"]["disp", scale]
        #         if not self.opt.v1_multiscale:
        #             mask = F.interpolate(
        #                 mask, [self.opt.height, self.opt.width],
        #                 mode="bilinear", align_corners=False)
        #
        #         reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
            #     weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
            #     loss += weighting_loss.mean()
            #
            # if self.opt.avg_reprojection:
            #     reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            # else:
            #     reprojection_loss = reprojection_losses
            #
            # if not self.opt.disable_automasking:
            #     # add random numbers to break ties
            #     identity_reprojection_loss += torch.randn(
            #         identity_reprojection_loss.shape).cuda() * 0.00001
            #
            #     combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            # else:
            #     combined = reprojection_loss
            #
            # if combined.shape[1] == 1:
            #     to_optimise = combined
            # else:
            #     to_optimise, idxs = torch.min(combined, dim=1)
            #
            # if not self.opt.disable_automasking:
            #     outputs["identity_selection/{}".format(scale)] = (
            #         idxs > identity_reprojection_loss.shape[1] - 1).float()
            #
            # loss += to_optimise.mean()
            # mean_disp = disp.mean(2, True).mean(3, True)
            # norm_disp = disp / (mean_disp + 1e-7)
            # smooth_loss = get_smooth_loss(norm_disp, color)
            #
            # loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            # total_loss += loss
            # losses["loss/{}".format(scale)] = loss

        # total_loss /= self.num_scales
        # losses["loss"] = total_loss
        # return losses
        return None

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "network-sintel.pytorch")
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
