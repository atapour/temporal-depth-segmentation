import os
import glob
from collections import OrderedDict
import torch
from utils.segmentation import CrossEntropyLoss2d, colorize_segmentaions
from utils.depth import scale_pyramid, get_smooth_weight
from utils.optical_flow import estimate_flow, EPE
from model.networks import Encoder, Decoder_part_one, Decoder_part_two, Discriminator, FlowNetwork, AdvLoss

#-----------------------------------------
class TheModel():

    # this initializes all requirements for the model
    def __init__(self, args):

        self.args = args
        self.device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')

        self.loss_depth = 0
        self.loss_segmentation = 0

        self.l1_weight = args.l1_weight
        self.adv_weight = args.adv_weight
        self.smooth_weight = args.smooth_weight
        self.epe_weight = args.epe_weight
        self.segmentation_weight = args.segmentation_weight

        # losses that will be sent to the standard output for printing and visdom for plotting.
        self.loss_names = ['depth_L1', 'depth_adv', 'depth_smooth', 'epe_depth', 'segmentation']

        self.encoder_rgb = Encoder(args.input_nc).to(self.device)
        self.discriminator = Discriminator(args.input_nc + 1).to(self.device)
        self.encoder_depth = Encoder(1).to(self.device)
        self.decoder_1 = Decoder_part_one().to(self.device)
        self.decoder_2_depth = Decoder_part_two().to(self.device)
        self.decoder_2_segmentation = Decoder_part_two(num_classes=args.num_classes).to(self.device)

        self.num_frames = args.num_frames + 1
        self.checkpoint_save_dir = os.path.join(args.checkpoints_dir, args.name)

        # making the optimizers:
        self.optimizer_G_shared = torch.optim.Adam(list(self.encoder_rgb.parameters()) + list(self.encoder_depth.parameters()) + list(self.decoder_1.parameters()), lr=args.lr)
        self.optimizer_G_segmentation = torch.optim.Adam(self.decoder_2_segmentation.parameters(), lr=args.lr)
        self.optimizer_G_depth = torch.optim.Adam(self.decoder_2_depth.parameters(), lr=args.lr)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr)

        # creating the optical flow network and loading the pre-trained weights
        assert args.flow_model is not None

        if args.flow_model == 'sintel-clean':
            flowC = 'C'
        elif args.flow_model == 'sintel-final':
            flowC = 'F'
        elif args.flow_model == 'kitti':
            flowC = 'K'

        self.flowNetwork = FlowNetwork(flowC).to(self.device)
        flow_checkpoint = torch.load(os.path.join('./flow-weights', flowC + '.pth'))
        self.flowNetwork.load_state_dict(flow_checkpoint['model'])

        self.criterion_l1 = torch.nn.L1Loss()

        # label zero is to be ignored. synthia data
        weight = torch.ones(args.num_classes)
        weight[0] = 0

        self.criterion_CE = CrossEntropyLoss2d(weight.to(self.device))
        self.criterion_adv = AdvLoss(args.adv_loss).to(self.device)

    # this function sets up the model by loading and printing the model if necessary
    def set_up(self, args):
        if args.resume:
            if not os.listdir(self.checkpoint_save_dir):
                raise Exception('The specified checkpoints directory is empty. Resuming is not possible.')
            if args.which_checkpoint == 'latest':
                checkpoints = glob.glob(os.path.join(self.checkpoint_save_dir, '*.pth'))
                checkpoints.sort()
                latest = checkpoints[-1]
                step = latest.split('_')[1]
            elif args.which_checkpoint != 'latest' and args.which_checkpoint.isdigit():
                step = args.which_checkpoint
            else:
                raise Exception('The specified checkpoint is invalid.')
            self.load_networks(step)

        self.print_networks()

    # data inputs are assigned
    def assign_inputs(self, data):

        self.rgbs = data['rgb']
        self.fake_depths = [None] * self.num_frames
        self.fake_labels = [None] * self.num_frames
        self.real_depths = data['depth']
        self.real_labels = data['gt']
        self.real_depth_flows = [None] * (self.num_frames - 1)
        self.fake_depth_flows = [None] * (self.num_frames - 1)

    # forward pass
    def forward(self):

        encoded_rgb_curr = self.encoder_rgb(self.rgb_curr)
        encoded_rgb_prev = self.encoder_rgb(self.rgb_prev)
        encoded_depth_prev_fake = self.encoder_depth(self.depth_prev_fake)

        # re-arranging the feature lists so they can be passed into the decoder parts.
        # this is very messy. I don't like it, but can't be arsed to fix it now.
        ix7s = [encoded_rgb_curr[6], encoded_rgb_prev[6], encoded_depth_prev_fake[6]]
        ix6s = [encoded_rgb_curr[5], encoded_rgb_prev[5], encoded_depth_prev_fake[5]]
        ix5s = [encoded_rgb_curr[4], encoded_rgb_prev[4], encoded_depth_prev_fake[4]]
        ix4s = [encoded_rgb_curr[3], encoded_rgb_prev[3], encoded_depth_prev_fake[3]]
        ix3s = [encoded_rgb_curr[2], encoded_rgb_prev[2], encoded_depth_prev_fake[2]]
        ix2s = [encoded_rgb_curr[1], encoded_rgb_prev[1], encoded_depth_prev_fake[1]]
        ix1s = [encoded_rgb_curr[0], encoded_rgb_prev[0], encoded_depth_prev_fake[0]]

        # the features above that come from the encoder are passed into the decoder:
        decoder_1_output = self.decoder_1(ix7s, ix6s, ix5s, ix4s)
        # As there are four scales of depth, for now at least, we train the discriminator on the largest scale only
        self.depth_fake_pyramid = self.decoder_2_depth(decoder_1_output, ix3s, ix2s, ix1s)
        self.depth_curr_fake, _, _, _ = self.depth_fake_pyramid

        # current segmentation frame generate by the model:
        self.label_curr_fake = self.decoder_2_segmentation(decoder_1_output, ix3s, ix2s, ix1s)

    # this function will recurrently go through the frames
    def recur(self):

        # This loop reads through the loaded frames
        for i in range(0, self.num_frames - 1):

            self.label_curr_real = self.real_labels[i+1].to(self.device)

            # depth ground truth of the current frame:
            self.depth_curr_real = self.real_depths[i+1].to(self.device)

            # rgb of the previous frame
            self.rgb_prev = self.rgbs[i].to(self.device)
            # rgb of the current frame
            self.rgb_curr = self.rgbs[i+1].to(self.device)

            # if we are at the beginning of the sequence, the previous depth has not been computed yet, so we use the ground truth depth of the previous depth frame. If we are not at the beginning of the loop then it must be there is the fake depths list.
            if i == 0:
                self.fake_depths[i] = self.real_depths[i]

            # fake generated depth from the last time step:
            self.depth_prev_fake = self.fake_depths[i].to(self.device)

            self.forward()

            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            self.forward()

            # putting the generated depth frame into the fake depth list
            self.fake_depths[i+1] = self.depth_curr_fake
            # putting the generated segmentation frame into the fake segmentation list
            self.fake_labels[i+1] = self.label_curr_fake

            # calculating the optical flow for real and fake depth images
            flow_depth_real = estimate_flow(((self.depth_curr_real)), ((self.real_depths[i])), self.flowNetwork)
            flow_depth_fake = estimate_flow(((self.depth_curr_fake)), ((self.depth_prev_fake)), self.flowNetwork)
            # calculating the optical flow loss for depth frames
            self.loss_epe_depth = EPE(flow_depth_fake, flow_depth_real)

            self.depth_real_pyramid = scale_pyramid(self.depth_curr_real, 4)
            self.rgb_pyramid = scale_pyramid(self.rgb_curr, 4)

            self.loss_depth_L1 = 0
            for (depth_fake_i, depth_real_i) in zip(self.depth_fake_pyramid, self.depth_real_pyramid):
                self.loss_depth_L1 += self.criterion_l1(depth_fake_i, depth_real_i)

            self.dis_out = self.discriminator(torch.cat([self.rgb_curr, self.depth_curr_fake], 1))

            self.loss_depth_adv = ((1 - self.dis_out) ** 2).mean()

            self.loss_depth_smooth = get_smooth_weight(self.depth_fake_pyramid, self.rgb_pyramid, len(self.rgb_pyramid) - 1)

            self.loss_depth_L1 = self.l1_weight * self.loss_depth_L1
            self.loss_depth_adv = self.adv_weight * self.loss_depth_adv
            self.loss_depth_smooth = self.smooth_weight * self.loss_depth_smooth
            self.loss_epe_depth = self.epe_weight * self.loss_epe_depth

            self.loss_depth = self.loss_depth_L1 + self.loss_depth_adv + self.loss_depth_smooth + self.loss_epe_depth
            self.loss_segmentation = self.segmentation_weight * self.criterion_CE(self.label_curr_fake, self.label_curr_real[:, 0])

            # If you have enough memory on your GPU, the step should be taken outside this for loop, so that every whole recurrence loop translates to a single step with the gradients preserved.
            self.encoder_rgb.zero_grad()
            self.encoder_depth.zero_grad()
            self.decoder_1.zero_grad()
            self.decoder_2_depth.zero_grad()
            self.decoder_2_segmentation.zero_grad()

            self.loss_depth.backward(retain_graph=True)
            self.loss_segmentation.backward(retain_graph=True)

            self.optimizer_G_shared.step()
            self.optimizer_G_depth.step()
            self.optimizer_G_segmentation.step()

    # backward pass to train the discriminator with the loss
    def backward_D(self):

        # to train the discriminator on real RGB-D sample from the current frame:
        dis_out_real = self.discriminator(torch.cat([self.rgb_curr, self.depth_curr_real], 1))
        loss_D_real = self.criterion_adv(dis_out_real, True)

        # now to train the discriminator on the fake RGB-D sample from the current frame:
        dis_out_fake = self.discriminator(torch.cat([self.rgb_curr, self.depth_curr_fake.detach()], 1).detach())
        loss_D_fake = self.criterion_adv(dis_out_fake, False)

        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()

    # this function saves model checkpoints to disk
    def save_networks(self, step):

        save_filename = 'checkpoint_%s_steps.pth' % (step)
        save_path = os.path.join(self.checkpoint_save_dir, save_filename)

        print(f'saving the checkpoint to {save_path}.')

        torch.save({'state_dict_encoder_rgb': self.encoder_rgb.state_dict(),
                    'state_dict_encoder_depth': self.encoder_depth.state_dict(),
                    'state_dict_decoder_1': self.decoder_1.state_dict(),
                    'state_dict_decoder_2_depth': self.decoder_2_depth.state_dict(),
                    'state_dict_decoder_2_segmentation': self.decoder_2_segmentation.state_dict(),
                    'state_dict_discriminator': self.discriminator.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    'optimizer_G_depth_state_dict': self.optimizer_G_depth.state_dict(),
                    'optimizer_G_segmentation_state_dict': self.optimizer_G_segmentation.state_dict()}, save_path)

    # this function loads model checkpoints from disk
    def load_networks(self, step):

        load_filename = 'checkpoint_%s_steps.pth' % (step)
        load_path = os.path.join(self.checkpoint_save_dir, load_filename)

        print(f'loading the checkpoint from {load_path}.')

        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.encoder_rgb.load_state_dict(state_dict['state_dict_encoder_rgb'])
        self.encoder_depth.load_state_dict(state_dict['state_dict_encoder_depth'])
        self.decoder_1.load_state_dict(state_dict['state_dict_decoder_1'])
        self.decoder_2_depth.load_state_dict(state_dict['state_dict_decoder_2_depth'])
        self.decoder_2_segmentation.load_state_dict(state_dict['state_dict_decoder_2_segmentation'])
        self.discriminator.load_state_dict(state_dict['state_dict_discriminator'])
        self.optimizer_D.load_state_dict(state_dict['optimizer_D_state_dict'])
        self.optimizer_G_depth.load_state_dict(state_dict['optimizer_G_depth_state_dict'])
        self.optimizer_G_segmentation.load_state_dict(state_dict['optimizer_G_segmentation_state_dict'])

    # this function prints the network parameter count
    def print_networks(self):

        num_params = list(self.encoder_rgb.parameters()) + list(self.encoder_depth.parameters()) + list(self.decoder_1.parameters()) + list(self.decoder_2_segmentation.parameters()) + list(self.decoder_2_depth.parameters()) + list(self.discriminator.parameters())
        nl = '\n'
        print(f'There are {(sum([p.numel() for p in num_params]))} parameters in the whole model!{nl}')

    def get_loss(self):
        errors_ret = OrderedDict()

        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))

        return errors_ret

    # this function returns the images involved in the training for saving and displaying
    def get_images(self):

        im_ret = OrderedDict()

        # This loop reads through the loaded frames
        for i in range(1, self.num_frames - 1):

            # to colorize the segmentation output and the ground-truth:
            fake_label, real_label = colorize_segmentaions(self.fake_labels[i].detach().cpu(), self.real_labels[i])

            fake_label = fake_label.type(torch.FloatTensor)
            real_label = real_label.type(torch.FloatTensor)

            im_ret[f'rgb_{i}'] = self.rgbs[i]
            im_ret[f'real_depth_{i}'] = self.real_depths[i].detach()
            im_ret[f'fake_depth_{i}'] = self.fake_depths[i].detach()
            im_ret[f'real_label_{i}'] = real_label
            im_ret[f'fake_label_{i}'] = fake_label

        return im_ret
