from collections import OrderedDict
import colorama
import torch
from utils.segmentation import colorize_segmentaions
from model.networks import Encoder, Decoder_part_one, Decoder_part_two
#-----------------------------------------
 # setting up the pretty colors:
reset = colorama.Style.RESET_ALL
blue = colorama.Fore.BLUE
red = colorama.Fore.RED

#-----------------------------------------
class TheModel():
    def __init__(self, args):

        self.args = args
        self.phase = args.phase
        self.device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')
        self.encoder_rgb = Encoder(args.input_nc).to(self.device)
        self.encoder_depth = Encoder(1).to(self.device)
        self.decoder_1 = Decoder_part_one().to(self.device)
        self.decoder_2_depth = Decoder_part_two().to(self.device)
        self.decoder_2_segmentation = Decoder_part_two(num_classes=args.num_classes).to(self.device)
        self.results_path = args.results_path

    # this function sets up the model by loading and printing the model if necessary
    def set_up(self, args):
        if args.test_checkpoint_path is not None:

            print(f'loading the checkpoint from {red}{args.test_checkpoint_path}{reset}.')

            state_dict = torch.load(args.test_checkpoint_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            self.encoder_rgb.load_state_dict(state_dict['state_dict_encoder_rgb'])
            self.encoder_depth.load_state_dict(state_dict['state_dict_encoder_depth'])
            self.decoder_1.load_state_dict(state_dict['state_dict_decoder_1'])
            self.decoder_2_depth.load_state_dict(state_dict['state_dict_decoder_2_depth'])
            self.decoder_2_segmentation.load_state_dict(state_dict['state_dict_decoder_2_segmentation'])

        else:
            raise Exception('For inference, a checkpoint path must be passed as an argument.')

        self.print_networks()

    # data inputs are assigned
    def assign_inputs(self, data):

        self.rgb = data['rgb'].to(self.device)
        self.rgb_path = data['rgb_path']
        self.rgb_size = data['rgb_size']
        self.rgb_size = (int(self.rgb_size[0]), int(self.rgb_size[1]))

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

    # this function will be used to go through the frames for testing purposes
    def test(self, i):

        if i == 0:

            self.rgb_prev = self.rgb
            # Since no depth is available to be used as the input during test time at the first step, we prime the process with a grayscale RGB image of the current frame.
            self.depth_prev_fake = torch.unsqueeze(0.2989 * self.rgb[:, 0, :, :] + 0.5870 * self.rgb[:, 1, :, :] + 0.1140 * self.rgb[:, 2, :, :], 0)

        else:

            self.rgb_curr = self.rgb
            self.forward()

            # to keep the steps going in the next iteration:
            self.depth_prev_fake = self.depth_curr_fake
            self.rgb_prev = self.rgb_curr

    # this function prints the network parameter count
    def print_networks(self):

        num_params = list(self.encoder_rgb.parameters()) + list(self.encoder_depth.parameters()) + list(self.decoder_1.parameters()) + list(self.decoder_2_segmentation.parameters()) + list(self.decoder_2_depth.parameters())
        nl = '\n'
        print(f'{blue}There are {red}{(sum([p.numel() for p in num_params]))}{blue} parameters in the whole model{reset}!{nl}')

    # this function returns the output image and the RGB image during testing
    def get_test_outputs(self):

        im_ret = OrderedDict()
        im_ret['rgb'] = self.rgb

        # to colorize the segmentation output and the ground-truth:
        fake_label, _ = colorize_segmentaions(self.label_curr_fake.detach().cpu(), self.label_curr_fake.detach().cpu(), test='test')
        fake_label = fake_label.type(torch.FloatTensor)

        im_ret['segmentation'] = fake_label
        im_ret['depth'] = self.depth_curr_fake

        return im_ret

    # this function returns RGB image path to save the image during testing
    def get_test_paths(self):
        return self.rgb_path, self.results_path

    # this function returns the size of the image so it can be resized properly before saving
    def get_image_size(self):
        return self.rgb_size
 