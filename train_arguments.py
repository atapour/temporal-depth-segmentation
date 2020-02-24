import argparse

class Arguments():
    def __init__(self):

        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--name', required=True, type=str, help='experiment name')
        parser.add_argument('--phase', default='train', type=str, choices=['train', 'test'], help='determining whether the model is being trained or used for inference. Since this is the train_arguments file, this needs to set to train!!')
        parser.add_argument('--data_root', default='../../Data/synthia/processed-newer', type=str, help='path to data directory.')
        parser.add_argument('--batch_size', default=2, type=int, help='It is the size of your batch.')
        parser.add_argument('--num_epochs', default=500, type=int, help='number of training epochs to run')
        parser.add_argument('--num_frames', default=4, type=int, help='number of frames drawn to use for their temporal information')
        parser.add_argument('--num_classes', default=14, type=int, help='number of classes in the segmentation problem.')
        parser.add_argument('--flow_model', type=str, default='kitti', choices=['sintel-clean', 'sintel-final', 'kitti'], help='decide what model to use for the optical flow network; the options are sintel-clean, sintel-final, and kitti.')
        parser.add_argument('--input_nc', default=3, type=int, help='number of channels in the input image. This determines wether the images is RGB, Grayscale, or whatever weird other type!')
        parser.add_argument('--output_nc', default=64, type=int, help='number of channels in the output image.')
        parser.add_argument('--num_downs', default=8, type=int, help='number of downscaling done within the architecture')
        parser.add_argument('--num_workers', default=2, type=int, help='number of workers used in the dataloader.')
        parser.add_argument('--ngf', default=32, type=int, help='number of filters in first convolutional layer in the network.')
        parser.add_argument('--lr', default=0.001, type=int, help='learning rate')
        parser.add_argument('--l1_weight', default=200.0, type=float, help='L1 loss weight')
        parser.add_argument('--adv_weight', default=100.0, type=float, help='Adversarial loss weight')
        parser.add_argument('--smooth_weight', default=10.0, type=float, help='Smoothness loss weight')
        parser.add_argument('--epe_weight', default=0.1, type=float, help='Optical flow loss weight')
        parser.add_argument('--segmentation_weight', default=10, type=float, help='Segmentation loss weight')
        parser.add_argument('--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='which is training being done on.')
        parser.add_argument('--transform', default='resize', type=str, choices=['resize', 'crop'], help='inputs to the network must be 1024x256. You can choose to resize or crop them to those dimenstion.')
        parser.add_argument('--adv_loss', default='lsgan', type=str, choices=['lsgan', 'og', 'w'], help='determines the type of GAN loss. lsgan is the MSE loss advocated by cycleGAN, og is the original loss, and w is the wasserstien metric.')
        parser.add_argument('--resume', action='store_true', help='resume training from most recent checkpoint.')
        parser.add_argument('--which_checkpoint', type=str, default='latest', help='the checkpoint to be loaded to resume training. Checkpoints are identified and saved by the number of steps passed during training.')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='the path to where the model is saved.')
        parser.add_argument('--display', action='store_true', help='display the results periodically via visdom')
        parser.add_argument('--print_freq', default=100, type=int, help='how many steps before printing the loss values to the standard output for inspection purposes only.')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for visdom.')
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen using visdom.')
        parser.add_argument('--display_ncols', type=int, default=5, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display.')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display.')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main").')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display.')
        parser.add_argument('--save_checkpoint_freq', default=5000, type=int, help='how many steps before saving one sequence of images to disk for inspection purposes only.')

        self.initialized = True

        return parser

    def get_args(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_args(self, args):

        txt = '\n'

        txt += '-------------------- Arguments --------------------\n'

        for k, v in sorted(vars(args).items()):

            comment = ''
            default = self.parser.get_default(k)

            if v != default:
                comment = '\t[default: %s]' % str(default)

            txt += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)

        txt += '----------------------- End -----------------------'
        txt += '\n'

        print(txt)

    def parse(self):

        args = self.get_args()
        self.print_args(args)
        self.args = args

        return self.args
