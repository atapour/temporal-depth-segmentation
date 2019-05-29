import PIL
from PIL import Image
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToTensor, Compose, Grayscale, ColorJitter, Resize
from data.train_dataset import TemporalDataset
from data.test_dataset import TestDataset

#-----------------------------------------

# this function can help in collating batches if and when dataloading goes wrong
def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
#-----------------------------------------

# this function takes a PIL image and center crops it
def center_crop_to_256_factor(img):

    w, h = img.size
    return img.crop((w//2-512, h//2-128, w//2+512, h//2+128))
#-----------------------------------------

# this function takes a tensor and turns it into a long tensor if it is an int tensor. This one is used for the segmentation ground truth, so no divisions.
def to_longTensor_gt(img):
    # convert a IntTensor to a LongTensor
    if isinstance(img, torch.IntTensor):
        return img.type(torch.LongTensor)
    elif isinstance(img, torch.LongTensor):
        return img
    else:
        return img
#-----------------------------------------

# this function takes a tensor and turns it into a float tensor if it is an int tensor. this is used for the depth images so it devides the values
def to_floatTensor_depth(img):
    # converts an IntTensor to a FloatTensor
    if isinstance(img, torch.IntTensor):
        return img.type(torch.FloatTensor) / 65535.0
    elif isinstance(img, torch.FloatTensor):
        return img
#-----------------------------------------

# this class takes a torch tensor and adds some noise. This can be very helpful in preventing instability when thr input image is completely uniform, which of course cannot happen here. 
class AddNoise(object):

    def __init__(self,size):
        self.noise = torch.randn(*size) * 0.0000001

    def __call__(self,x):
        return x + self.noise
#-----------------------------------------

# this function returns the transforms applied to the images
def get_transform(args):

    if args.transform == 'resize':

        rgb_transforms = Compose([Resize((128, 512)), ToTensor(), AddNoise((3, 128, 512))])

        if args.phase == 'train':
            gt_transforms = Compose([Resize((128, 512), interpolation=PIL.Image.NEAREST), ToTensor(), to_longTensor_gt])
            depth_transforms = Compose([Resize((128, 512)), ToTensor(), to_floatTensor_depth])
        else:
            gt_transforms = []
            depth_transforms = []

    elif args.transform == 'crop':

        rgb_transforms = Compose([center_crop_to_256_factor, ToTensor(), AddNoise((3, 128, 512))])

        if args.phase == 'train':
            gt_transforms = Compose([center_crop_to_256_factor, ToTensor(), to_longTensor_gt])
            depth_transforms = Compose([center_crop_to_256_factor, ToTensor(), to_floatTensor_depth])
        else:
            gt_transforms = []
            depth_transforms = []

    else:
        if args.phase == 'train':
            raise ValueError('the value (%s) for --transform is not valid.' % args.transform)

    torchvision_transforms = []

    if args.phase == 'train':
        grayscale = Grayscale(num_output_channels=3)
        grayscale.probability = 0.075
        torchvision_transforms.append(grayscale)
        colorJitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        colorJitter.probability = 0.1
        torchvision_transforms.append(colorJitter)

        return rgb_transforms, torchvision_transforms, depth_transforms, gt_transforms

    elif args.phase == 'test':

        return rgb_transforms
#-----------------------------------------

# this function creates the dataset from the temporal dataset class
def create_dataset(args):

    if args.phase == 'train':

        rgb_transforms, torchvision_transforms, depth_transforms, gt_transforms = get_transform(args)
        return TemporalDataset(args.data_root, args.num_frames + 1, rgb_transforms, depth_transforms, gt_transforms, torchvision_transforms)

    elif args.phase == 'test':

        rgb_transforms = get_transform(args)
        return TestDataset(args.data_root, rgb_transforms)
#-----------------------------------------

# this function creates the dataloader
def create_loader(args):

    data_loader = DataLoader()
    data_loader.initialize(args)
    print("The data has been loaded.")
    return data_loader
#-----------------------------------------

# The DataLoader class based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class DataLoader():

    def initialize(self, args):

        self.dataset = create_dataset(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=args.phase == 'train', num_workers=int(args.num_workers), drop_last=args.phase == 'train', collate_fn=my_collate)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            yield data
#-----------------------------------------
