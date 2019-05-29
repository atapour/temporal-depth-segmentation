import os
import os.path
import random
import bisect
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TemporalDataset(data.Dataset):
    def __init__(self, root, num_frames, transform_rgb=[], transform_depth=[], transform_gt=[], rgb_torchvision_transforms=[], phase='train'):

        self.num_frames = num_frames
        self.phase = phase
        self.root = os.path.join(root, phase)
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.transform_gt = transform_gt
        self.rgb_torchvision_transforms = rgb_torchvision_transforms

        self._rgb_path = os.path.join('%s', '%s', 'RGB', '%06d.png')
        self._depth_path = os.path.join('%s', '%s', 'Depth', '%06d.png')
        self._gt_path = os.path.join('%s', '%s', 'GT', '%06d.png')

        self.ids = list()
        self.cumulated_frames = [0]

        self.min_frames = 2000

        for seq_id in range(0, len(os.listdir(self.root))-1):
            num_frames = len(os.listdir(os.path.join(self.root, str(seq_id), 'RGB')))
            if phase == 'test' and num_frames < self.min_frames:
                self.min_frames = num_frames
            self.ids.append((self.root, seq_id, num_frames))
            self.cumulated_frames.append(self.cumulated_frames[-1]+num_frames)

    def __getitem__(self, index):

        num_drawn_frames = self.num_frames

        rectified_cumulated_frames = [self.cumulated_frames[i] - i * (num_drawn_frames - 1) for i in range(len(self.cumulated_frames))]

        random_index = random.randint(0, rectified_cumulated_frames[-1] - 1)
        insert_idx = bisect.bisect_right(rectified_cumulated_frames, random_index)
        seq_id = insert_idx - 1
        start_index_in_seq = random_index - rectified_cumulated_frames[seq_id]

        rgbs = []
        depths = []
        gts = []

        for index_in_seq in range(start_index_in_seq, start_index_in_seq+num_drawn_frames):
            rgb, depth, gt = self.pull_item(seq_id, index_in_seq)

            for transform in self.rgb_torchvision_transforms:
                if random.random() < transform.probability:
                    rgb = transform(rgb)

            gt = self.convert_labels(gt)
            gt = gt.convert('I')

            rgb_out = self.transform_rgb(rgb)
            depth_out = self.transform_depth(depth)
            gt_out = self.transform_gt(gt)

            rgbs += [rgb_out]
            depths += [depth_out]
            gts += [gt_out]

        return {'rgb': rgbs, 'depth': depths, 'gt':gts, 'seq_id':seq_id, 'index_in_seq':index_in_seq}

    def __len__(self):
        num_drawn_frames = self.num_frames
        return self.cumulated_frames[-1] // num_drawn_frames

    def pull_item(self, seq_id, index_in_seq):

        root, seq_index, _ = self.ids[seq_id]

        rgb = Image.open(self._rgb_path % (root, seq_index, index_in_seq))
        depth = Image.open(self._depth_path % (root, seq_index, index_in_seq))
        gt = Image.open(self._gt_path % (root, seq_index, index_in_seq))

        return rgb, depth, gt

    # this function converts the Synthia labels so they are simpler
    def convert_labels(self, labels):

        labels = np.array(labels)

        # labels 13 and 14 were essentially void so we turn them to zeros and move label 15 which is 15 to 13. now synthia has 14 labels
        labels[labels == 13] = 0
        labels[labels == 14] = 0
        labels[labels == 15] = 13

        return Image.fromarray(labels)
