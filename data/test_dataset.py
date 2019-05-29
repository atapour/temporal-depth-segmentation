import os
import os.path
import torch.utils.data as data
from PIL import Image

class TestDataset(data.Dataset):

    def __init__(self, root, transform_rgb=[]):

        self.transform_rgb = transform_rgb
        self.rgb_paths = []

        assert os.path.isdir(root), '%s is not a valid directory' % root

        for r, _, fnames in sorted(os.walk(root)):
            for fname in fnames:
                if self.is_png_image_file(fname):
                    path = os.path.join(r, fname)
                    self.rgb_paths.append(path)
                    self.rgb_paths.sort()

    def __getitem__(self, index):

        rgb_path = self.rgb_paths[index]

        rgb = Image.open(rgb_path).convert('RGB')
        rgb_size = rgb.size
        rgb = self.transform_rgb(rgb)

        return {'rgb': rgb, 'rgb_path': rgb_path, 'rgb_size': rgb_size}

    def __len__(self):
        return len(self.rgb_paths)

    def is_png_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])
