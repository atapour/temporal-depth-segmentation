import colorama
import torch
from test_arguments import Arguments
from data import create_loader
from model import create_model
from utils.general import save_images

# setting up the colors:
reset = colorama.Style.RESET_ALL
blue = colorama.Fore.BLUE
red = colorama.Fore.RED

args = Arguments().parse()

args.phase = 'test'
args.batch_size = 1

data_loader = create_loader(args)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

nl = '\n'
print(f'{blue}There are a total number of {red}{dataset_size}{blue} frames in the data set.{reset}{nl}')

model = create_model(args)
model.set_up(args)

print(f'{red}Processing the frames has begun.. {reset}{nl}')

for j, data in enumerate(data_loader):
    with torch.no_grad():

        model.assign_inputs(data)
        model.test(j)

        if j == 0:
            print(f'The first frame is not processed. {red}I have my reasons!{reset}{nl}')
            continue

        output = model.get_test_outputs()
        img_path = model.get_test_paths()[0]

        print('%04d: processing image... %s' % (j, img_path))
        save_images(model.get_test_paths(), model.get_test_outputs(), size=model.get_image_size())
