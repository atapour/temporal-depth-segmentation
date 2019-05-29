import time
import colorama
from train_arguments import Arguments
from data import create_loader
from model import create_model
from utils.general import Display

# setting up the colors:
reset = colorama.Style.RESET_ALL
green = colorama.Fore.GREEN
red = colorama.Fore.RED

args = Arguments().parse()
data_loader = create_loader(args)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

nl = '\n'
print(f'{green}There are a total number of {red}{dataset_size}{green} sequences of {red}{args.num_frames}{green} frames in the training set.{reset}{nl}')

model = create_model(args)
model.set_up(args)
display = Display(args)
global_step = 0
total_steps = 0

print(f'Training has begun!{nl}')

for epoch in range(0, args.num_epochs):
    data_time_start = time.time()

    for j, data in enumerate(data_loader):
        processing_time_start = time.time()

        if global_step % args.print_freq == 0:
            t_data = processing_time_start - data_time_start

        total_steps += args.batch_size
        model.assign_inputs(data)
        model.recur()

        if global_step % args.display_freq == 0:
            display.display_current_results(model.get_images())

        if global_step % args.print_freq == 0:
            loss = model.get_loss()
            t_proc = (time.time() - processing_time_start) / args.batch_size
            display.print_current_loss(epoch, global_step, loss, t_proc, t_data)
            if args.display_id > 0:
                display.plot_current_loss(epoch, float(total_steps) / dataset_size, loss)

        global_step += 1

        if total_steps % args.save_checkpoint_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save_networks(total_steps)
