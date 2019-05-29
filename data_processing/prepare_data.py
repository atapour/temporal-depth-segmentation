import os
import os.path
import argparse
from multiprocessing import Process
import cv2

# this function makes a directory
def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# this function processes the depth image
def process_depth(input_depth_path, output_depth_path):
    depth = cv2.imread(input_depth_path, cv2.IMREAD_UNCHANGED)
    h_in, w_in, _ = depth.shape
    depth_cropped = depth[(h_in//2) - (h_out//2) : (h_in//2) + (h_out//2), (w_in//2) - (w_out//2) : (w_in//2) + (w_out//2)]
    depth_cropped_b, depth_cropped_g, depth_cropped_r = cv2.split(depth_cropped)
    cv2.imwrite(output_depth_path, depth_cropped_r)

# this function processes the RGB image
def process_rgb(input_rgb_path, output_rgb_path):

    rgb = cv2.imread(input_rgb_path, cv2.IMREAD_COLOR)
    h_in, w_in, _ = rgb.shape
    rgb_cropped = rgb[(h_in//2) - (h_out//2) : (h_in//2) + (h_out//2), (w_in//2) - (w_out//2) : (w_in//2) + (w_out//2)]
    cv2.imwrite(output_rgb_path, rgb_cropped)

# this function processes the class label image
def process_gt(input_gt_path, output_gt_path):
    gt = cv2.imread(input_gt_path, cv2.IMREAD_UNCHANGED)
    h_in, w_in, _ = gt.shape
    gt_cropped = gt[(h_in//2) - (h_out//2) : (h_in//2) + (h_out//2), (w_in//2) - (w_out//2) : (w_in//2) + (w_out//2)]                 
    gt_cropped_b, gt_cropped_g, gt_cropped_r = cv2.split(gt_cropped)
    cv2.imwrite(output_gt_path, gt_cropped_r)

parser = argparse.ArgumentParser('processing data!')

parser.add_argument('--input_root', type=str, default="/media/amir/storage2/Data/synthia/unrarred", help='point it towards the synthia root.')
parser.add_argument('--output_root', type=str, default="/media/amir/storage2/Data/synthia/new", help='point it towards where you want to copy everything. The directoty called processed-new will be made here.')

parser.add_argument('--height', type=int, default=256, help='height of the center crop')
parser.add_argument('--width', type=int, default=1024, help='width of the center crop')

args = parser.parse_args()


root_input_path = args.input_root
make_dir(args.output_root)
root_output_path = os.path.join(args.output_root, 'processed')
make_dir(root_output_path)
seq_output_path = os.path.join(root_output_path, '%d')

h_out = args.height
w_out = args.width

i = 1

main_seqS = os.listdir(root_input_path)
main_seqS.sort()
faceS = ['B', 'F', 'L', 'R']
left_rightS = ['Left', 'Right']

_depth_dir = os.path.join(root_input_path, '%s', 'Depth', 'Stereo_'+'%s', 'Omni_'+'%s')
_rgb_dir = os.path.join(root_input_path, '%s', 'RGB', 'Stereo_'+'%s', 'Omni_'+'%s')
_seg_dir = os.path.join(root_input_path, '%s', 'GT', 'LABELS' , 'Stereo_'+'%s', 'Omni_'+'%s')

print('starting...')

for seq in main_seqS:
    for left_right in left_rightS:
        for face in faceS:

            input_depth_dir = _depth_dir % (seq, left_right, face)
            input_rgb_dir = _rgb_dir % (seq, left_right, face)
            input_seg_dir = _seg_dir % (seq, left_right, face)

            if (os.path.isdir(input_seg_dir) and os.path.isdir(input_depth_dir) and os.path.isdir(input_rgb_dir)):

                num_rgb_files = len([name for name in os.listdir(input_rgb_dir) if os.path.isfile(os.path.join(input_rgb_dir, name))])
                num_depth_files = len([name for name in os.listdir(input_depth_dir) if os.path.isfile(os.path.join(input_depth_dir, name))])
                num_seg_files = len([name for name in os.listdir(input_seg_dir) if os.path.isfile(os.path.join(input_seg_dir, name))])

                if(num_rgb_files == num_depth_files and num_depth_files == num_seg_files):

                    make_dir(seq_output_path % i)

                    output_rgb_dir = os.path.join(seq_output_path % i, 'RGB')
                    output_depth_dir = os.path.join(seq_output_path % i, 'Depth')
                    output_seg_dir = os.path.join(seq_output_path % i, 'GT')

                    make_dir(output_rgb_dir)
                    make_dir(output_depth_dir)
                    make_dir(output_seg_dir)

                    print('processing %d images from %s and writing them in %s.' % (num_rgb_files, input_rgb_dir, output_rgb_dir))

                    for frm_num in range(num_rgb_files):

                        input_rgb_path = os.path.join(input_rgb_dir, '%06d.png' % (frm_num))
                        input_depth_path = os.path.join(input_depth_dir, '%06d.png' % (frm_num))
                        input_gt_path = os.path.join(input_seg_dir, '%06d.png' % (frm_num))

                        output_rgb_path = os.path.join(output_rgb_dir, '%06d.png' % (frm_num))
                        output_depth_path = os.path.join(output_depth_dir, '%06d.png' % (frm_num))
                        output_gt_path = os.path.join(output_seg_dir, '%06d.png' % (frm_num))

                        p_depth = Process(target=process_depth, args=(input_depth_path, output_depth_path,))
                        p_rgb = Process(target=process_rgb, args=(input_rgb_path, output_rgb_path,))
                        p_gt = Process(target=process_gt, args=(input_gt_path, output_gt_path,))

                        p_depth.start()
                        p_rgb.start()
                        p_gt.start()

                        p_depth.join()
                        p_rgb.join()
                        p_gt.join()

                    print(' ...  Done!')
                i+=1
print('All finished. %s sequnces have been processed' % i)
