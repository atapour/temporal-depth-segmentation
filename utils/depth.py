# This file contains util stuff needed for depth image processing
import torch.nn.functional as F
import torch

# ------------------------------------------
# this fuction is based on the code in https://github.com/lyndonzheng/Synthetic2Realistic/blob/master/util/task.py
def scale_pyramid(img, num_scales):

    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = F.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=True)
        scaled_imgs.append(scaled_img)

    return scaled_imgs

# ------------------------------------------
def gradient_x(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx

#------------------------------------------
def gradient_y(img):
    gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gy

#------------------------------------------
# this fuction calculates the gradient loss
def get_smooth_weight(depths, images, num_scales):

    depth_gradient_x = [gradient_x(d) for d in depths]
    depth_gradient_y = [gradient_y(d) for d in depths]

    image_gradient_x = [gradient_x(img) for img in images]
    image_gradient_y = [gradient_y(img) for img in images]

    weight_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradient_x]
    weight_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradient_y]

    smoothness_x = [depth_gradient_x[i] * weight_x[i] for i in range(num_scales)]
    smoothness_y = [depth_gradient_y[i] * weight_y[i] for i in range(num_scales)]

    loss_x = [torch.mean(torch.abs(smoothness_x[i])) / 2**i for i in range(num_scales)]
    loss_y = [torch.mean(torch.abs(smoothness_y[i])) / 2**i for i in range(num_scales)]

    return sum(loss_x + loss_y)
