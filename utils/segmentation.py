# This file contains util stuff needed for segmentation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight, ignore_index=0)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)

#-----------------------------------------
def colorize_segmentaions(fake, real, test=None):

    fake = np.squeeze(fake.numpy()[0, :, :, :])
    real = real.cpu().numpy()[0, :, :, :]

    ind = np.argmax(fake, axis=0)

    if test is None:
        ind[np.squeeze(real) == 0] = 0
    else:
        real = real[0, :, :]

    r = ind.copy()
    g = ind.copy()
    b = ind.copy()

    r_gt = real.copy()
    g_gt = real.copy()
    b_gt = real.copy()

    Void = [0, 0, 0]
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Road = [128, 64, 128]
    Sidewalk = [0, 0, 192]
    Fence = [64, 64, 128]
    Vegetation = [128, 128, 0]
    Pole = [192, 192, 128]
    Car = [64, 0, 128]
    TrafficSign = [192, 128, 128]
    Pedestrian = [64, 64, 0]
    Bicycle = [0, 128, 192]
    LaneMarking = [0, 172, 0]
    TrafficLight = [0, 128, 128]

    label_colours = np.array([Void, Sky, Building, Road, Sidewalk, Fence, Vegetation, Pole, Car, TrafficSign, Pedestrian, Bicycle, LaneMarking, TrafficLight])

    for l in range(1, len(label_colours)-1):
        r[ind == l] = label_colours[l, 0]
        g[ind == l] = label_colours[l, 1]
        b[ind == l] = label_colours[l, 2]

        r_gt[real == l] = label_colours[l, 0]
        g_gt[real == l] = label_colours[l, 1]
        b_gt[real == l] = label_colours[l, 2]

    rgb = np.zeros((3, ind.shape[0], ind.shape[1]))

    rgb[0, :, :] = r/255.0
    rgb[1, :, :] = g/255.0
    rgb[2, :, :] = b/255.0

    rgb_gt = np.zeros((3, ind.shape[0], ind.shape[1]))

    rgb_gt[0, :, :] = r_gt/255.0
    rgb_gt[1, :, :] = g_gt/255.0
    rgb_gt[2, :, :] = b_gt/255.0

    return torch.from_numpy(rgb), torch.from_numpy(rgb_gt)
