# This file contains util stuff needed for optical flow processing
import math
import torch
import torch.nn.functional as F

#-----------------------------------------
def EPE(input_flow, target_flow):
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()

#-----------------------------------------
def estimate_flow(tensorInputFirst, tensorInputSecond, moduleNetwork):

    tensorOutput = torch.FloatTensor()

    tensorInputFirst = tensorInputFirst.cuda()
    tensorInputSecond = tensorInputSecond.cuda()
    tensorOutput = tensorOutput.cuda()

    assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))
    assert(tensorInputFirst.size(3) == tensorInputSecond.size(3))

    tensorInputFirst = tensorInputFirst.expand(tensorInputFirst.size(0), 3, tensorInputFirst.size(2), tensorInputFirst.size(3))
    tensorInputSecond = tensorInputSecond.expand(tensorInputSecond.size(0), 3, tensorInputSecond.size(2), tensorInputSecond.size(3))

    intWidth = tensorInputFirst.size(3)
    intHeight = tensorInputFirst.size(2)

    tensorPreprocessedFirst = tensorInputFirst.view(tensorInputFirst.size(0), 3, intHeight, intWidth)
    tensorPreprocessedSecond = tensorInputSecond.view(tensorInputFirst.size(0), 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tensorFlow = F.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    tensorOutput.resize_(2, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])

    return tensorOutput
