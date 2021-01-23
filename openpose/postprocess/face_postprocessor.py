# ------------------------------------------------------------------------
# Modified from https://github.com/Hzzone/pytorch-openpose
# ------------------------------------------------------------------------
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label

from openpose import util


class FacePostprocessor(object):

    def __init__(self, ):
        pass
    

    def __call__(self, heatmap):
        """
        heatmap: ndarray, [71, h, w]
        """
        thre = 0.05
        all_peaks = []
        for part in range(70):
            map_ori = heatmap[part, :, :]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)

            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = util.npmax(map_ori)
            all_peaks.append([x, y])
        return np.array(all_peaks)
