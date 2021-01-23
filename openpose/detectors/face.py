# ------------------------------------------------------------------------
# Modified from https://github.com/Hzzone/pytorch-openpose
# ------------------------------------------------------------------------
import cv2
import numpy as np

import paddle.fluid.dygraph as dg

from openpose.models import build_face_model
from openpose.postprocess.face_postprocessor import FacePostprocessor
from openpose.util import padRightDownCorner


class Face(object):

    def __init__(self, ):
        self.face_model = build_face_model()
        self.face_model.eval()
        self.postprocessor = FacePostprocessor()
    

    def __call__(self, oriImg):
        h, w, _ = oriImg.shape

        scale_search = [0.5, 1.0, 1.5, 2.0]
        # scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        multiplier = [x * boxsize / h for x in scale_search]
        avg_output = np.zeros((71, h, w))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256. - 0.5
            im = np.ascontiguousarray(im)

            data = dg.to_variable(im)
            with dg.no_grad():
                output = self.face_model(data)[-1]
            heatmap = output.numpy()[0].transpose((1, 2, 0)) # [h, w, c]
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap.transpose((2, 0, 1)) # [c, h, w]
            
            avg_output += heatmap / len(multiplier)
        
        return self.postprocessor(avg_output)
