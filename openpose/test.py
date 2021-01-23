import sys
sys.path.append('/home/aistudio/openpose')
import numpy as np

import paddle.fluid as F
import paddle.fluid.dygraph as dg
# from model import BodyPose, HandPose
# from models.pose_body_25 import BodyPose
from models.pose_body_18 import BodyPose
from models.pose_hand_21 import HandPose
from models.pose_face_70 import FacePose


if __name__ == '__main__':
    # body_model = BodyPose()
    # for k, v in body_model.state_dict().items():
    #     print(k + ": " + str(v.shape))
    # hand_model = HandPose()
    # for k, v in hand_model.state_dict().items():
    #     print(k + ": " + str(v.shape))
    face_model = FacePose()
    for k, v in face_model.state_dict().items():
        print(k + ": " + str(v.shape))

    # hand_model = HandPose()
    # for k, v in hand_model.state_dict().items():
    #     print(k + ": " + str(v.shape))
    # with dg.guard():
    #     body_model = BodyPose()
    #     state_dict, _ = F.load_dygraph('/home/aistudio/openpose/pretrained_models/body_pose_model.pdparams')
    #     body_model.load_dict(state_dict)

    #     fake_data = dg.to_variable(np.zeros((1, 3, 512, 512)).astype("float32"))
    #     out1, out2 = body_model(fake_data)
    #     print(out1.shape) # [1, 38, 64, 64]
    #     print(out2.shape) # [1, 19, 64, 64]

    #     hand_model = HandPose()
    #     state_dict, _ = F.load_dygraph('/home/aistudio/openpose/pretrained_models/hand_pose_model.pdparams')
    #     hand_model.load_dict(state_dict)

    #     fake_data = dg.to_variable(np.zeros((1, 3, 512, 512)).astype("float32"))
    #     out = hand_model(fake_data)
    #     print(out.shape) # [1, 22, 64, 64]
