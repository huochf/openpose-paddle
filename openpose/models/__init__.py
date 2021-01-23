import paddle.fluid.dygraph as dg

from .pose_body_25 import BodyPose as BodyPose25
from .pose_body_18 import BodyPose as BodyPose18
from .pose_body_15 import BodyPose as BodyPose15
from .pose_face_70 import FacePose
from .pose_hand_21 import HandPose

def build_body_model(body_points=25):
    if body_points == 25:
        body_model = BodyPose25()
        state_dict, _ = dg.load_dygraph('/home/aistudio/openpose/pretrained_models/pose_body_25_iter_584000.pdparams')
        body_model.load_dict(state_dict)
    elif body_points == 18:
        body_model = BodyPose18()
        state_dict, _ = dg.load_dygraph('/home/aistudio/openpose/pretrained_models/pose_body_18_iter_440000.pdparams')
        body_model.load_dict(state_dict)
    elif body_points == 15:
        body_model = BodyPose15()
        state_dict, _ = dg.load_dygraph('/home/aistudio/openpose/pretrained_models/pose_body_15_iter_160000.pdparams')
        body_model.load_dict(state_dict)
    else:
        raise ValueError()
    
    return body_model
        

def build_face_model():
    face_model = FacePose()
    state_dict, _ = dg.load_dygraph('/home/aistudio/openpose/pretrained_models/pose_face_70_iter_116000.pdparams')
    face_model.load_dict(state_dict)
    return face_model


def build_hand_model():
    hand_model = HandPose()
    state_dict, _ = dg.load_dygraph('/home/aistudio/openpose/pretrained_models/pose_hand_21_102000.pdparams')
    hand_model.load_dict(state_dict)
    return hand_model