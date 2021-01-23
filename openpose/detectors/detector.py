import numpy as np 

from openpose.util import faceDetect, handDetect
from openpose.detectors.body import Body
from openpose.detectors.face import Face
from openpose.detectors.hand import Hand


class PoseDetector(object):

    def __init__(self, pose_points=25, detect_face=True, detect_hand=True):

        self.pose_points = pose_points
        self.detect_face = detect_face
        self.detect_hand = detect_hand

        self.body_detector = Body(pose_points)

        if detect_face:
            self.face_detector = Face()
        if detect_hand:
            self.hand_detector = Hand()
    

    def __call__(self, oriImg):
        candidate, subset = self.body_detector(oriImg)
        all_faces = []
        all_hands = []
        if self.detect_face:
            faces = faceDetect(candidate, subset, oriImg, pose_points=self.pose_points)
            for face in faces:
                if face == []:
                    all_faces.append([])
                    continue

                x, y, w = face
                face_keypoints = self.face_detector(oriImg[y:y+w, x:x+w, :])
                face_keypoints[:, 0] = np.where(face_keypoints[:, 0] == 0, 
                                                face_keypoints[:, 0],
                                                face_keypoints[:, 0] + x)
                face_keypoints[:, 1] = np.where(face_keypoints[:, 1] == 0,
                                                face_keypoints[:, 1],
                                                face_keypoints[:, 1] + y)
                all_faces.append(face_keypoints)
        
        if self.detect_hand:
            hands = handDetect(candidate, subset, oriImg,)
            for hand_pair in hands:
                hands_per_person = []
                for hand in hand_pair:
                    if hand == []:
                        hands_per_person.append([])
                        continue

                    x, y, w, _ = hand
                    hand_keypoints = self.hand_detector(oriImg[y:y+w, x:x+w, :])
                    hand_keypoints[:, 0] = np.where(hand_keypoints[:, 0] == 0,
                                                    hand_keypoints[:, 0],
                                                    hand_keypoints[:, 0] + x)
                    hand_keypoints[:, 1] = np.where(hand_keypoints[:, 1] == 0,
                                                    hand_keypoints[:, 1],
                                                    hand_keypoints[:, 1] + y)
                    hands_per_person.append(hand_keypoints)
                all_hands.append(hands_per_person)
        
        results = []
        for i, person in enumerate(subset):
            body_keypoints = np.array(candidate)[np.array(person[:-2]).astype(np.int)][:, :2]
            body_keypoints[np.array(person[:-2]).astype(np.int) == -1] *= -1

            if all_faces == []:
                face_keypoints = []
            else:
                face_keypoints = all_faces[i]
            
            if all_hands == []:
                hand_keypoints = []
            else:
                hand_keypoints = all_hands[i]
            
            results.append(
                {'body': body_keypoints, 'face': face_keypoints, 'hand': hand_keypoints}
            )
        
        return results
