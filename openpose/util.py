# ------------------------------------------------------------------------
# Modified from https://github.com/Hzzone/pytorch-openpose
# ------------------------------------------------------------------------

import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from postprocess.parameters import (
    POSE_BODY_PAIRS_RENDER, 
    POSE_BODY_COLORS_RENDER,
    HAND_PAIRS_RENDER,
    HAND_COLORS_RENDER,
    FACE_PAIRS_RENDER
)


def padRightDownCorner(img, stride, padValue):
    h, w = img.shape[:2]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad



def draw_pose(canvas, person_keypoint_list, pose_points=25):
    for person in person_keypoint_list:
        body_keypoints = person['body']
        hands_keypoints = person['hand']
        face_keypoints = person['face']

        stickwidth = 4
        limbSeq = POSE_BODY_PAIRS_RENDER[pose_points]
        colors = POSE_BODY_COLORS_RENDER[pose_points]
        for i, points in enumerate(body_keypoints):
            x, y = points
            if x < 0 or y < 0:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i][::-1], thickness=-1)
        for i in range(len(limbSeq)):
            index_1, index_2 = limbSeq[i]

            cur_canvas = canvas.copy()
            y1, x1 = body_keypoints[index_1]
            y2, x2 = body_keypoints[index_2]
            if (x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0):
                continue
            mX = (x1 + x2) / 2
            mY = (y1 + y2) / 2
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limbSeq[i][1]][::-1])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        if hands_keypoints != []:
            draw_handpose(canvas, hands_keypoints)
        if face_keypoints != []:
            draw_facepose(canvas, [face_keypoints])
    
    return canvas


def draw_bodypose(canvas, candidate, subset, pose_points):
    stickwidth = 4
    limbSeq = POSE_BODY_PAIRS_RENDER[pose_points]
    colors = POSE_BODY_COLORS_RENDER[pose_points]
    for i in range(pose_points):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i][::-1], thickness=-1)
    
    for i in range(len(limbSeq)):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limbSeq[i][1]][::-1])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    stickwidth = 2
    limbSeq = HAND_PAIRS_RENDER
    colors = HAND_COLORS_RENDER

    for peaks in all_hand_peaks:
        for i, peak in enumerate(peaks):
            x, y = peak
            cv2.circle(canvas, (int(x), int(y)), 2, (colors[i][1], colors[i][2], colors[i][0]), thickness=-1)
    
    for peaks in all_hand_peaks:
        for ie, e in enumerate(limbSeq):
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                y1, x1 = peaks[e[0]]
                y2, x2 = peaks[e[1]]
                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2
                cur_canvas = canvas.copy()
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, (colors[e[1]][1], colors[e[1]][2], colors[e[1]][0]))
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
    return canvas


def draw_handpose_v2(canvas, all_hand_peaks, show_number=False):
    # all_head_peaks: [n, 21, 2]
    fig = Figure(figsize=plt.figaspect(canvas))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for peaks in all_hand_peaks:
        for ie, e in enumerate(HAND_PAIRS_RENDER):
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                ax.plot([x1, x2], [y1, y2], color=np.array(HAND_COLORS_RENDER[e[1]])[[1, 2, 0]] / 255.)
    
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas



def draw_facepose(canvas, all_face_peaks):
    stickwidth = 2
    limbSeq = FACE_PAIRS_RENDER

    for peaks in all_face_peaks:
        for i, peak in enumerate(peaks):
            x, y = peak
            cv2.circle(canvas, (int(x), int(y)), 2, (255, 255, 255), thickness=-1)
    
    for peaks in all_face_peaks:
        for ie, e in enumerate(limbSeq):
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                y1, x1 = peaks[e[0]]
                y2, x2 = peaks[e[1]]
                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2
                cur_canvas = canvas.copy()
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, (255, 255, 255))
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
    return canvas


def draw_facepose_v2(canvas, all_face_peaks, ):
    # all_head_peaks: [n, 21, 2]
    fig = Figure(figsize=plt.figaspect(canvas))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for peaks in all_face_peaks:
        for ie, e in enumerate(FACE_PAIRS_RENDER):
            if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                ax.plot([x1, x2], [y1, y2], color=(1, 1, 1))
    
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg,):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5 for pose-25 or pose-18
    # RShoulder: 2, RElbow: 3, RWrist: 4, LShoulder: 5, LElbow: 6, LWrist: 7 
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            detect_result.append([])
            continue
        
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])
        
        hands_pair = []
        for x1, y1, x2, y2, x3, y3, is_left in hands:
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            width = 1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)

            x -= width / 2
            y -= width / 2

            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)

            if width >= 20:
                hands_pair.append([int(x), int(y), int(width), is_left])
            else:
                hands_pair.append([])
        detect_result.append(hands_pair)

    return detect_result


# detect face according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/face/faceDetector.cpp
def faceDetect(candidate, subset, oriImg, pose_points=25):
    # nose: 0, neck: 1, Lear: 18, Rear: 17, Leye: 16, Reye: 15 for pose-25
    # nose: 0, neck: 1, Lear: 17, Rear: 16, Leye: 15, Reye: 14 for pose-18
    # head: 0, neck: 1 for pose-15

    image_height, image_width = oriImg.shape[0:2]

    if pose_points == 15:
        face = []
        for person in subset.astype(int):
            if np.sum(person[[0, 1]] == -1) == 0: # both head and neck are detected
                x1, y1 = candidate[person[0]][:2] # head position
                x2, y2 = candidate[person[1]][:2] # neck position
                distance_head_neck = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                width = 1.33 * distance_head_neck
                x, y = x1, y2
                face.append([x, y, width])
            else:
                face.append([])
    else: # for pose-18 or pose-25
        if pose_points == 18:
            nose_index = 0
            neck_index = 1
            Lear_index = 17
            Rear_index = 16
            Leye_index = 15
            Reye_index = 14
        else:
            assert pose_points == 25, "value error"
            nose_index = 0
            neck_index = 1
            Lear_index = 18
            Rear_index = 17
            Leye_index = 16
            Reye_index = 15
    
        face = []
        for person in subset.astype(int):
            x, y = 0, 0 # center point of bounding box
            width = 0 # width of bounding box
            counter = 0

            nose_detected = person[nose_index] != -1
            neck_detected = person[neck_index] != -1
            Lear_detected = person[Lear_index] != -1
            Rear_detected = person[Rear_index] != -1
            Leye_detected = person[Leye_index] != -1
            Reye_detected = person[Reye_index] != -1
            if neck_detected and nose_detected:
                x1, y1 = candidate[person[neck_index]][:2]
                x2, y2 = candidate[person[nose_index]][:2]
                distance_neck_nose = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                # if profile (i.e., only 1 eye and ear visible) --> avg(headNose, eye & ear position)
                if (Leye_detected == Lear_detected) and \
                   (Reye_detected == Lear_detected) and \
                   (Leye_detected != Reye_detected):
                    if Leye_detected:
                       x3, y3 = candidate[person[Leye_index]][:2]
                       x4, y4 = candidate[person[Lear_index]][:2]
                    else:
                       x3, y3 = candidate[person[Reye_index]][:2]
                       x4, y4 = candidate[person[Rear_index]][:2]
                    x += (x2 + x3 + x4) / 3
                    y += (y2 + y3 + y4) / 3
                    distance_nose_eye = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
                    distance_nose_ear = math.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2)
                    width += 0.85 * (distance_nose_eye + distance_nose_ear + distance_neck_nose)
                # else --> 2 * distance_neck_nose
                else:
                    x += (x1 + x2) / 2
                    y += (y1 + y2) / 2
                    width += 2 * distance_neck_nose
                counter += 1
            # 3 * distance_eyes
            if Leye_detected and Reye_detected:
                x3, y3 = candidate[person[Leye_index]][:2]
                x4, y4 = candidate[person[Reye_index]][:2]
                distance_eyes = math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
                x += (x3 + x4) / 2
                y += (y3 + y4) / 2
                width += 3 * distance_eyes
                counter += 1
            # 2 * distance_ears
            if Lear_detected and Rear_detected:
                x3, y3 = candidate[person[Lear_index]][:2]
                x4, y4 = candidate[person[Rear_index]][:2]
                distance_ears = math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
                x += (x3 + x4) / 2
                y += (y3 + y4) / 2
                width += 2 * distance_ears
                counter += 1
            
            if counter > 0:
                x /= counter
                y /= counter
                width /= counter
                face.append([x, y, width])
            else:
                face.append([])

    detect_result = []
    for item in face:
        if item == []:
            detect_result.append([])
            continue
        x, y, width = item
        x -= width / 2
        y -= width / 2

        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > image_width: width1 = image_width - x
        if y + width > image_height: width2 = image_height - y
        width = min(width1, width2)

        if width >= 20:
            detect_result.append([int(x), int(y), int(width)])
        else:
            detect_result.append([])

    return detect_result


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j





