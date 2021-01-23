# ------------------------------------------------------------------------
# Modified from https://github.com/Hzzone/pytorch-openpose
# ------------------------------------------------------------------------
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter


from .parameters import POSE_BODY_PART_PAIRS, POSE_MAP_INDEX


class BodyPostprocessor(object):

    def __init__(self, pose_points=25, thre1=0.1, thre2=0.05):
        assert pose_points in [15, 18, 25], "value error"
        self.pose_points = pose_points
        self.limb_list = POSE_BODY_PART_PAIRS[pose_points]
        self.map_idx = POSE_MAP_INDEX[pose_points]
        self.thre1 = thre1 # threshold to filter peaks
        self.thre2 = thre2


    def __call__(self, predict_heatmap):
        """
            predict_heatmap: np.ndarray, [1, c, h, w],
            c = 26 + 52 or 19 + 38 or 16 + 28
            predict_heatmap[:self.pose_points]     -> points confidence heatmap
            predict_heatmap[self.pose_points + 1:] -> PAF
        """
        heatmap = predict_heatmap[:self.pose_points + 1, :, :]
        paf = predict_heatmap[self.pose_points + 1:, :, :]
        all_peaks = self._find_peaks(heatmap)
        connection_all, special_k = self._sublayer_match(all_peaks, paf)
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        candidate, subset = self._build_pose_keypoints(candidate, connection_all, special_k)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset
    

    def _find_peaks(self, heatmap):
        """
        Params: heatmap: [self.pose_points + 1, h, w]
        Return: all_peaks: list with length self.pose_points
            each sub_list are all peaks for according keypoints
        for keypoints1: [(x, y, s, id), (x, y, s, id), ...]
        for keypoints2: [(x, y, s, id), (x, y, s, id), ...]
        ...
        for keypointsn: [(x, y, s, id), (x, y, s, id), ...]
        """
        all_peaks = []
        peak_counter = 0

        for part in range(self.pose_points):
            map_ori = heatmap[part] # [h, w], for part-th keypoints
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, 
                 one_heatmap >= map_down, one_heatmap > self.thre1)
            )
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_idx = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_idx[i], ) for i in range(len(peak_idx))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        
        return all_peaks


    def _sublayer_match(self, all_peaks, paf):
        """
        Return: connection_all: list,[[(id1, id2, s, idx1, idx2), ...], ...]
        """
        _, h, w = paf.shape
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(self.map_idx)):
            score_mid = paf[self.map_idx[k]].transpose((1, 2, 0)) # [h, w, 2]
            candA = all_peaks[self.limb_list[k][0]]
            candB = all_peaks[self.limb_list[k][1]]
            nA = len(candA)
            nB = len(candB)
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])
                        
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * h / norm - 1, 0
                        )
                        criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
                
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        
        return connection_all, special_k


    def _build_pose_keypoints(self, candidate, connection_all, special_k):
        subset = -1 * np.ones((0, self.pose_points + 2))
        for k in range(len(self.map_idx)): # for each part pairs
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(self.limb_list[k])

                for i in range(len(connection_all[k])): # for all pairs in sublayers
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
                    
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) * (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0] == 0): # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    
                    # if find no partA in the subset, create a new subset
                    elif not found and k < self.pose_points - 1:
                        row = -1 * np.ones(self.pose_points + 2)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts accur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        return candidate, subset
