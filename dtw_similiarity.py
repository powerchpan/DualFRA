import joblib
import numpy as np
import pandas as pd
#from data_prep.keypoints_to_image import get_shoulder_features
from numpy import array, zeros, argmin, inf, equal, ndim
from sklearn.metrics.pairwise import manhattan_distances
#from data_prep.load_data_2d import load_3d_keypoints
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import heapq
import scipy.signal as signal




# target_first_dir = '/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/keypoints_3d_multiPersons/side_good/'
target_first_dir = '/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/keypoints_3d_multiPersons/front_good/'
raw_fig_dir = '/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/fig_raw/dtw/front/'
predict_dir = '/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/fig_predict/dtw/front/'
standard_path = '/home/user/ply/code/automatic_mulit_human/data/standard_data.npy'

# raw_fig_dir = '/home/user/ply/code/automatic_mulit_human/data/feature_sts/hospital/raw_multi/'
# predict_dir = '/home/user/ply/code/automatic_mulit_human/data/feature_sts/hospital/fig/'


def load_3d_keypoints(path_dir):
    each_frame_people_num = []
    #print(path_dir)
    people_data = np.load(path_dir, allow_pickle=True)
    people_num = len(people_data)
    i =0
    while i<people_num:
        each_frame_people_num.append(people_data[i])
        i+=1
    keypoints = np.array(each_frame_people_num)
    return people_num, keypoints


def get_shoulder_features(path):
    #print(path)
    features = np.load(path, allow_pickle=True)
    Lshoulder = features[:,2,:]
    Rshoulder = features[:, 5,:]
    Lshoulder_x = Lshoulder[:,0]
    Rshoulder_x = Rshoulder[:,0]
    Lshoulder_y = Lshoulder[:,1]
    Rshoulder_y = Rshoulder[:,1]
    neck = features[:,1,1]
    Lankle_y = features[:,14,1]
    Rankle_y = features[:,11,1]
    ankle = (Lankle_y+Rankle_y)/2
    height = list(abs(ankle-neck))
    max_height = max(height)
    min_height = min(height)
    shoulder_x = abs(Lshoulder_x-Rshoulder_x)
    shoulder_y = (Lshoulder_y+Rshoulder_y)/2
    shoulder_y_list = list(shoulder_y)
    min_shoulder = min(shoulder_y_list)
    j = 0
    shoulder_x_list = []
    shoulder_y_list = []
    while j<len(shoulder_x):
        shoulder_y[j] = (shoulder_y[j]-min_shoulder)/max_height
        shoulder_x_list.append(shoulder_x[j])
        shoulder_y_list.append(shoulder_y[j])
        j+=1
    shoulder_x = np.array(shoulder_x_list)
    shoulder_y = np.array(shoulder_y_list)
    return shoulder_x,shoulder_y



def load_shoudler_x_y(feature_dir):
    i = 1
    shoulder_x_list = []
    shoulder_y_list = []
    while i < 43:
        if ((i == 17) | (i == 19) | (i == 22)):
           i += 1
           continue
        # if((i==12)|(i==14)|(i==21)|(i==23)|(i==58)|(i==60)|(i==76)|(i==77)|(i==85)|(i==86)|(i==87)|(i==91)|(i==96)|(i==102)|(i==117)|(i==140)|(i==142)|(i==147)):
        #     i += 1
        #     continue
        end_feature_path = str(i) + '.npy'
        feature_path = feature_dir + end_feature_path
        shoulder_x, shoulder_y = get_shoulder_features(feature_path)
        shoulder_x_list.append(shoulder_x)
        shoulder_y_list.append(shoulder_y)
        i+=1
    return shoulder_x_list, shoulder_y_list


def load_3d_shoudler_x_y(standrad_path):
    # shoulder_x_list = []
    # shoulder_y_list = []
    # while i < 43:
        # if ((i == 17) | (i == 19) | (i == 22)):
        #    i += 1
        #    continue
        # # if((i==12)|(i==14)|(i==21)|(i==23)|(i==58)|(i==60)|(i==76)|(i==77)|(i==85)|(i==86)|(i==87)|(i==91)|(i==96)|(i==102)|(i==117)|(i==140)|(i==142)|(i==147)):
        # #     i += 1
        # #     continue
        # end_feature_path = str(i) + '.npy'
        # feature_path = feature_dir + end_feature_path
    # standard_keypoints = np.load(standrad_path)
    shoulder_x, shoulder_y = get_shoulder_features(standrad_path)
        # shoulder_x_list.append(shoulder_x)
        # shoulder_y_list.append(shoulder_y)
        # i+=1
    return shoulder_x, shoulder_y

#使用dtw算法计算标准起坐视频中人动作的信号和target视频信号中每个人动作信号的相似度
def dtw_similiarity(standard, target):
    # if(len(target)<40):
    #     similiarity = 1000000
    #     return similiarity
    standard = np.array(standard).reshape(-1, 1)
    target = np.array(target).reshape(-1, 1)
    r, c = len(standard), len(target)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    # 浅复制
    # print D1

    for i in range(r):
        for j in range(c):
            first = [standard[i]]
            second = [target[j]]
            first = np.array(first)
            second = np.array(second)
            first = np.nan_to_num(first)
            second = np.nan_to_num(second)

            D1[i, j] = manhattan_distances(first, second)
            # D1[i, j] = manhattan_distances(standard[i], target[j])
    # 生成原始距离矩阵
    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
    # 代码核心，动态计算最短距离

    i, j = array(D0.shape) - 2
    # 最短路径
    # print i,j
    p, q = [i], [j]
    while (i > 0 or j > 0):
        tb = argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return D1[-1,-1]

def get_similiarity_test(standard_dir):
    shoulder_x_list, shoulder_y_list = load_3d_shoudler_x_y(standard_dir)
    standard_x = shoulder_x_list[1]
    standard_y = shoulder_y_list[1]
    # standard_y = standard_y[:310]
    target_x = shoulder_x_list[2]
    target_y = shoulder_y_list[2]
    similirity_x = dtw_similiarity(standard_x, target_x)
    similirity_y = dtw_similiarity(standard_y, target_y)
    return similirity_x, similirity_y

def get_target_shoulder_features(single_keypoints):
    Lshoulder = single_keypoints[:, 2, :]
    Rshoulder = single_keypoints[:, 5, :]
    Lshoulder_x = Lshoulder[:, 0]
    Rshoulder_x = Rshoulder[:, 0]
    Lshoulder_y = Lshoulder[:, 1]
    Rshoulder_y = Rshoulder[:, 1]
    neck = single_keypoints[:, 1, 1]
    Lankle_y = single_keypoints[:, 14, 1]
    Rankle_y = single_keypoints[:, 11, 1]
    ankle = (Lankle_y + Rankle_y) / 2
    height = list(abs(ankle - neck))
    max_height = max(height)
    min_height = min(height)
    shoulder_x = abs(Lshoulder_x - Rshoulder_x)
    shoulder_y = (Lshoulder_y + Rshoulder_y) / 2
    shoulder_y_list = list(shoulder_y)
    min_shoulder = min(shoulder_y_list)
    j = 0
    shoulder_x_list = []
    shoulder_y_list = []
    while j < len(shoulder_x):
        shoulder_y[j] = (shoulder_y[j] - min_shoulder) / max_height
        shoulder_x_list.append(shoulder_x[j])
        shoulder_y_list.append(shoulder_y[j])
        j += 16
    shoulder_x = np.array(shoulder_x_list)
    shoulder_y = np.array(shoulder_y_list)
    shoulder_y = signal.medfilt(shoulder_y, 3)
    return shoulder_x, shoulder_y

def get_similiarity(standard_dir, target_dir):
    shoulder_x_list, shoulder_y_list = load_3d_shoudler_x_y(standard_dir)
    # standard_x = shoulder_x_list[0]
    # standard_y = shoulder_y_list[0]
    standard_x = shoulder_x_list
    standard_y = shoulder_y_list
    #print(standard_y.shape)
    # standard_y = standard_y[:50]
    standard_y = signal.medfilt(standard_y, 3)
    #print(target_dir)
    people_num, target_keypoints = load_3d_keypoints(target_dir)
    similirity_x = []
    similirity_y = []

    #17.mp4后生成的openpose数据需要多加的步骤
    # target_keypoints = target_keypoints[1:]
    # target_arr = np.zeros((len(target_keypoints), 25, 3))
    # i = 0
    # print(target_dir)
    # for element in target_keypoints:
    #     target_arr[i] = element
    #     i+=1
    #17.mp4后的数据处理到此结束
    #print('people_num:'+str(people_num))
    frame_num_list = []
    if people_num>1:
        for people in target_keypoints:
            people_arr = np.zeros((len(people), 25, 3))
            if(len(people)<300):
                similirity_x.append(1000000)
                similirity_y.append(1000000)
                # target_keypoints.remove(people)
                # continue
            else:
                i = 0
                for element in people:
                    people_arr[i] = element
                    i+=1
                people = people_arr
                # people = np.array(people)
                frame_num_list.append(len(people))
                lshoulder_keypoints_y = people[:, 2, 1]
                rshoulder_keypoints_y = people[:, 5, 1]
                shoulder_y = (lshoulder_keypoints_y+rshoulder_keypoints_y)/2
                shoulder_y = signal.medfilt(shoulder_y, 5)
                frame_num = len(shoulder_y)
                hipy = []
                i = 0
                x = []
                while i < frame_num:
                    x.append(i)
                    hipy.append(shoulder_y[i])
                    i += 1
                hip_y = np.array(hipy)
                x = np.array(x)
                hip_y = signal.medfilt(hip_y, 7)
                #plt.plot(x, hip_y)
                #plt.savefig(raw_fig_path)
                people = torch.from_numpy(people)
                F.normalize(people)
                # target_Lshoulder_x = people[:, 2, 0]
                # target_Rshoulder_x = people[:,5, 0]
                # target_Lshoulder_y = people[:, 2, 1]
                # target_Rshoulder_y = people[:, 5, 1]
                # target_x = abs(target_Lshoulder_x-target_Rshoulder_x)
                # target_y = (target_Lshoulder_y+target_Rshoulder_y)/2
                target_x, target_y = get_target_shoulder_features(people)
                max_y = max(target_y)
                min_y = min(target_y)
                #print(max_y)
                #print(min_y)
                #print(abs(max_y-min_y))
                if(abs(max_y-min_y)<0.03):
                    similirity_people_y = 1000
                    similirity_y.append(similirity_people_y)
                    continue
                similirity_people_x = dtw_similiarity(standard_x, target_x)
                similirity_people_y = dtw_similiarity(standard_y, target_y)
                similirity_x.append(similirity_people_x)
                similirity_y.append(similirity_people_y)
        #plt.clf()
    else:
        print('This is a single_human video.')
        similirity_x.append(0)
        similirity_y.append(0)
    return similirity_x, similirity_y

def choose_keypoints(standard_dir, target_dir):
    similiarity_x, similiarity_y = get_similiarity(standard_dir, target_dir)
    # similarity = (similiarity_x + similiarity_y)/2
    similarity = similiarity_y
    people_num, target_keypoints =load_3d_keypoints(target_dir)
    similarity_list = similarity
    #print(similarity_list)
    min_index = similarity_list.index(min(similarity_list))
    max_index = similarity_list.index(max(similarity_list))
    keypoints_min = target_keypoints[min_index]
    #print('min_index:'+str(min_index))

    #print('max_index:'+str(max_index))
    keypoints_max = target_keypoints[max_index]
    return min_index, keypoints_min, keypoints_max


# i = 159
# while i<163:
#     # if((i==12)|(i==14)|(i==21)|(i==23)|(i==58)|(i==60)|(i==76)|(i==77)|(i==85)|(i==86)|(i==87)|(i==91)|(i==96)|(i==102)|(i==117)|(i==140)|(i==142)|(i==147)):
#     #     i += 1
#     #     continue
#     if ((i == 23) | (i == 60) | (i == 74) | (i == 77) | (i == 85) | (i == 87) | (i == 102) | (i == 114) | (i == 115) | (
#             i == 117) | (i == 118) | (i == 147)):
#         i += 1
#         continue
#     print('i:'+str(i))
#     target_end_path = ''+str(i)+'.npy'
#     raw_fig_end_path = ''+str(i)+'.jpg'
#     target_path = target_first_dir+target_end_path
#     raw_fig_path = raw_fig_dir+raw_fig_end_path
#     similiarist_path = predict_dir+str(i)+'.jpg'
#     not_similiarist_path = predict_dir+str(i)+'_max.jpg'
#     min_index, keypoints, keypoints_max = choose_keypoints(
#         standard_path, target_path)
#     # min_index, keypoints, keypoints_max = choose_keypoints(
#     #     '/home/user/ply/code/automatic_mulit_human/data/feature_sts/', target_path)
#     keypoints_arr = np.zeros((len(keypoints), 25, 3))
#     keypoints_max_arr = np.zeros((len(keypoints_max), 25, 3))
#     j = 0
#     for element in keypoints:
#         keypoints_arr[j] = element
#         j+=1
#     j = 0
#     for element in keypoints_max:
#         keypoints_max_arr[j] = element
#         j+=1
#     keypoints = keypoints_arr
#     keypoints_max = keypoints_max_arr
#     lshoulder_keypoints_y = keypoints[:, 2, 1]
#     rshoulder_keypoints_y = keypoints[:, 5, 1]
#     shoulder_y = (lshoulder_keypoints_y + rshoulder_keypoints_y) / 2
#     frame_num = len(shoulder_y)
#     heady = []
#     j = 0
#     x = []
#     while j < frame_num:
#         x.append(j)
#         heady.append(shoulder_y[j])
#         j += 3
#     head_y = np.array(heady)
#     x = np.array(x)
#     head_y = signal.medfilt(head_y, 7)
#     plt.plot(x, head_y)
#     plt.savefig(similiarist_path)
#     plt.clf()
#     #
#     lshoulder_keypoints_y = keypoints_max[:, 2, 1]
#     rshoulder_keypoints_y = keypoints_max[:, 5, 1]
#     shoulder_y = (lshoulder_keypoints_y + rshoulder_keypoints_y) / 2
#     frame_num = len(shoulder_y)
#     heady = []
#     j = 0
#     x = []
#     while j < frame_num:
#         x.append(j)
#         heady.append(shoulder_y[j])
#         j += 3
#     head_y = np.array(heady)
#     x = np.array(x)
#     head_y = signal.medfilt(head_y, 7)
#     plt.plot(x, head_y)
#     plt.savefig(not_similiarist_path)
#     plt.clf()
#     i+=1
