# -*- coding: utf-8 -*-
"""
INPUT ARGUMENTS:

dict: data dict
    Structure: {cid: {iid: [[uid, behavior, geohash, time], ...], ...}, ...}
                (int) (int)  (int) (int)    (string) (int)
    Attention: this geohash is the user's geohash, which means the user's location

beginTime: start of the time period(int)
    

endTime: end of the time period(int)
    
"""

import numpy as np
import math

def new_divide(a, b):
    re = -1
    try:
        re = float(a) / float(b)
    except:
        if a == 0:
            re = 0
    return re

def wrapper(data, P_item_id_unique, time_start, time_end):
    feat_item = {}
    feat_cat = {}
    for cid in data:
        temp_feat_item = {}
        for iid in data[cid]:
            temp_feat = extract_item_independent(data[cid][iid], time_start, time_end)
            temp_feat_item[iid] = temp_feat
        temp_c_feat = extract_cat_independent(data[cid], temp_feat_item, time_start, time_end)
        feat_cat[cid] = temp_c_feat
        feat_item.update({k: temp_feat_item[k] for k in temp_feat_item if k in P_item_id_unique})
    return feat_item, feat_cat

def extract_cat_independent(data, feat_item, time_start, time_end):
    """
    INPUT:
    data: {iid: [[uid, b, geohash, time], ...], ...}

    number of items
    mean of feat_item
    4/3
    4/2
    4/1

    update feat_item:
    feat_item normalized rank by behavior 4
    feat_item normalized rank by 4/1
    tf
    #idf
    #idf^2
    """
    feat = []
    feat.append(len(data))
    temp_data = np.array(feat_item.values(), dtype = np.float)
    useful = [0,1,2,3,4,5,6,7,16,17,18,19,23]
    temp_data = temp_data[:, useful]
    #temp_data = [sum(i) for i in zip(*feat_item.values())]
    #feat += np.sum(temp_data, axis=0).tolist()
    feat += np.mean(temp_data, axis=0).tolist()
    #feat += np.median(temp_data, axis=0).tolist()
    #feat += np.max(temp_data, axis=0).tolist()
    #feat += np.min(temp_data, axis=0).tolist()

    #feat += temp_data / float(len(data))
    temp_data_2 = np.sum(temp_data, axis=0)
    feat += [new_divide(temp_data_2[4], temp_data_2[1]), new_divide(temp_data_2[4], temp_data_2[2]), new_divide(temp_data_2[4], temp_data_2[3])]

    b_4 = temp_data[:, 4]
    b_4.sort()

    b_41 = temp_data[:, 7]
    b_41.sort()

    for k in feat_item:
        if len(feat_item) == 1:
            feat_item[k] += [1, 1]
        else:
            ind1 = np.where(b_4 == feat_item[k][4])[0][0]
            ind2 = np.where(b_41 == feat_item[k][7])[0][0]
            feat_item[k] += [ind1*1/float(len(feat_item) - 1), ind2*1/float(len(feat_item) - 1)]
        # if temp_data_2[4] == 0:
        #     feat_item[k] += [0, 0, 0]
        # elif feat_item[k][4] == 0:
        #     feat_item[k] += [0, 0, 0]
        # else:
        #     if feat_item[k][4] != 1:
        #         feat_item[k] += [float(feat_item[k][4]) / float(temp_data_2[4]), math.log(temp_data_2[4])/abs(math.log(feat_item[k][4])), (math.log(temp_data_2[4])/abs(math.log(feat_item[k][4])))**2]
        #     else:
        #         feat_item[k] += [float(feat_item[k][4]) / float(temp_data_2[4]), 0, 0]
        if temp_data_2[4] == 0:
            feat_item[k] += [0]
        elif feat_item[k][4] == 0:
            feat_item[k] += [0]
        else:
            feat_item[k] += [float(feat_item[k][4]) / float(temp_data_2[4])]

    return feat

def extract_item_independent(data, time_start, time_end):
    """
    input:
    [[uid, b, geohash, time], ...]

    number of user
    total number of behavior 1/2/3/4
    4/3
    4/2
    4/1
    mean number of behavior 1/2/3/4 for each user
    #median number of behavior 1/2/3/4 for each user
    max number of behavior 1/2/3/4 for each user
    #min number of behavior 1/2/3/4 for each user
    number of user who has 1/2/3/4
    (number of user who has 4) / (number of user who has 1/2/3)
    total spent time
    average spent time
    mean of behavior time to time_end
    median of behavior time to time_end
    mean of behavior 1/2/3/4 time to time_end
    median of behavior 1/2/3/4 time to time_end

    """
    feat = []
    u_dict = {}
    u_time = {}
    u_b_time = {1:[], 2:[], 3:[], 4:[]}
    for entry in data:
        if time_start <= entry[3] and entry[3] < time_end: 
            if entry[0] in u_dict:
                u_dict[entry[0]][entry[1] - 1] += 1
                u_time[entry[0]].append(entry[3])
            else:
                u_dict[entry[0]] = [0.0, 0.0, 0.0, 0.0]
                u_dict[entry[0]][entry[1] - 1] += 1
                u_time[entry[0]] = [entry[3]]
            u_b_time[entry[1]].append(entry[3])
    
    feat.append(len(u_dict))

    bs = np.array(u_dict.values(), dtype=np.float)

    temp = np.sum(bs, axis = 0).tolist()

    feat += temp
    feat += [new_divide(temp[3], temp[0]), new_divide(temp[3], temp[1]), new_divide(temp[3], temp[2])]
    feat += np.mean(bs, axis = 0).tolist()
    #feat += np.median(bs, axis = 0).tolist()
    feat += np.max(bs, axis = 0).tolist()
    #feat += np.min(bs, axis = 0).tolist()

    bs_2 = np.array(u_dict.values(), dtype=np.float)
    bs_2[bs_2 > 1] = 1

    temp = np.sum(bs_2, axis = 0).tolist()
    feat += temp
    feat += [new_divide(temp[3], temp[0]), new_divide(temp[3], temp[1]), new_divide(temp[3], temp[2])]

    times = [max(t) - min(t) for t in u_time.values()]
    feat += [sum(times), new_divide(sum(times), len(times))]

    all_time = np.array([item for sublist in u_time.values() for item in sublist], dtype=np.float)
    feat += [time_end - np.mean(all_time), time_end - np.median(all_time)]

    for bb in u_b_time:
        if u_b_time[bb] == []:
            u_b_time[bb] = [time_start]
        temp = np.array(u_b_time[bb], dtype=np.float)
        feat += [time_end - np.mean(temp), time_end - np.median(temp)]

    return feat

