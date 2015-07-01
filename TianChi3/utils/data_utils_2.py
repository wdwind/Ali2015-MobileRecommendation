"""
Data utils.
"""

import cPickle as pickle
import csv
from datetime import datetime
import os
import itertools

def load_P_item(filename):
    """Load items subset"""
    with open(filename, 'rb') as fp:
        itemdict = pickle.load(fp)
        P_num = itemdict['item_num']
        P_item_id = itemdict['item_id']
        P_item_geo = itemdict['item_geo']
        P_item_cat = itemdict['item_cat']
        return P_num, P_item_id, P_item_geo, P_item_cat

def load_var(filename):
    """Load single variable."""
    with open(filename, 'rb') as fp:
        single_var = pickle.load(fp)
        return single_var

def load_variables(filename, keys):
    """Load dictionary values by keys"""
    with open(filename, 'rb') as fp:
        datadict = pickle.load(fp)
        if len(keys) == 0:
            return datadict[keys[0]]
        return [datadict[key] for key in keys]

def get_data_dict_4(P_item_id_unique, P_cid_unique, time_start, time_end, label=False, type='uci', folder='newdata/date/'):
    if (label and type != 'uci') or (label and time_end == '2014-12-18'):
        raise ArithmeticError('Wrong input parameters')
    if type != 'uci' and type != 'ci':
        raise AttributeError('Wrong type')
    # include time_start and time_end
    dates = os.listdir(folder)
    dates.sort()
    start = dates.index(time_start)
    end = dates.index(time_end)
    #folder_name = 'data/date/' + dates[start]
    #raw_file_name = folder_name + '/raw.pkl'
    data = []
    for i in xrange(start, end + 1):
        print 'Processing date %s...' % dates[i]
        folder_name = folder + dates[i]
        raw_file_name = folder_name + '/raw.pkl'
        temp_data = load_var(raw_file_name)
        #for j in xrange(len(temp_data) - 1, -1, -1):
        #    if temp_data[j][2] not in P_item_id_unique:
        #        del temp_data[j]
        data += temp_data
    if type == 'uci':
        data_dict, _ = gen_uci(data)
    elif type == 'ci':
        data_dict, _ = gen_ci_2(data, P_cid_unique)
    if label:
        folder_name = folder + dates[end + 1]
        raw_file_name = folder_name + '/raw.pkl'
        temp_data = load_var(raw_file_name)
        for j in xrange(len(temp_data) - 1, -1, -1):
            if temp_data[j][2] not in P_item_id_unique:
                del temp_data[j]
        result = get_next_day_result(temp_data)
        label_dict = {tuple([entry[0], entry[2]]):1 for entry in result}
        return data_dict, label_dict, result
    return data_dict

def gen_uci(data):
    U_data_dict = {}
    num = 0
    for entry in data:
        num += 1
        uid = entry[0]
        cid = entry[1]
        iid = entry[2]
        b = entry[3]
        geo = entry[4]
        time = entry[5]
        if uid in U_data_dict:
            if cid in U_data_dict[uid]:
                if iid in U_data_dict[uid][cid]:
                    U_data_dict[uid][cid][iid].append([b, geo, time])
                else:
                    U_data_dict[uid][cid][iid] = [[b, geo, time]]
            else:
                U_data_dict[uid][cid] = {iid:[[b, geo, time]]}
        else:
            U_data_dict[uid] = {cid:{iid:[[b, geo, time]]}}
    return U_data_dict, num

def gen_ci_2(data, P_cid_unique):
    U_data_dict = {}
    num = 0
    for entry in data:
        num += 1
        uid = entry[0]
        cid = entry[1]
        iid = entry[2]
        b = entry[3]
        geo = entry[4]
        time = entry[5]
        if cid in U_data_dict and cid in P_cid_unique:
            if iid in U_data_dict[cid]:
                U_data_dict[cid][iid].append([uid, b, geo, time])
            else:
                U_data_dict[cid][iid] = [[uid, b, geo, time]]
        else:
            U_data_dict[cid] = {iid:[[uid, b, geo, time]]}
    return U_data_dict, num

def get_next_day_result(data, remove_duplicate=True):
    result = []
    for entry in data:
        uid = entry[0]
        cid = entry[1]
        iid = entry[2]
        b = entry[3]
        if b != 4:
            continue
        result.append([uid, cid, iid])
    if remove_duplicate:
        result.sort()
        result = list(result for result,_ in itertools.groupby(result))
    return result




