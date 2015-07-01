"""
Feature Extraction
=====

Extract feature vector from the raw data.

"""

import numpy as np
from numpy_utils import find_1D
from datetime import datetime
import data_utils_2 as data_utils
import csv
import math

class Feature(object):
    
    def __init__(self, item_dict='../data/itemdict'):
        """
        Class initialization.

        Parameters
        ----------
        funs_list : functions list. This list should contains three child list.
                    The first child list contains the user level feature Extraction
                    functions. The second contains the cat level functions.
                    The third contains the item level functions.

        """
        self.item_geos = None
        self.cat_geos = None
        self.read_geos(item_dict)

    def read_geos(self, item_dict):
        """
        Read geohash for items and corresponding categories, and the results are 
        stored in two dictionaries self.item_geos and self.cat_geos.
        The stucture of the self.item_geos and self.cat_geos:
            The key of the dict is the iid (item id) or cid (category id), and 
            the value is a list of all geohash corresponding to the specific id.

        """
        if self.item_geos is None:
            self.item_geos = {}
            self.cat_geos = {}
            _, P_item_id, P_item_geo, P_item_cat = data_utils.load_P_item(item_dict)
            self.P_item_id = P_item_id
            P_item_id_unique = np.unique(P_item_id).tolist()
            P_item_id_unique = dict((el, 0) for el in P_item_id_unique)
            self.P_item_id_unique = P_item_id_unique
            self.P_item_cat = P_item_cat
            P_item_cat_unique = np.unique(P_item_cat).tolist()
            P_item_cat_unique = dict((el, 0) for el in P_item_cat_unique)
            self.P_item_cat_unique = P_item_cat_unique
            for i in xrange(P_item_geo.shape[0]):
                if P_item_geo[i] != '':
                    if P_item_id[i] in self.item_geos:
                        self.item_geos[P_item_id[i]].append(P_item_geo[i])
                    else:
                        self.item_geos[P_item_id[i]] = [P_item_geo[i]]
                    if P_item_cat[i] in self.cat_geos:
                        self.cat_geos[P_item_cat[i]].append(P_item_geo[i])
                    else:
                        self.cat_geos[P_item_cat[i]] = [P_item_geo[i]]

    def new_divide(self, a, b):
        re = -1
        try:
            re = float(a) / float(b)
        except:
            if a == 0:
                re = 0
        return re

    def geo_dist(self, geo1, geo2, normalize=False):
        """
        Calculate the distance between two geohashes.

        Parameters
        ----------
        geo1 : geohashes 1, string.
        geo2 : geohashes 2, string.
        The two input geohashes should have the same length, otherwise there 
        will be an error.

        Returns
        -------
        The number of the same prefix of two geohashes. The higher, the closer.

        Examples
        --------
        >>> a = feature_extraction.geo_dist('9adsf12', '9a3241s')
        2

        """
        l1 = len(geo1)
        l2 = len(geo2)
        if l1 != l2:
            #print geo1
            #print geo2
            raise AttributeError('Wrong geos. Geo1: %s, geo2: %s' % (geo1, geo2))
        same = 0
        for i in xrange(l1):
            if geo1[i] != geo2[i]:
                break
            else: same += 1
        if normalize:
            """
            1 5009.4km x 4992.6km
            2 1252.3km x 624.1km
            3 156.5km x 156km
            4 39.1km x 19.5km
            5 4.9km x 4.9km
            6 1.2km x 0.61km
            7 152.9m x 152.4m
            """
            if same == 0 or same == 1:
                pass
            elif same == 2:
                same *= 32
            elif same == 3:
                same *= 1024.5
            elif same == 4:
                same *= 32802
            elif same == 5:
                same *= 1041646.4
            elif same == 6:
                same *= 34166571.6
            elif same == 7:
                same *= 1073297286.6
        return same

    def extract_features_2_2(self, data, time_thresh=None):
        """
        USER level 2.

        Parameters
        ----------
        data : data dict. Structure: {cid: {iid: [[behavior_type, user_geohash, time]]}}
        time_thresh : the time of the day to be predicted to day `2014-11-18 00`.
                        e.g., if one needs to predict the sale data in day `2014-12-18 00`,
                        then this time_thresh shoule be set to 
                        feature_extraction.duration_hours('2014-12-18 00')

        Returns
        -------
        1-D #numpy# feature vector. Features:
            total number of different categories
            total number of different items
            total number of behaviours 1
            total number of behaviours 2
            total number of behaviours 3
            total number of behaviours 4
            average number of behaviours 1
            average number of behaviours 2
            average number of behaviours 3
            average number of behaviours 4
            4/1
            4/2
            4/3
            total number of cats with 2
            total number of cats with 3
            total number of cats with 4
            total number of items with 2  #---------------- need to be added
            total number of items with 3  #---------------- need to be added
            total number of items with 4  #---------------- need to be added
            number of cats with (4/3/2/1) / number of cats without (4/3/2/1)
            number of items with (4/3/2/1) / number of items without (4/3/2/1)
            #log of above

        dict {cid:[1-D feature vector], ...}
            #average number of behaviours 1 except cid
            #average number of behaviours 2 except cid
            #average number of behaviours 3 except cid
            number of behaviours 4 except cid
            average number of behaviours 4 except cid
            #4/1 except cid
            #4/2 except cid
            #4/3 except cid
            
        """
        #dim = 6
        #dim = 14

        #feature = np.zeros(dim, dtype=np.float)
        #feature = []

        #id = 0

        #feature[id:id+2] = self.extract_features_2(data, time_thresh)
        #id += 2
        #feature += self.extract_features_2(data, time_thresh)
        #feature.append(len(data))
        c_b = {cid:[0.0, 0.0, 0.0, 0.0] for cid in data if cid > 0 and cid in self.P_item_cat_unique}

        num_items = 0.0
        b_1 = 0.0
        b_2 = 0.0
        b_3 = 0.0
        b_4 = 0.0
        # b_1_e = 0.0
        # b_2_e = 0.0
        # b_3_e = 0.0
        # b_4_e = 0.0
        c_1 = 0.0
        c_2 = 0.0
        c_3 = 0.0
        c_4 = 0.0
        c_i1 = 0.0
        c_i2 = 0.0
        c_i3 = 0.0
        c_i4 = 0.0
        for cid in data:
            if cid > 0:
                temp_c_1 = 0
                temp_c_2 = 0
                temp_c_3 = 0
                temp_c_4 = 0
                for iid in data[cid]:
                    if iid > 0:
                        temp_c_i1 = 0
                        temp_c_i2 = 0
                        temp_c_i3 = 0
                        temp_c_i4 = 0
                        num_items += 1
                        for entry in data[cid][iid]:
                            if entry[0] == 1:
                                b_1 += 1
                                temp_c_1 = 1
                                temp_c_i1 = 1
                                # if cid != CID:
                                #     b_1_e += 1
                                if cid in self.P_item_cat_unique:
                                    c_b[cid][0] += 1
                            elif entry[0] == 2:
                                b_2 += 1
                                temp_c_2 = 1
                                temp_c_i2 = 1
                                # if cid != CID:
                                #     b_2_e += 1
                                if cid in self.P_item_cat_unique:
                                    c_b[cid][1] += 1
                            elif entry[0] == 3:
                                b_3 += 1
                                temp_c_3 = 1
                                temp_c_i3 = 1
                                # if cid != CID:
                                #     b_3_e += 1
                                if cid in self.P_item_cat_unique:
                                    c_b[cid][2] += 1
                            elif entry[0] == 4:
                                b_4 += 1
                                temp_c_4 = 1
                                temp_c_i4 = 1
                                # if cid != CID:
                                #     b_4_e += 1
                                if cid in self.P_item_cat_unique:
                                    c_b[cid][3] += 1
                        c_i1 += temp_c_i1
                        c_i2 += temp_c_i2
                        c_i3 += temp_c_i3
                        c_i4 += temp_c_i4
                c_1 += temp_c_1
                c_2 += temp_c_2
                c_3 += temp_c_3
                c_4 += temp_c_4
        # feature[id] = num_items
        # id += 1
        # feature[id] = b_1
        # id += 1
        # feature[id] = b_2
        # id += 1
        # feature[id] = b_3
        # id += 1
        # feature[id] = b_4
        # id += 1

        feature = [len(data), num_items, b_1, b_2, b_3, b_4, b_1/(len(data)), b_2/(len(data)), b_3/(len(data)), b_4/(len(data))]#, b_1_e/(len(data)-1), b_2_e/(len(data)-1), b_3_e/(len(data)-1), b_4_e/(len(data)-1)]
        feature += [self.new_divide(b_4, b_3), self.new_divide(b_4, b_2), self.new_divide(b_4, b_1)]
        #feature += [b_4_e/b_3_e, b_4_e/b_2_e, b_4_e/b_1_e]
        feature += [c_2, c_3, c_4]
        feature += [c_i2, c_i3, c_i4]
        feature += [self.new_divide(c_1, len(data) - c_1), self.new_divide(c_2, len(data) - c_2), self.new_divide(c_3, len(data) - c_3), self.new_divide(c_4, len(data) - c_4)]
        feature += [self.new_divide(c_i1, num_items - c_i1), self.new_divide(c_i2, num_items - c_i2), self.new_divide(c_i3, num_items - c_i3), self.new_divide(c_i4, num_items - c_i4)]

        # feature[id:] = np.log(feature[0:id] + 2)
        #logable = [True for i in feature]

        c_feat = {cid:0 for cid in c_b}
        for cid in c_b:
            c_feat[cid] = [c_b[cid][3], self.new_divide(c_b[cid][3], len(data)-1)]
            #c_feat[cid] = [self.new_divide(c_b[cid][0], len(data)-1), self.new_divide(c_b[cid][1], len(data)-1), self.new_divide(c_b[cid][2], len(data)-1), self.new_divide(c_b[cid][3], len(data)-1), self.new_divide(c_b[cid][3], c_b[cid][0]), self.new_divide(c_b[cid][3], c_b[cid][1]), self.new_divide(c_b[cid][3], c_b[cid][2])]

        return feature, c_feat

    def  extract_features_3(self, data, time_thresh=None, time_start=0):
        """
        CAT level.
        Get feature vector from one user's behaviours of one category.
        #The structure of 'data' is a dict {item_id: [behavior_type, user_geohash, time]}

        Parameters
        ----------
        data : data dict. Structure: {iid: [[behavior_type, user_geohash, time]]}
        time_thresh : the time of the day to be predicted to day `2014-11-18 00`.
                        e.g., if one needs to predict the sale data in day `2014-12-18 00`,
                        then this time_thresh shoule be set to 
                        feature_extraction.duration_hours('2014-12-18 00')

        Returns
        -------
        1-D numpy feature vector. This function extracts the following features:
            total number of different items
            total number of behaviours 1
            total number of behaviours 2
            total number of behaviours 3
            total number of behaviours 4
            4/3
            4/2
            4/1
            \ mean 4/i for all items
            \ median 4/i for all items
            #\ max 4/i for all items
            #\ min 4/i for all items
            count of items with 2
            count of items with 3
            count of items with 4
            number of items with (4/3/2/1) / number of items without (4/3/2/1)
            \ mean number of behaviours i for all item
            \ mean number of behaviours i for the items with i
            \ median number of behaviours i for the items with i
            \ max number of behaviours i for the items with i #------------------------- useful???
            #\ min number of behaviours i for the items with i
            \ mean time between last behaviour to pred day for all item
            \ max time between last behaviour to pred day for all item #--------------------- min???
            \ min time between last behaviour to pred day for all item
            \ time between last behaviour 1 to pred day
            \ time between last behaviour 2 to pred day
            \ time between last behaviour 3 to pred day
            \ time between last behaviour 4 to pred day
            \ mean overall duration
            \ median overall duration
            \ max overall duration
            #\ min overall duration
            #\ min time between last behaviour to pred day for all item
            #total number of items
            #log(t+1) for all above

            Total dimensions: 2 * (1 + 4 + 3 + 12 + 3)

        dict {iid: [1-D feature vector], ...}
            count of items with behaviours 2 except iid
            count of items with behaviours 3 except iid
            count of items with behaviours 4 except iid
            average number of behaviours 1 except iid
            average number of behaviours 2 except iid
            average number of behaviours 3 except iid
            average number of behaviours 4 except iid
            min time between last behaviour 1 to pred day except iid
            min time between last behaviour 2 to pred day except iid
            min time between last behaviour 3 to pred day except iid
            min time between last behaviour 4 to pred day except iid
            min time between last behaviour to pred day except iid
            #4/1 except cid
            #4/2 except cid
            #4/3 except cid

        """
        if time_thresh is None:
            time_thresh = self.TIME_THRESH
        raw_data = []
        iid_dict = {}
        for iid in data:
            if iid > 0:
                temp = [[iid] + entry for entry in data[iid]]
                raw_data += temp
                iid_dict[iid] = [0.0, 0.0, 0.0, 0.0]
                for entry in data[iid]:
                    if entry[0] == 1:
                        iid_dict[iid][0] += 1
                    elif entry[0] == 2:
                        iid_dict[iid][1] += 1
                    elif entry[0] == 3:
                        iid_dict[iid][2] += 1
                    elif entry[0] == 4:
                        iid_dict[iid][3] += 1
        data = raw_data

        #dim = 46

        #feature = np.zeros(dim, dtype=np.float)
        #feature = [0 for i in xrange(dim)]
        feature = []
        item_ids = np.array([data[i][0] for i in xrange(len(data))])
        behaviors = np.array([data[i][1] for i in xrange(len(data))])
        times = np.array([data[i][3] for i in xrange(len(data))])

        b_1 = find_1D(behaviors == 1)
        b_2 = find_1D(behaviors == 2)
        b_3 = find_1D(behaviors == 3)
        b_4 = find_1D(behaviors == 4)

        item_unique = np.unique(item_ids)

        #feature[0] = item_unique.shape[0]
        feature.append(item_unique.shape[0])

        # feature[1] = b_1.shape[0]
        # feature[2] = b_2.shape[0]
        # feature[3] = b_3.shape[0]
        # feature[4] = b_4.shape[0]
        feature += [b_1.shape[0], b_2.shape[0], b_3.shape[0], b_4.shape[0]]
        feature += [self.new_divide(b_4.shape[0], b_1.shape[0]), self.new_divide(b_4.shape[0], b_2.shape[0]), self.new_divide(b_4.shape[0], b_3.shape[0])]

        iid_ratio = np.array([[self.new_divide(entry[3], entry[0]), self.new_divide(entry[3], entry[1]), self.new_divide(entry[3], entry[2])] for entry in iid_dict.values() if entry[3] != 0])
        if iid_ratio.shape == (0L,):
            iid_ratio = np.array([[0.0, 0.0, 0.0]])
        feature += np.mean(iid_ratio, axis=0).tolist()
        feature += np.median(iid_ratio, axis=0).tolist()
        #feature += np.max(iid_ratio, axis=0).tolist()
        #feature += np.min(iid_ratio, axis=0).tolist()

        #feature += [float(b_1.shape[0])/feature[0], float(b_2.shape[0])/feature[0], float(b_3.shape[0])/feature[0], float(b_4.shape[0])/feature[0]
        #feature += [float(b_1_e.shape[0])/feature[0], float(b_2_e.shape[0])/feature[0], float(b_3_e.shape[0])/feature[0], float(b_4_e.shape[0])/feature[0]

        #feature += [np.unique(item_ids_except[b_2_e]).shape[0]]

        # feature[5] = feature[0] - np.unique(item_ids[b_2]).shape[0]
        # feature[6] = feature[0] - np.unique(item_ids[b_3]).shape[0]
        # feature[7] = feature[0] - np.unique(item_ids[b_4]).shape[0]

        feature += [np.unique(item_ids[b_2]).shape[0], np.unique(item_ids[b_3]).shape[0], np.unique(item_ids[b_4]).shape[0]]
        feature += [self.new_divide(np.unique(item_ids[b_1]).shape[0], item_unique.shape[0] - np.unique(item_ids[b_1]).shape[0]), self.new_divide(np.unique(item_ids[b_2]).shape[0], item_unique.shape[0] - np.unique(item_ids[b_2]).shape[0]), self.new_divide(np.unique(item_ids[b_3]).shape[0], item_unique.shape[0] - np.unique(item_ids[b_3]).shape[0]), self.new_divide(np.unique(item_ids[b_4]).shape[0], item_unique.shape[0] - np.unique(item_ids[b_4]).shape[0])]

        _, counts_1 = np.unique(item_ids[b_1], return_counts=True)
        _, counts_2 = np.unique(item_ids[b_2], return_counts=True)
        _, counts_3 = np.unique(item_ids[b_3], return_counts=True)
        _, counts_4 = np.unique(item_ids[b_4], return_counts=True)

        if counts_1.size == 0:
            counts_1 = np.zeros(1)
        if counts_2.size == 0:
            counts_2 = np.zeros(1)
        if counts_3.size == 0:
            counts_3 = np.zeros(1)
        if counts_4.size == 0:
            counts_4 = np.zeros(1)

        # feature[8] = np.sum(counts_1) / (feature[0] + 0.0)
        # feature[9] = np.sum(counts_2) / (feature[0] + 0.0)
        # feature[10] = np.sum(counts_3) / (feature[0] + 0.0)
        # feature[11] = np.sum(counts_4) / (feature[0] + 0.0)

        feature += [self.new_divide(np.sum(counts_1), feature[0]), self.new_divide(np.sum(counts_2), feature[0]), self.new_divide(np.sum(counts_3), feature[0]), self.new_divide(np.sum(counts_4), feature[0])]
        
        feature += [counts_1.mean(), counts_2.mean(), counts_3.mean(), counts_4.mean()]
        feature += [np.median(counts_1), np.median(counts_2), np.median(counts_3), np.median(counts_4)]
        feature += [np.max(counts_1), np.max(counts_2), np.max(counts_3), np.max(counts_4)]

        # feature[12] = counts_1.max()
        # feature[13] = counts_2.max()
        # feature[14] = counts_3.max()
        # feature[15] = counts_4.max()

        #feature += [counts_1.max(), counts_2.max(), counts_3.max(), counts_4.max()]

        # feature[16] = counts_1.min()
        # feature[17] = counts_2.min()
        # feature[18] = counts_3.min()
        # feature[19] = counts_4.min()

        #feature += [counts_1.min(), counts_2.min(), counts_3.min(), counts_4.min()]

        last_b = [time_thresh - max([times[i] for i in find_1D(item_ids == item)]) for item in item_unique]
        last_b_1 = time_thresh - max([time_start] + [times[i] for i in b_1])
        last_b_2 = time_thresh - max([time_start] + [times[i] for i in b_2])
        last_b_3 = time_thresh - max([time_start] + [times[i] for i in b_3])
        last_b_4 = time_thresh - max([time_start] + [times[i] for i in b_4])
        time_dict = {item: [time_thresh - max([time_start] + [times[i] for i in find_1D(item_ids == item) if i in b_1]), time_thresh - max([time_start] + [times[i] for i in find_1D(item_ids == item) if i in b_2]), time_thresh - max([time_start] + [times[i] for i in find_1D(item_ids == item) if i in b_3]), time_thresh - max([time_start] + [times[i] for i in find_1D(item_ids == item) if i in b_4])] for item in item_unique}
        time_dict_values = np.array(time_dict.values(), dtype=np.float)
        time_dict_keys = time_dict.keys()
        duration_all = np.array([max([times[i] for i in find_1D(item_ids == item)]) - min([times[i] for i in find_1D(item_ids == item)]) for item in item_unique])

        # feature[20] = sum(last_b) / len(last_b)
        # feature[21] = max(last_b)
        # feature[22] = min(last_b)

        #feature += [sum(last_b) / len(last_b), max(last_b), min(last_b)]
        feature += [self.new_divide(sum(last_b), len(last_b)), max(last_b), min(last_b)]
        feature += [last_b_1, last_b_2, last_b_3, last_b_4]
        feature += [np.mean(duration_all), np.median(duration_all), np.max(duration_all)]

        #feature[23] = item_ids.shape[0];

        #feature[23:] = np.log(feature[0:23] + 2)

        #global id
        #id += 1
        #if id%100000 == 0:
        #    print id

        i_feat = {iid:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for iid in iid_dict if iid in self.P_item_id_unique}
        counts = np.array([iid_dict[iid] for iid in iid_dict if iid in self.P_item_id_unique])
        counts_cp = np.copy(counts)
        counts_cp = np.sum(counts_cp, axis=0)
        counts[counts > 1] = 1
        counts = np.sum(counts, axis=0)
        for iid in i_feat:
            i_feat[iid][3] = self.new_divide(counts_cp[0] - iid_dict[iid][0], item_unique.shape[0] - 1)
            i_feat[iid][4] = self.new_divide(counts_cp[1] - iid_dict[iid][1], item_unique.shape[0] - 1)
            i_feat[iid][5] = self.new_divide(counts_cp[2] - iid_dict[iid][2], item_unique.shape[0] - 1)
            i_feat[iid][6] = self.new_divide(counts_cp[3] - iid_dict[iid][3], item_unique.shape[0] - 1)
            if iid_dict[iid][1] > 0:
                i_feat[iid][0] = counts[1] - 1
            else:
                i_feat[iid][0] = counts[1]
            if iid_dict[iid][2] > 0:
                i_feat[iid][1] = counts[2] - 1
            else:
                i_feat[iid][1] = counts[2]
            if iid_dict[iid][3] > 0:
                i_feat[iid][2] = counts[3] - 1
            else:
                i_feat[iid][2] = counts[3]
            temp_keys = [i for i in xrange(len(time_dict_keys))]
            del temp_keys[time_dict_keys.index(iid)]
            temp_values = time_dict_values[temp_keys, :]
            if temp_values.shape == (0L,) or temp_values.shape == (1L, 0L) or temp_values.shape[0] == 0:
                i_feat[iid] += [time_thresh - time_start, time_thresh - time_start, time_thresh - time_start, time_thresh - time_start, time_thresh - time_start]
            else:
                #print temp_values
                #print temp_values.shape
                temp_values = np.min(temp_values, axis=0)
                i_feat[iid] += temp_values.tolist()
                i_feat[iid].append(np.min(temp_values))

        return feature, i_feat

    def extract_features_5(self, data, iid, time_thresh=None, time_start=0):
        """
        ITEM level.
        Get feature vector from one user's behaviours of one item.
        The structure of 'data' is a list of [behavior_type, user_geohash, time]

        Parameters
        ----------
        data : data dict. Structure: [[behavior_type, user_geohash, time]]
        iid : item id.
        time_thresh : the time of the day to be predicted to day `2014-11-18 00`.
                        e.g., if one needs to predict the sale data in day `2014-12-18 00`,
                        then this time_thresh shoule be set to 
                        feature_extraction.duration_hours('2014-12-18 00')

        Returns
        -------
        1-D numpy feature vector. This function extracts the following features:
            max same prefix between user and item
            normalized (max same prefix between user and item)
            total number of behaviours 1
            total number of behaviours 2
            total number of behaviours 3
            total number of behaviours 4
            #4/3
            #4/2
            #4/1
            time between last behaviour to pred day
            time between last behaviour 1 to pred day
            time between last behaviour 2 to pred day
            time between last behaviour 3 to pred day
            time between last behaviour 4 to pred day
            time between first behaviour to pred day
            time between first behaviour 1 to pred day
            time between first behaviour 2 to pred day
            time between first behaviour 3 to pred day
            time between first behaviour 4 to pred day
            overall duration
            behaviour 1 duration
            behaviour 2 duration
            behaviour 3 duration
            behaviour 4 duration
            #log(t+1) for all above

            Total dimensions: 2 * (4 + 5 + 1) + 2

        """

        if time_thresh is None:
            time_thresh = self.TIME_THRESH

        #dim = 40
        #feature = np.zeros(dim, dtype=np.float)
        feature = []
        user_geos = np.array([data[i][1] for i in xrange(len(data)) if data[i][1] != ''])
        behaviors = np.array([data[i][0] for i in xrange(len(data))])
        times = np.array([data[i][2] for i in xrange(len(data))])

        b_1 = find_1D(behaviors == 1)
        b_2 = find_1D(behaviors == 2)
        b_3 = find_1D(behaviors == 3)
        b_4 = find_1D(behaviors == 4)

        #item_unique = np.unique(item_ids)

        # feature[0] = b_1.shape[0]
        # feature[1] = b_2.shape[0]
        # feature[2] = b_3.shape[0]
        # feature[3] = b_4.shape[0]
        feature += [b_1.shape[0], b_2.shape[0], b_3.shape[0], b_4.shape[0]]
        #feature += [float(b_4.shape[0])/float(b_1.shape[0] + self.small), float(b_4.shape[0])/float(b_2.shape[0] + self.small), float(b_4.shape[0])/float(b_3.shape[0] + self.small)]

        def check_time(times, behavior, type='max'):
            if behavior.shape[0] == 0:
                return time_start
            else:
                if type == 'max':
                    return np.max(times[behavior])
                elif type == 'min': 
                    return np.min(times[behavior])
                elif type == 'duration':
                    return np.max(times[behavior]) - np.min(times[behavior])

        # feature[4] = time_thresh - np.max(times)
        # feature[5] = time_thresh - check_time(times, b_1)
        # feature[6] = time_thresh - check_time(times, b_2)
        # feature[7] = time_thresh - check_time(times, b_3)
        # feature[8] = time_thresh - check_time(times, b_4)
        feature += [time_thresh - np.max(times), time_thresh - check_time(times, b_1), time_thresh - check_time(times, b_2), time_thresh - check_time(times, b_3), time_thresh - check_time(times, b_4)]

        # feature[9] = time_thresh - np.min(times)
        # feature[10] = time_thresh - check_time(times, b_1, type='min')
        # feature[11] = time_thresh - check_time(times, b_2, type='min')
        # feature[12] = time_thresh - check_time(times, b_3, type='min')
        # feature[13] = time_thresh - check_time(times, b_4, type='min')
        feature += [time_thresh - np.min(times), time_thresh - check_time(times, b_1, type='min'), time_thresh - check_time(times, b_2, type='min'), time_thresh - check_time(times, b_3, type='min'), time_thresh - check_time(times, b_4, type='min')]
    
        # feature[14] = np.max(times) - np.min(times)
        # feature[15] = check_time(times, b_1, type='duration')
        # feature[16] = check_time(times, b_2, type='duration')
        # feature[17] = check_time(times, b_3, type='duration')
        # feature[18] = check_time(times, b_4, type='duration')
        feature += [np.max(times) - np.min(times), check_time(times, b_1, type='duration'), check_time(times, b_2, type='duration'), check_time(times, b_3, type='duration'), check_time(times, b_4, type='duration')]

        #feature[19:38] = np.log(feature[0:19] + 2)

        dist = -1
        dist_norm = -1
        if iid in self.item_geos:
            for i_geo in self.item_geos[iid]:
                for u_geo in user_geos:
                    dist = max(dist, self.geo_dist(i_geo, u_geo))
                    dist_norm = max(dist_norm, self.geo_dist(i_geo, u_geo, normalize=True))
        
        # feature[38] = dist
        # feature[39] = dist_norm
        feature = [dist, dist_norm] + feature

        #global id
        #id += 1
        #if id%100000 == 0:
        #    print id

        return feature

    # def duration_hours(self, end, start='2014-11-18 00'):
    #     """
    #     Duration hours between two time strings.

    #     Parameters
    #     ----------
    #     end : end time. Format: 'yyyy-mm-dd hh'
    #     start : start time. Format: 'yyyy-mm-dd hh'. Default: '2014-11-18 00'.

    #     Returns
    #     -------
    #     The duration hours between the end time and the start time.

    #     """
    #     dt0 = datetime.strptime(start, '%Y-%m-%d %H')
    #     dt1 = datetime.strptime(end, '%Y-%m-%d %H')
    #     hours = lambda x, y: int((x - y).total_seconds() / 3600)
    #     return hours(dt1, dt0)

    def extract_features_item_2(self, data, label_dict=None, time_thresh=None, time_start=0):
        """
        Feature extraction in item level, which means for each item (if the item
        is in the final prediction list) this function will return a feature 
        vector.

        Parameters
        ----------
        data : data dict. Structure: {uid: {cid: {iid: [[behavior_type, user_geohash, time]]}}}
        label : True, return the labels for the feature vectors;
                False, do not return the labels.
        time_thresh : the time of the day to be predicted to day `2014-11-18 00`.
                        e.g., if one needs to predict the sale data in day `2014-12-18 00`,
                        then this time_thresh shoule be set to 
                        feature_extraction.duration_hours('2014-12-18 00')

        Returns
        -------
        features : N*D feature matrix. N is the number of items, D is the dimension of features.
        y : 1*N vector. Labels.
        keys : keys for each feature vector. It is a list of length N. Each element of the list
               is [uid, cid, iid].
        
        """
        if time_thresh is None:
            time_thresh = self.TIME_THRESH
        y = []
        features = []
        keys = []
        for uid in data:
            if uid > 0:
                temp_features_user, c_feat = self.extract_features_2_2(data[uid], time_thresh)
                #temp_features_user.append(self.extract_features_2_2(data[uid], time_thresh))
                for cid in data[uid]:
                    if cid > 0:
                        temp_features_cat, i_feat = self.extract_features_3(data[uid][cid], time_thresh, time_start)
                        #temp_features_cat.append(self.extract_features_3(data[uid][cid], time_thresh))
                        for iid in data[uid][cid]:
                            if iid in self.P_item_id_unique:
                                temp_features_item = self.extract_features_5(data[uid][cid][iid], iid, time_thresh, time_start)
                                #temp_features_item.append(self.extract_features_5(data[uid][cid][iid], iid, time_thresh))
                                if label_dict is not None:
                                    if tuple([uid, iid]) in label_dict:
                                        y.append(1)
                                    else:
                                        y.append(0)
                                keys.append([uid, cid, iid])
                                #features.append(np.hstack(temp_features_user + temp_features_cat + temp_features_item))
                                features.append(temp_features_item + temp_features_cat + temp_features_user + i_feat[iid] + c_feat[cid])
        return np.array(features), np.array(y), keys

    # def test_data(self, path='../../new/tianchi_mobile_recommend_train_user_toy.csv'):
    #     """
    #     Get the test data.

    #     Returns
    #     -------
    #     [U_data_dict_XY, special_case, last_days]
    #     See data_utils.gendata2
        
    #     """

    #     #_, P_item_id, _, _ = data_utils.load_P_item('newdata/itemdict')
    #     ## item dict, an efficiency trick
    #     #P_item_id_unique = np.unique(P_item_id).tolist()
    #     #P_item_id_unique = dict((el,0) for el in P_item_id_unique)

    #     dt0 = datetime.strptime('2014-11-18 00', '%Y-%m-%d %H')
    #     dt1 = datetime.strptime('2014-12-18 00', '%Y-%m-%d %H')
    #     duration_hours = lambda x, y: int((x - y).total_seconds() / 3600)
    #     time_thresh = duration_hours(dt1, dt0)

    #     U_data_dict = {}

    #     total_num = 0  
    #     #'../../new/tianchi_mobile_recommend_train_user_toy.csv'
    #     with open(path) as csvfile:
    #         spamreader = csv.reader(csvfile)
    #         id = -1
    #         for row in spamreader:
    #             if id == -1:
    #                 id += 1
    #                 continue
    #             uid = int(row[0])
    #             cid = int(row[4])
    #             iid = int(row[1])
    #             b = int(row[2])
    #             geo = row[3]
    #             dt_t = datetime.strptime(row[5], '%Y-%m-%d %H')
    #             time = duration_hours(dt_t, dt0)
    #             if uid in U_data_dict:
    #                 if cid in U_data_dict[uid]:
    #                     if iid in U_data_dict[uid][cid]:
    #                         U_data_dict[uid][cid][iid].append([b, geo, time])
    #                     else:
    #                         U_data_dict[uid][cid][iid] = [[b, geo, time]]
    #                 else:
    #                     U_data_dict[uid][cid] = {iid:[[b, geo, time]]}
    #             else:
    #                 U_data_dict[uid] = {cid:{iid:[[b, geo, time]]}}
    #             id += 1
    #             if id%100000 == 0:
    #                 print id

    #     return data_utils.gendata2(U_data_dict, self.P_item_id_unique, 0, 720, 744)
        