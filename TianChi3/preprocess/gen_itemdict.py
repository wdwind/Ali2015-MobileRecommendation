import csv
import os
import cPickle as pickle
import numpy as np


def gen_itemdict(part=1):
    P_num = 0

    with open('../data_' + str(part) + '/tianchi_mobile_recommend_train_item.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        spamreader.next()
        for row in spamreader:
            P_num += 1

    P_item_id = np.zeros(P_num, dtype=np.int)
    P_item_geo = np.zeros(P_num, dtype='S7')
    P_item_cat = np.zeros(P_num, dtype=np.int)

    with open('../data_' + str(part) + '/tianchi_mobile_recommend_train_item.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        spamreader.next()
        id = 0
        for row in spamreader:
            P_item_id[id] = row[0]
            P_item_geo[id] = row[1]
            P_item_cat[id] = row[2]
            id += 1

    itemdict = {'item_num':P_num, 'item_id':P_item_id, 'item_geo':P_item_geo, 'item_cat':P_item_cat}

    with open('../data_' + str(part) + '/itemdict', 'wb') as fp:
        pickle.dump(itemdict, fp, protocol=2)