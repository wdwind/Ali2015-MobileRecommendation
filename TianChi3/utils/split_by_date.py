#-*-coding:utf-8-*-
"""
将tianchi_mobile_recommend_train_user.csv按照日期分割为31份**.pkl文件，放在'/data/date/'目录下。

[[uid (int), cid (int), iid (int), behavior_type (int), geohash (string), hour to 2014.11.18 00 (int)], ...]

例子：
[[99512554, 37320317, 9232, 3, '94gn6nd', 20], ...]

"""

import csv
import os
import cPickle as pickle
from datetime import datetime
from utils.time_utils import duration_hours, dt0

def splitByDate(part=1):
    date_dictionary = {}
    try:
        os.mkdir('../data_' + str(part) + '/date')
    except:
        pass
    with open('../data_' + str(part) + '/tianchi_mobile_recommend_train_user_toy.csv') as csvfile:
        rows = csv.reader(csvfile)
        rows.next()
        for row in rows:
            dt_t = datetime.strptime(row[5], '%Y-%m-%d %H')
            data = [int(row[0]), int(row[4]), int(row[1]), int(row[2]), row[3], duration_hours(dt_t, dt0)]
            date = row[-1].split(" ")[0]
            if date in date_dictionary:
                date_dictionary[date].append(data)
            else:
                date_dictionary[date] = [data]
    os.chdir('../data_' + str(part) + '/date/')
    for date in date_dictionary:
        os.mkdir(date)
        file_name = "raw.pkl"
        with open(date + '/' + file_name, 'wb') as fp:
            pickle.dump(date_dictionary[date], fp, protocol=2)
    os.chdir('../../TianChi3/')

