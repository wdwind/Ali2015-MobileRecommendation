#-*-coding:utf-8-*-
"""
生成feature matrix
默认：
    train data : 2014-11-18 ~ 2014-12-16 (with label)
    val data   : 2014-11-19 ~ 2014-12-17 (with label)
    test data  : 2014-11-21 ~ 2014-12-19 (without label)

"""

import os
import numpy as np
import utils.data_utils_2 as data_utils
import utils.feature_extraction_new_3 as feature_extraction
import cPickle as pickle
from datetime import datetime
from utils.time_utils import *
#import utils.time_utils

def concat3(data_list, ks_list):
	# data size from small to big
	data = data_list[0]
	ks_dict = {tuple([ks_list[0][i][0], ks_list[0][i][2]]):i for i in xrange(len(ks_list[0]))}
	for i in xrange(1, len(data_list)):
		temp = np.zeros((data_list[i].shape[0], data.shape[1]))
		for j in xrange(len(ks_list[i])):
			if tuple([ks_list[i][j][0], ks_list[i][j][2]]) in ks_dict:
				ind = ks_dict[tuple([ks_list[i][j][0], ks_list[i][j][2]])]
				temp[j, :] = data[ind, :]
		data = np.hstack((data_list[i], temp))
		temp_key = {tuple([ks_list[i][k][0], ks_list[i][k][2]]):k for k in xrange(len(ks_list[i]))}
	return data

def gen_uci_feats(path='../data_',part=1):
    dates = os.listdir(path + str(part) + '/date')
    dates.sort()
    #dates

    global dt0
    global duration_hours

    fe = feature_extraction.Feature(path + str(part) + '/itemdict')

    _, P_item_id, _, _ = data_utils.load_P_item(path + str(part) + '/itemdict')
    P_item_id_unique = np.unique(P_item_id).tolist()
    P_item_id_unique = dict((el,0) for el in P_item_id_unique)

    # Train
    print 'Extracting train feats...'

    train_time_start_1 = '2014-11-18'
    train_time_start_2 = '2014-12-15'
    train_time_start_3 = '2014-12-13'
    train_time_start_4 = '2014-12-09'

    train_time_end = '2014-12-15'
    train_time_thresh = '2014-12-16'

    dt_train_start_1 = datetime.strptime(train_time_start_1 + ' 00', '%Y-%m-%d %H')
    dt_train_start_2 = datetime.strptime(train_time_start_2 + ' 00', '%Y-%m-%d %H')
    dt_train_start_3 = datetime.strptime(train_time_start_3 + ' 00', '%Y-%m-%d %H')
    dt_train_start_4 = datetime.strptime(train_time_start_4 + ' 00', '%Y-%m-%d %H')
    dt_train_thresh = datetime.strptime(train_time_thresh + ' 00', '%Y-%m-%d %H')

    U_train_item_1, label_train_1, result_train_truth_1 = data_utils.get_data_dict_4(P_item_id_unique, None, train_time_start_1, train_time_end, True, 'uci', path + str(part) + '/date/')
    X_train_1, y_train_1, ks_train_1 = fe.extract_features_item_2(U_train_item_1, label_train_1, duration_hours(dt_train_thresh, dt0), duration_hours(dt_train_start_1, dt0))

    U_train_item_2, label_train_2, result_train_truth_2 = data_utils.get_data_dict_4(P_item_id_unique, None, train_time_start_2, train_time_end, True, 'uci', path + str(part) + '/date/')
    X_train_2, y_train_2, ks_train_2 = fe.extract_features_item_2(U_train_item_2, label_train_2, duration_hours(dt_train_thresh, dt0), duration_hours(dt_train_start_2, dt0))

    U_train_item_3, label_train_3, result_train_truth_3 = data_utils.get_data_dict_4(P_item_id_unique, None, train_time_start_3, train_time_end, True, 'uci', path + str(part) + '/date/')
    X_train_3, y_train_3, ks_train_3 = fe.extract_features_item_2(U_train_item_3, label_train_3, duration_hours(dt_train_thresh, dt0), duration_hours(dt_train_start_3, dt0))

    U_train_item_4, label_train_4, result_train_truth_4 = data_utils.get_data_dict_4(P_item_id_unique, None, train_time_start_4, train_time_end, True, 'uci', path + str(part) + '/date/')
    X_train_4, y_train_4, ks_train_4 = fe.extract_features_item_2(U_train_item_4, label_train_4, duration_hours(dt_train_thresh, dt0), duration_hours(dt_train_start_4, dt0))

    train_list = [X_train_1, y_train_1, ks_train_1, result_train_truth_1, X_train_2, y_train_2, ks_train_2, result_train_truth_2, X_train_3, y_train_3, ks_train_3, result_train_truth_3, X_train_4, y_train_4, ks_train_4, result_train_truth_4]

    print 'Raw train feats list: ' + path + str(part) + '/train_list'
    with open(path + str(part) + '/train_list', 'wb') as fp:
        pickle.dump(train_list, fp, protocol=2)

    X_train = concat3([X_train_2, X_train_3, X_train_4, X_train_1], [ks_train_2, ks_train_3, ks_train_4, ks_train_1])
    y_train = y_train_1
    ks_train = ks_train_1

    train_concat = [X_train, y_train, ks_train]

    print 'Concatenated train feats: ' + path + str(part) + '/train_concat'
    with open(path + str(part) + '/train_concat', 'wb') as fp:
        pickle.dump(train_concat, fp, protocol=2)

    # Val
    print 'Extracting val feats...'

    val_time_start_1 = '2014-11-19'
    val_time_start_2 = '2014-12-16'
    val_time_start_3 = '2014-12-14'
    val_time_start_4 = '2014-12-10'

    val_time_end = '2014-12-16'
    val_time_thresh = '2014-12-17'

    dt_val_start_1 = datetime.strptime(val_time_start_1 + ' 00', '%Y-%m-%d %H')
    dt_val_start_2 = datetime.strptime(val_time_start_2 + ' 00', '%Y-%m-%d %H')
    dt_val_start_3 = datetime.strptime(val_time_start_3 + ' 00', '%Y-%m-%d %H')
    dt_val_start_4 = datetime.strptime(val_time_start_4 + ' 00', '%Y-%m-%d %H')
    dt_val_thresh = datetime.strptime(val_time_thresh + ' 00', '%Y-%m-%d %H')

    U_val_item_1, label_val_1, result_val_truth_1 = data_utils.get_data_dict_4(P_item_id_unique, None, val_time_start_1, val_time_end, True, 'uci', path + str(part) + '/date/')
    X_val_1, y_val_1, ks_val_1 = fe.extract_features_item_2(U_val_item_1, label_val_1, duration_hours(dt_val_thresh, dt0), duration_hours(dt_val_start_1, dt0))

    U_val_item_2, label_val_2, result_val_truth_2 = data_utils.get_data_dict_4(P_item_id_unique, None, val_time_start_2, val_time_end, True, 'uci', path + str(part) + '/date/')
    X_val_2, y_val_2, ks_val_2 = fe.extract_features_item_2(U_val_item_2, label_val_2, duration_hours(dt_val_thresh, dt0), duration_hours(dt_val_start_2, dt0))

    U_val_item_3, label_val_3, result_val_truth_3 = data_utils.get_data_dict_4(P_item_id_unique, None, val_time_start_3, val_time_end, True, 'uci', path + str(part) + '/date/')
    X_val_3, y_val_3, ks_val_3 = fe.extract_features_item_2(U_val_item_3, label_val_3, duration_hours(dt_val_thresh, dt0), duration_hours(dt_val_start_3, dt0))

    U_val_item_4, label_val_4, result_val_truth_4 = data_utils.get_data_dict_4(P_item_id_unique, None, val_time_start_4, val_time_end, True, 'uci', path + str(part) + '/date/')
    X_val_4, y_val_4, ks_val_4 = fe.extract_features_item_2(U_val_item_4, label_val_4, duration_hours(dt_val_thresh, dt0), duration_hours(dt_val_start_4, dt0))

    val_list = [X_val_1, y_val_1, ks_val_1, result_val_truth_1, X_val_2, y_val_2, ks_val_2, result_val_truth_2, X_val_3, y_val_3, ks_val_3, result_val_truth_3, X_val_4, y_val_4, ks_val_4, result_val_truth_4]

    print 'Raw val feats list: ' + path + str(part) + '/val_list'
    with open(path + str(part) + '/val_list', 'wb') as fp:
        pickle.dump(val_list, fp, protocol=2)

    X_val = concat3([X_val_2, X_val_3, X_val_4, X_val_1], [ks_val_2, ks_val_3, ks_val_4, ks_val_1])
    y_val = y_val_1
    ks_val = ks_val_1
    result_val_truth = result_val_truth_1

    val_concat = [X_val, y_val, ks_val, result_val_truth]

    print 'Concatenated val feats: ' + path + str(part) + '/val_concat'
    with open(path + str(part) + '/val_concat', 'wb') as fp:
        pickle.dump(val_concat, fp, protocol=2)

    # Test
    print 'Extracting test feats...'

    test_time_start_1 = '2014-11-21'
    test_time_start_2 = '2014-12-18'
    test_time_start_3 = '2014-12-16'
    test_time_start_4 = '2014-12-12'

    test_time_end = '2014-12-18'
    test_time_thresh = '2014-12-19'

    duration_hours = lambda x, y: int((x - y).total_seconds() / 3600)

    dt_test_start_1 = datetime.strptime(test_time_start_1 + ' 00', '%Y-%m-%d %H')
    dt_test_start_2 = datetime.strptime(test_time_start_2 + ' 00', '%Y-%m-%d %H')
    dt_test_start_3 = datetime.strptime(test_time_start_3 + ' 00', '%Y-%m-%d %H')
    dt_test_start_4 = datetime.strptime(test_time_start_4 + ' 00', '%Y-%m-%d %H')
    dt_test_thresh = datetime.strptime(test_time_thresh + ' 00', '%Y-%m-%d %H')

    U_test_item_1 = data_utils.get_data_dict_4(P_item_id_unique, None, test_time_start_1, test_time_end, False, 'uci', path + str(part) + '/date/')
    X_test_1, _, ks_test_1 = fe.extract_features_item_2(U_test_item_1, None, duration_hours(dt_test_thresh, dt0), duration_hours(dt_test_start_1, dt0))

    U_test_item_2 = data_utils.get_data_dict_4(P_item_id_unique, None, test_time_start_2, test_time_end, False, 'uci', path + str(part) + '/date/')
    X_test_2, _, ks_test_2 = fe.extract_features_item_2(U_test_item_2, None, duration_hours(dt_test_thresh, dt0), duration_hours(dt_test_start_2, dt0))

    U_test_item_3 = data_utils.get_data_dict_4(P_item_id_unique, None, test_time_start_3, test_time_end, False, 'uci', path + str(part) + '/date/')
    X_test_3, _, ks_test_3 = fe.extract_features_item_2(U_test_item_3, None, duration_hours(dt_test_thresh, dt0), duration_hours(dt_test_start_3, dt0))

    U_test_item_4 = data_utils.get_data_dict_4(P_item_id_unique, None, test_time_start_4, test_time_end, False, 'uci', path + str(part) + '/date/')
    X_test_4, _, ks_test_4 = fe.extract_features_item_2(U_test_item_4, None, duration_hours(dt_test_thresh, dt0), duration_hours(dt_test_start_4, dt0))

    test_list = [X_test_1, ks_test_1, X_test_2, ks_test_2, X_test_3, ks_test_3, X_test_4, ks_test_4]

    print 'Raw test feats list: ' + path + str(part) + '/test_list'
    with open(path + str(part) + '/test_list', 'wb') as fp:
        pickle.dump(test_list, fp, protocol=2)

    X_test = concat3([X_test_2, X_test_3, X_test_4, X_test_1], [ks_test_2, ks_test_3, ks_test_4, ks_test_1])
    ks_test = ks_test_1

    test_concat = [X_test, ks_test]

    print 'Concatenated test feats: ' + path + str(part) + '/test_concat'
    with open(path + str(part) + '/test_concat', 'wb') as fp:
        pickle.dump(test_concat, fp, protocol=2)
