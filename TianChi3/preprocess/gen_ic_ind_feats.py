#-*-coding:utf-8-*-
"""
生成item independent feature matrix，和cat independent feature matrix
默认：
    train data : 2014-11-18 ~ 2014-12-16 (with label)
    val data   : 2014-11-19 ~ 2014-12-17 (with label)
    test data  : 2014-11-21 ~ 2014-12-19 (without label)

"""

import os
import numpy as np
import utils.data_utils_2 as data_utils
import utils.feature_extraction_ic as feature_extraction
import cPickle as pickle
from datetime import datetime
from utils.time_utils import *

def gen_ic_feats(path='../data_',part=1):
    print 'Extracting user/cat-independent feats...'

    dates = os.listdir(path + str(part) + '/date')
    dates.sort()
    #dates

    _, P_item_id, _, P_item_cat = data_utils.load_P_item(path + str(part) + '/itemdict')
    P_item_id_unique = np.unique(P_item_id).tolist()
    P_item_id_unique = dict((el,0) for el in P_item_id_unique)
    P_item_cat_unique = np.unique(P_item_cat).tolist()
    P_item_cat_unique = dict((el, 0) for el in P_item_cat_unique)

    global duration_hours
    global dt0

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

    U_train_item_1 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, train_time_start_1, train_time_end, False, 'ci', path + str(part) + '/date/')
    fi_train_1, fc_train_1 = feature_extraction.wrapper(U_train_item_1, P_item_id_unique, duration_hours(dt_train_start_1, dt0), duration_hours(dt_train_thresh, dt0))

    U_train_item_2 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, train_time_start_2, train_time_end, False, 'ci', path + str(part) + '/date/')
    fi_train_2, fc_train_2 = feature_extraction.wrapper(U_train_item_2, P_item_id_unique, duration_hours(dt_train_start_2, dt0), duration_hours(dt_train_thresh, dt0))

    U_train_item_3 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, train_time_start_3, train_time_end, False, 'ci', path + str(part) + '/date/')
    fi_train_3, fc_train_3 = feature_extraction.wrapper(U_train_item_3, P_item_id_unique, duration_hours(dt_train_start_3, dt0), duration_hours(dt_train_thresh, dt0))

    U_train_item_4 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, train_time_start_4, train_time_end, False, 'ci', path + str(part) + '/date/')
    fi_train_4, fc_train_4 = feature_extraction.wrapper(U_train_item_4, P_item_id_unique, duration_hours(dt_train_start_4, dt0), duration_hours(dt_train_thresh, dt0))

    train_ic_list = [fi_train_1, fc_train_1, fi_train_2, fc_train_2, fi_train_3, fc_train_3, fi_train_4, fc_train_4]

    print 'Raw train ic feats list: ' + path + str(part) + '/train_ic_list'
    with open(path + str(part) + '/train_ic_list', 'wb') as fp:
        pickle.dump(train_ic_list, fp, protocol=2)
    
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

    U_val_item_1 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, val_time_start_1, val_time_end, False, 'ci', path + str(part) + '/date/')
    fi_val_1, fc_val_1 = feature_extraction.wrapper(U_val_item_1, P_item_id_unique, duration_hours(dt_val_start_1, dt0), duration_hours(dt_val_thresh, dt0))

    U_val_item_2 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, val_time_start_2, val_time_end, False, 'ci', path + str(part) + '/date/')
    fi_val_2, fc_val_2 = feature_extraction.wrapper(U_val_item_2, P_item_id_unique, duration_hours(dt_val_start_2, dt0), duration_hours(dt_val_thresh, dt0))

    U_val_item_3 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, val_time_start_3, val_time_end, False, 'ci', path + str(part) + '/date/')
    fi_val_3, fc_val_3 = feature_extraction.wrapper(U_val_item_3, P_item_id_unique, duration_hours(dt_val_start_3, dt0), duration_hours(dt_val_thresh, dt0))

    U_val_item_4 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, val_time_start_4, val_time_end, False, 'ci', path + str(part) + '/date/')
    fi_val_4, fc_val_4 = feature_extraction.wrapper(U_val_item_4, P_item_id_unique, duration_hours(dt_val_start_4, dt0), duration_hours(dt_val_thresh, dt0))

    val_ic_list = [fi_val_1, fc_val_1, fi_val_2, fc_val_2, fi_val_3, fc_val_3, fi_val_4, fc_val_4]

    print 'Raw val ic feats list: ' + path + str(part) + '/val_ic_list'
    with open(path + str(part) + '/val_ic_list', 'wb') as fp:
        pickle.dump(val_ic_list, fp, protocol=2)


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


    U_test_item_1 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, test_time_start_1, test_time_end, False, 'ci', path + str(part) + '/date/')
    fi_test_1, fc_test_1 = feature_extraction.wrapper(U_test_item_1, P_item_id_unique, duration_hours(dt_test_start_1, dt0), duration_hours(dt_test_thresh, dt0))

    U_test_item_2 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, test_time_start_2, test_time_end, False, 'ci', path + str(part) + '/date/')
    fi_test_2, fc_test_2 = feature_extraction.wrapper(U_test_item_2, P_item_id_unique, duration_hours(dt_test_start_2, dt0), duration_hours(dt_test_thresh, dt0))

    U_test_item_3 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, test_time_start_3, test_time_end, False, 'ci', path + str(part) + '/date/')
    fi_test_3, fc_test_3 = feature_extraction.wrapper(U_test_item_3, P_item_id_unique, duration_hours(dt_test_start_3, dt0), duration_hours(dt_test_thresh, dt0))

    U_test_item_4 = data_utils.get_data_dict_4(P_item_id_unique, P_item_cat_unique, test_time_start_4, test_time_end, False, 'ci', path + str(part) + '/date/')
    fi_test_4, fc_test_4 = feature_extraction.wrapper(U_test_item_4, P_item_id_unique, duration_hours(dt_test_start_4, dt0), duration_hours(dt_test_thresh, dt0))

    test_ic_list = [fi_test_1, fc_test_1, fi_test_2, fc_test_2, fi_test_3, fc_test_3, fi_test_4, fc_test_4]

    print 'Raw test ic feats list: ' + path + str(part) + '/test_ic_list'
    with open(path + str(part) + '/test_ic_list', 'wb') as fp:
        pickle.dump(test_ic_list, fp, protocol=2)

    print 'Completed!'
