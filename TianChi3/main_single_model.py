import os
import numpy as np
import utils.data_utils_2 as data_utils
import utils.evaluation as evaluation
import cPickle as pickle
import csv
from datetime import datetime
from sklearn import tree, svm, linear_model, preprocessing, metrics, decomposition
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb

def concat_ic_feats_2(feat_list, ks):
    # item/cat independent feats concatenation
	feat_lens = [len(d.values()[0]) for d in feat_list]
	# print feat_lens
	ic_feats = np.zeros((len(ks), sum(feat_lens)))
	def get_value(feat_dict, key, length):
		if key in feat_dict:
			return feat_dict[key]
		else:
			return [0] * length
	for i in xrange(len(ks)):
		temp = []
		for j in xrange(0, len(feat_list), 2):
			temp += get_value(feat_list[j], ks[i][2], feat_lens[j]) + get_value(feat_list[j + 1], ks[i][1], feat_lens[j + 1])
		ic_feats[i] = np.array(temp)
		if i % 100000 == 0:
			print i
	return ic_feats

### IMPORTANT VAR
path = '../data_1'

# Load data
X_train, y_train, ks_train = data_utils.load_var(path + '/train_concat')
X_val, y_val, ks_val, result_val_truth = data_utils.load_var(path + '/val_concat')

feat_list_train = data_utils.load_var(path + '/train_ic_list')
ic_feats_train = concat_ic_feats_2(feat_list_train, ks_train)

feat_list_val = data_utils.load_var(path + '/val_ic_list')
ic_feats_val = concat_ic_feats_2(feat_list_val, ks_val)

X_train_raw = np.hstack((X_train, ic_feats_train))
X_val_raw = np.hstack((X_val, ic_feats_val))


### Tunning Logistic Regression

X_train_logreg = np.hstack((X_train_raw, np.log(X_train_raw + 2)))
X_val_logreg = np.hstack((X_val_raw, np.log(X_val_raw + 2)))

scaler = preprocessing.StandardScaler().fit(X_train_logreg)

X_scaled_train = scaler.transform(X_train_logreg)
X_scaled_val = scaler.transform(X_val_logreg)

# Parameters to be tuned
pcas=[0.999, 0.995, 0.99, 0.95]
Cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
tops = [360, 380, 400, 420, 440, 461, 480, 500, 520, 540, 560, 700, 1000, 1200, 1500, 2000, 3000, 5000]

models = []

log_file = path + '/logs/logreg_1'
with open(log_file, 'wb') as f:
	for p in pcas:
	    #del X_scaled_pca_train, X_scaled_pca_val
	    pca = decomposition.PCA(p)
	    pca.fit(X_scaled_train)
	    X_scaled_pca_train = pca.transform(X_scaled_train)
	    X_scaled_pca_val = pca.transform(X_scaled_val)
	    for c in Cs:
	        logreg = linear_model.LogisticRegression(C=c, verbose=1)
	        logreg.fit(X_scaled_pca_train, y_train)
	        pred_pca_val = logreg.predict_proba(X_scaled_pca_val)
	        fpr, tpr, thresholds = metrics.roc_curve(y_val, pred_pca_val[:, 0])
	        roc_auc = metrics.auc(fpr, tpr)
	        models.append([p, c, X_scaled_pca_val.shape, scaler, pca, logreg, pred_pca_val[:, 0]])
	        for top in tops:
	            t1 = np.argsort(pred_pca_val[:,0])[0:top]
	            y_pred_val = logreg.predict(X_scaled_pca_val)
	            y_pred_val[t1] = 1
	            result_val_pred = []
	            for i in xrange(len(ks_val)):
	                if y_pred_val[i] == 1:
	                    result_val_pred.append(ks_val[i])
	            E1 = evaluation.Evaluation(result_val_pred, result_val_truth)
	            f1 = E1.F1()
	            i1 = E1.intersection()
	            p1 = E1.precision()
	            r1 = E1.recall()
	            print 'pca: %f, c: %e, top: %d, auc: %f, f1: %f, i1: %d, p1: %f, r1: %f' % (p, c, top, roc_auc, f1, i1, p1, r1)
	            f.write('pca: %f, c: %e, top: %d, auc: %f, f1: %f, i1: %d, p1: %f, r1: %f\n' % (p, c, top, roc_auc, f1, i1, p1, r1))

models_file = path + '/models/logreg1.pkl'
with open(models_file, 'wb') as fp:
	pickle.dump(models, fp, protocol=2)


### Tunning GBDT
lrs = [5e-2, 1e-1, 3e-1]
depths = [2,3,4,7,9]
ns = [100, 200, 300, 400]
tops = [360, 380, 400, 420, 440, 461, 480, 500, 520, 540, 560, 700, 1000, 1200, 1500, 2000, 3000, 5000]

gbdts = []

log_file = path + '/logs/gbdt1'
with open(log_file, 'wb') as f:
	for d in depths:
		for lr in lrs:
			for n in ns:
				params = {'learning_rate':lr, 
						  'max_depth':d, 
						  'n_estimators':n,
					   	  'objective':'binary:logistic',
					   	  'silent':True}
				gbdt = xgb.XGBClassifier(**params)
				gbdt.fit(X_train_raw, y_train)
				pred_val = gbdt.predict_proba(X_val_raw)
				fpr, tpr, thresholds = metrics.roc_curve(y_val, pred_val[:, 0])
				roc_auc = metrics.auc(fpr, tpr)
				gbdts.append(gbdt)
				for top in tops:
					t1 = np.argsort(pred_val[:,0])[0:top]
					y_pred_val = np.zeros_like(y_val)
					y_pred_val[t1] = 1
					result_val_pred = []
					for i in xrange(len(ks_val)):
						if y_pred_val[i] == 1:
							result_val_pred.append(ks_val[i])
					E1 = evaluation.Evaluation(result_val_pred, result_val_truth)
					f1 = E1.F1()
					i1 = E1.intersection()
					p1 = E1.precision()
					r1 = E1.recall()
					print 'd: %d, lr: %e, n: %d, top: %d, auc: %f, f1: %f, i1: %d, p1: %f, r1: %f' % (d, lr, n, top, roc_auc, f1, i1, p1, r1)
					f.write('d: %d, lr: %e, n: %d, top: %d, auc: %f, f1: %f, i1: %d, p1: %f, r1: %f\n' % (d, lr, n, top, roc_auc, f1, i1, p1, r1))

### For GBDT, bacause xgboost is used, the models cannot be saved in one file
model_file_prefix = path + '/models/gbdt'
for id in xrange(len(gbdts)):
    gbdts[id]._Booster.dump_model(model_file_prefix + str(id) + '.dump.raw.txt')
    gbdts[id]._Booster.save_model(model_file_prefix + str(id) + '.model')


print 'Completed!'

# How to use pretrained GBDT model???

# from sklearn.preprocessing import LabelEncoder
# gbdt_new = xgb.XGBClassifier()
# gbdt_new._Booster.load_model('model_path')
# gbdt_new._le = LabelEncoder().fit(y_train)
# # OK!!!


### Make prediction on test data
X_test, ks_test = data_utils.load_var(path + '/test_concat')

feat_list_test = data_utils.load_var(path + '/test_ic_list')
ic_feats_test = concat_ic_feats_2(feat_list_test, ks_test)

X_test_raw = np.hstack((X_test, ic_feats_test))

pred_test = gbdt.predict_proba(X_test_raw)[:, 0]

top = 440
t1 = np.argsort(pred_test)[0:top]
y_pred_test = np.zeros_like(pred_test)
y_pred_test[t1] = 1

result_test_pred = []

for i in xrange(len(ks_test)):
    if y_pred_test[i] == 1:
        result_test_pred.append([ks_test[i][0], ks_test[i][2]])

result_file = path + '/results/result1.csv'
with open(result_file, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['user_id', 'item_id'])
    for pred in result_test_pred:
        spamwriter.writerow(pred)


### Then the `result_file` can be submitted.
### For the best single GBDT model, the F1 score is around 10.4%, which has a rank about 100 in Season 1.

### To improve the performance
###     1. By changing the time parameters in utils.gen_feats, and utils.gen_ic_ind_feats, more labeled data can be generated, which is useful because the dataset is highly unbalanced.
###     2. Add more time intervals in utils.gen_feats, and utils.gen_ic_ind_feats
###     3. Cross-validation can be used to select better hyper-parameters.
###     4. Model ensemble can be used. A possilbe method:
###         1) Train multiple 'not bad' single models with different algorithms (logistic regression, GBDT, adaboost, randomforest...)
###         2) Use single models' outputs as input to train a new logistic regression
###         3) Use the outer logistic regression to make the prediction
