# -*- coding: utf-8 -*-
"""
Created on Wed Mar 2 08:45:48 2016
@author: mariarosa

This script imports the pain connectivity dataset and runs a SVM-based
classification of patients vs controls from multiple sites.

"""

# Import packages
import numpy as np
import load_pain_data as pd
from scipy import stats
from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from random import sample

# Get features
X_UK = pd.X_UK
Y_UK = pd.Y_UK
X_JP = pd.X_JP
Y_JP = pd.Y_JP
X_US = pd.X_US
Y_US = pd.Y_US

# Classify gender
# Y_US = [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
#     1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,1, 0, 0, 0, 0, 1, 1, 1, \
#     1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# Y_US = np.array(Y_US)

# Classify depressed
# Diagnosis
# Y_US = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
#      0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, \
#      1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0]
# Y_US = np.array(Y_US)
# Depression score
# BDI = [3, 0, 0, 0, 0, 4, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 3, 0, 0, \
#     0, 0, 0, 1, 4, 10, 6, 7, 0, 8, 0, 15, 1, 12, 13, 8, 6, 4, 4, 3, \
#     13, 4, 6, 9, 5, 3, 12, 0, 0, 17, 0, 10, 0, 0, 0, 4, 5, 2, 10, 19, 19, 2]

# Z transform
X_UK = np.arctanh(X_UK)
X_JP = np.arctanh(X_JP)
X_US = np.arctanh(X_US)

# Classifier (SVM)
clf = svm.SVC(kernel='linear')
parameters = {'C':[0.0001,0.001, 0.01, 0.1, 1.0, 10, 100]}
# clfgrid = GridSearchCV(clf, parameters, cv=lolAll)

# T-test threshold for feature selection
ttest_thres = 0.05

# Number of permutations
nperms = 1

# Number of re-sampling times
nsamp = 200

# Initialise sensitivity, specificity and weights
se_AllReUS_perms = np.ones(nperms)*0
sp_AllReUS_perms = np.ones(nperms)*0
weights = np.ones((nsamp,X_US.shape[1]))*0

# Find patients and controls for UK
n_uk = int(sum(Y_UK))
ind_uk = np.asarray(range(len(Y_UK)))
ind_uk_n = ind_uk[Y_UK==0]
ind_uk_p = ind_uk[Y_UK==1]

# Find patients and controls for US
n_jp = int(sum(Y_JP))
ind_jp = np.asarray(range(len(Y_JP)))
ind_jp_n = ind_jp[Y_JP==0]
ind_jp_p = ind_jp[Y_JP==1]

# Create sampling balanced labels
Y_AllRe = np.hstack((np.hstack((np.ones(n_uk),np.ones(n_uk)*0)),np.hstack((np.ones(n_jp),np.ones(n_jp)*0))))

# Create labels for stratified CV
labelsUK = np.concatenate((range(n_uk),range(n_uk)))
labelsJP = np.concatenate((range(n_jp),range(n_jp)))
labelsAll = np.concatenate((labelsUK,max(labelsUK)+1+labelsJP))
lolAll = cross_validation.LeaveOneLabelOut(labelsAll)

# Set stratified CV
clfgrid = GridSearchCV(clf, parameters, cv=lolAll)

# Run classification and permutations
print 'Train(UK+JP)+sampling>>Test(US)'

for perms in range(nperms):

    print('Permutation: {0} of {1}'.format(perms, nperms))

    se_AllReUS = []
    sp_AllReUS = []

    for i in range(nsamp):

        # Get nem sample for UK and JP
        sp_uk = sample(ind_uk_n,n_uk)
        ind_uk_all = np.hstack((ind_uk_p,sp_uk))
        sp_jp = sample(ind_jp_n,n_jp)
        ind_jp_all = np.hstack((ind_jp_p,sp_jp))

        # Get new data for new sample
        X_AllRe = np.vstack((X_UK[ind_uk_all,::],X_JP[ind_jp_all,::]))

        # Suffle training labels
        if (perms != 0):
            Y_AllRe = shuffle(Y_AllRe)

        # New training and testing data data
        X_train = X_AllRe
        Y_train = Y_AllRe
        X_test = X_US
        Y_test = Y_US

        # Scale training and testing data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Run feature selection
        tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres

        # Train grid search search for optimal parameter
        clfgrid.fit(X_train[::,tt_ind],Y_train)

        # Get optimal parameter to test
        clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])

        # Train with optimal C value
        clf.fit(X_train[::,tt_ind],Y_train)

        # Save weights and decision function
        if (perms == 0):
            weights[i,tt_ind] = clf.coef_
            dec_tmp = clf.decision_function(X_test[::,tt_ind])

        # Get predictions
        pred_tmp = clf.predict(X_test[::,tt_ind])
        se_AllReUS.append(sum(pred_tmp[Y_test==1]==1)/float(sum(Y_test==1)))
        sp_AllReUS.append(sum(pred_tmp[Y_test==0]==0)/float(sum(Y_test==0)))

    # Calculate specificity and sensitivity
    se_AllReUS_perms[perms] = np.mean(np.array(se_AllReUS))
    sp_AllReUS_perms[perms] = np.mean(np.array(sp_AllReUS))

print "Accuracies:"
fmt_string = "'{}' - Accuracy: '{}', Sensitivity: '{}', Specificity: '{}'"
print fmt_string.format('AllReUS',np.round((sp_AllReUS_perms[0] + se_AllReUS_perms[0])/2, decimals=2),np.round(se_AllReUS_perms[0], decimals=2),np.round(sp_AllReUS_perms[0], decimals=2))

print "P-value"
if (nperms > 1):
    pval_acc = sum(((sp_AllReUS_perms[1::] + se_AllReUS_perms[1::])/2)>=((sp_AllReUS_perms[0] + se_AllReUS_perms[0])/2))/float((nperms-1))
    pval_se = sum(se_AllReUS_perms[1::]>=se_AllReUS_perms[0])/float((nperms-1))
    pval_sp = sum(sp_AllReUS_perms[1::]>=sp_AllReUS_perms[0])/float((nperms-1))
    fmt_string = "'{}' - Accuracy: '{}', Sensitivity: '{}', Specificity: '{}'"
    print fmt_string.format('AllReUS',np.round(pval_acc, decimals=2),np.round(pval_se, decimals=2),np.round(pval_sp, decimals=2))
