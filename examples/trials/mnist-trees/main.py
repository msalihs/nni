# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nni
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging
import numpy as np

LOG = logging.getLogger('sklearn_classification')

def load_data():
    '''Load dataset, use 20newsgroups dataset'''
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_default_parameters():
    '''get default parameters'''
    params = {
        'model': "RandomForest",
        'max_depth': 2,
        'n_estimators': 5
    }
    return params

def select_model(model):
    if model['_name'] == "RandomForest":
        clf = RandomForestClassifier()
        clf.max_features = model['max_features']
        clf.min_samples_split = model['min_samples_split']
    elif model['_name'] == "LightGBM":
        clf = LGBMClassifier()
        clf.num_leaves = model['num_leaves']
        clf.learning_rate = model['learning_rate']
    elif model['_name'] == "XGBoost":
        clf = XGBClassifier()
        clf.booster = model['booster']
        clf.learning_rate = model['learning_rate']
    else:
        raise ValueError
    return clf

def get_model(PARAMS):
    '''Get model according to parameters'''
    model = select_model(PARAMS.get('model'))
    model.n_estimators = PARAMS.get('n_estimators')
    model.max_depth = PARAMS.get('max_depth')

    return model

def run(X_train, X_test, y_train, y_test, model):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    LOG.debug('score: %s' % score)
    nni.report_final_result(score)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
