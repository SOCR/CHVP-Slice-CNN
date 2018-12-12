import numpy as np
from os.path import basename, splitext, join
import json
import math
import os
import argparse
import sys

sys.path.insert(0, './src')

from getData import read_train_files, read_test_files
# from test import test_model, test_model_prior, ROC_score

#keras import
import keras

from keras.models import model_from_json
from keras.layers.wrappers import TimeDistributed
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback

from sklearn.metrics import roc_auc_score as roc

project_dir = "./"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
feats_dir = join(project_dir, "feats")

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN training parameters.')
    # parse command line options
    parser.add_argument('-load', '--load_model', type=str, help='Name of the load model.')
    parser.add_argument('-save', '--save_model', type=str, help='Name of the save model.')
    parser.add_argument('-num_epoch', '--num_epoch', type=int, help ="Number of epoch to train the model")
    parser.add_argument('-train_file_dir', '--train_file_dir', type=str, help ="Directory of training data.")
    parser.add_argument('-test_file_dir', '--test_file_dir', type=str, help ="Directory of testing data.")
    parser.add_argument('-num_sessions_each_dim', '--num_sessions_each_dim', 
        type=int, help ="Number of sessions for each dimension.")
    
    args = parser.parse_args()
    return args

def ROC_score(model, x_test, y_true):
    
    y_predict = model.predict([np.asarray(x_test[0]), np.asarray(x_test[1]), np.asarray(x_test[2])])
    # transform to numpy
    y_predict = np.asarray(y_predict)
    y_true = np.asarray(y_true)
    roc_score = round(roc(y_true, y_predict), 5)
    print('roc: ', roc_score)
    
    diff = 0
    for i in range(len(y_true)):
        predict_label = int(y_predict[i][0] < y_predict[i][1])
        if predict_label != y_true[i][1]:
            diff += 1
            
    acc_score = round(float(len(y_true) - diff)/len(y_true), 5)
    print('acc: ', acc_score)

    return roc_score, acc_score

def acc_score(model, x_test, y_true):
    y_predict = model.predict([np.asarray(x_test[0]), np.asarray(x_test[1]), np.asarray(x_test[2])])
    y_predict = np.asarray(y_predict)
    y_true = np.asarray(y_true)
    # print(y_score)
    diff = 0
    for i in range(len(y_true)):
        predict_label = int(y_predict[i][0] < y_predict[i][1])
        if predict_label != y_true[i][1]:
            diff += 1
            
    acc_score = round(float(len(y_true) - diff)/len(y_true), 5)
    print('acc: ', acc_score)
    return acc_score


args = arg_parse()
model_name = args.load_model


load_weight = False

load_model_name = args.load_model
save_model_name = args.save_model

models_dir = join(models_dir, load_model_name)
# parameter
nEpoch = args.num_epoch

model_structure = join(models_dir, 'model_structure')

model_weight_path = join(models_dir, 'initialize_weight')


# Read file 
X_train, Y_train = read_train_files(join(corpus_dir, args.train_file_dir), args.num_sessions_each_dim)

# Need both x and y to calculate score
x_test, y_test = read_train_files(join(corpus_dir, args.test_file_dir), args.num_sessions_each_dim)

# new model directory for new parameter
models_dir = join(project_dir, "models")

# make model directory
os.makedirs(join(models_dir, save_model_name))
models_dir = join(models_dir, save_model_name)



# load model structure and weight
with open(model_structure) as json_file:
    model_architecture = json.load(json_file)

model = model_from_json(model_architecture)

if load_weight:
    model.load_weights(model_weight_path, by_name = True)

model.compile(loss='mean_squared_error', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])

# model.get_weights()
# model.summary()

# Save model structure
model_architecture = model.to_json()

with open(models_dir+'/model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)

acc_highest = 0
roc_highest = 0
for i in range(nEpoch):
    model.fit([X_train[0], X_train[1], X_train[2]], Y_train, epochs = 1) 
    # test
    # acc = test_model(model, testList, 0.7)
    roc_score, acc_score = ROC_score(model, x_test, y_test)
    if (roc_score > roc_highest):
        # model.save_weights(join(models_dir,'Epoch_'+str(i+1)+'_ACC_'+str(acc)))
        model.save_weights(join(models_dir,'Epoch_'+str(i+1) + '_roc_' + str(roc_score)))
        roc_highest = roc_score

    if acc_score > acc_highest:
        model.save_weights(join(models_dir,'Epoch_'+str(i+1) + '_acc_' + str(acc_score)))
        acc_highest = acc_score
    print('epoch %s completed <---------------------' % (str(i)))
