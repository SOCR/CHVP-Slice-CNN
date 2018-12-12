import numpy as np
from os import listdir
from os.path import isfile, join
import re
import librosa
import math
import logging
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
feats_dir = join(project_dir, "feats")

DIM = 3

def read_features(features_dir, num_sessions_each_dim):
    '''
    Cut each cube data into 3 sections w.r.t. x,y,z axis.
    
    inputs:
        features_dir: directory name to store all targeted data.
        num_sessions_each_dim: how many number of sections for each axis
    output:
        x: numpy array of (list of [[sections with x axis], [sections with y axis], [sections with z axis]])
    '''
    x = [[] for i in range(DIM)]
    
    filenames = [f for f in listdir(features_dir) if isfile(join(features_dir, f))]
    for filename in filenames:
        data = np.load(join(features_dir, filename))
        cut_interval = math.floor(data.shape[0] / num_sessions_each_dim)
        log.debug("cut_interval: %d", cut_interval)
     
        for i in range(DIM):
            sections = np.zeros((num_sessions_each_dim, data.shape[0], data.shape[0]))
            # Cut a section every cut_interval
            for j in range(0, num_sessions_each_dim):
                index = [slice(None), slice(None), slice(None)]
                index[i] = (j + 1) * cut_interval - 1
                sections[j, :, :] = np.array(data[tuple(index)])
            x[i].append(sections)
        log.debug("Section shape: " + ",".join([str(s) for s in sections.shape]))
    x = np.asarray(x)
    # Normalize the value from 0 - 255 to 0 - 1
    x = x / 255.0
    return x

def read_train_files(features_dir, num_sessions_each_dim):
    '''
    Give directory of train features return 
        (1) training feature: x_train
        (2) labels: y_train 
    
    inputs:
        features_dir: directory name to store training data.
        num_sessions_each_dim: how many number of sections for each axis
    output:
        x_train: numpy array of list of [[sections with x axis], [sections with y axis], [sections with z axis]]
        y_train: numpy array of list of labels. for each label, [1, 0] = 0; [0, 1] = 1
    '''
    x_train = read_features(features_dir, num_sessions_each_dim) 

    filenames = [f for f in listdir(features_dir) if isfile(join(features_dir, f))]

    y_train = []
    for filename in filenames:
        target = filename.split(".")[0]
        label = int(target.split("_")[2])
        y_train.append(np.array([0, 1]) if label else np.array([1, 0]))
    y_train = np.asarray(y_train)
    return x_train, y_train



def read_test_files(features_dir, num_sessions_each_dim):
    '''
    Give directory of test features return 
        (1) test feature: x_test 
    
    inputs:
        features_dir: directory name to store test data.
        num_sessions_each_dim: how many number of sections for each axis
    output:
        x_test: numnpy array of list of [[sections with x axis], [sections with y axis], [sections with z axis]]
    '''
    x_test = read_features(features_dir, num_sessions_each_dim) 

    return x_test 



if __name__ == '__main__':
    x_train, y_train = read_train_files(join(corpus_dir,'train_data'), 10)
    x_test = read_test_files(join(corpus_dir,'test_data'), 10)

    # print(x_train[0][0].shape)
    # print(y_train[0].shape)

 
