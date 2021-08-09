# importing the files required for training
import numpy as np
import pickle

training_file = '../Classifier_project/train.p'
test_file = '../Classifier_project/test.p'
validation_file = '../Classifier_project/valid.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(test_file, mode='rb') as f:
    test = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_test, y_test = test['features'], test['labels']
x_valid, y_valid = valid['features'], valid['labels']

# Dataset summary and exploration
n_train = x_train.shape[0]
n_validation = x_valid.shape[0]
n_test = x_test.shape[0]
image_shape = x_train[0].shape

# n_classes = np.
