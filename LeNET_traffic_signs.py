# importing the files required for training
import matplotlib.pyplot as plt
import numpy as np
import pickle

training_file = '../Traffic_sign_classifier_LeNET/training_data/train.p'
test_file = '../Traffic_sign_classifier_LeNET/training_data/test.p'
validation_file = '../Traffic_sign_classifier_LeNET/training_data/valid.p'

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

n_classes = np.unique(y_train).shape[0]

print("Number of training examples", n_train)
print("Number of testing examples", n_test)
print("Number of validation examples", n_validation)
print("The shape of the image is ", image_shape)
print("Number of classes are ", n_classes)


rows, cols = 4, 12
fig, ax_array = plt.subplots(rows, cols)
plt.suptitle('Random samples from the given training data set')
for class_idx, ax in enumerate(ax_array.ravel()):
    if class_idx < n_classes:
        cur_x = x_train[y_train == class_idx]
        cur_img = cur_x[np.random.randint(len(cur_x))]
        ax.imshow(cur_img)
        ax.set_title('{:02d}'.format(class_idx))
    else:
        ax.axis('off')

plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.show()


plt.hist(y_train, bins=n_classes, alpha=1.0, label='Training data')
plt.hist(y_test, bins=n_classes, alpha=1.0, label='Testing data')
plt.hist(y_valid, bins=n_classes, alpha=0.6, label='Validation data')
plt.legend(loc='upper center')
plt.show()
