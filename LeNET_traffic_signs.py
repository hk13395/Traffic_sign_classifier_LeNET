# importing the files required for training
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops.gen_nn_ops import bias_add
from tensorflow.python.ops.gen_random_ops import truncated_normal
from tensorflow.python.ops.gen_state_ops import variable
tf.compat.v1.disable_eager_execution()

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


# pre processing of the data
def preprocess(X, equalize_his=True):
    X = np.array([np.expand_dims(cv2.cvtColor(
        rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    if equalize_his:
        X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2)
                     for img in X])

    X = np.float32(X)

    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo(np.float32).eps)

    return X


x_train_norm = preprocess(x_train)
x_test_norm = preprocess(x_test)


x_train_norm, x_valid_norm, y_train, y_valid = train_test_split(
    x_train_norm, y_train, test_size=0.2, random_state=42)

image_datagen = ImageDataGenerator(
    rotation_range=15.0, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1)

# leNET architecture model design
epochs = 30
batch_size = 128


def LeNET(x, n_Classes):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1
    conv1_w = tf.Variable(tf.random.truncated_normal(
        shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[
                         1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, conv1_b)
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='VALID')

    # Convolutional Layer 2
    conv2_w = tf.Variable(tf.random.truncated_normal(
        shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1_pool, conv2_w, strides=[
                         1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, conv2_b)
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='VALID')

    # Convolutionla Layer 3
    conv3_w = tf.Variable(tf.random.truncated_normal(
        shape=(5, 5, 16, 400), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(400))
    conv3 = tf.nn.conv2d(conv2_pool, conv3_w, strides=[
                         1, 1, 1, 1], padding='VALID')
    conv3 = tf.nn.bias_add(conv3, conv3_b)
    conv3 = tf.nn.relu(conv3)

    # Fully Connected layer 1
    fc_0 = tf.concat([tf.compat.v1.layers.flatten(
        conv2_pool), tf.compat.v1.layers.flatten(conv3)], 1)

    fc_0 = tf.nn.dropout(fc_0, rate=keep_prob)

    fc1_w = tf.Variable(tf.random.truncated_normal(
        shape=(fc_0.shape[1], n_classes), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(n_classes))
    fc1 = tf.matmul(fc_0, fc1_w) + fc1_b
    logits = fc1
    """
    fc1 = tf.nn.relu(fc1)

    # Fully Connected layer 2
    fc2_w = tf.Variable(tf.random.truncated_normal(
        shape=(fc1.shape[1], 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)


    # Fully connected layer 3
    fc3_w = tf.Variable(tf.random.truncated_normal(shape=(fc2.shape[1], 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2,fc3_w) + fc3_b
    """
    return logits


# Palceholders
tf.compat.v1.reset_default_graph()
x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.compat.v1.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.compat.v1.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

# training pipeline
learn_rate = 0.001
logits = LeNET(x, n_classes)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, one_hot_y)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate)
train_step = optimizer.minimize(loss_function)


# Model evaluation
correct_prediction = tf.equal(
    tf.argmax(logits, 1), tf.argmax(one_hot_y))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(x_data, y_data):
    num_examples = x_data.shape[0]
    total_accuracy = 0
    sess = tf.compat.v1.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = x_data[offset: offset +
                                  batch_size], y_data[offset: offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={
                            x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += accuracy*len(batch_x)

    return total_accuracy/num_examples


checkpoint = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    num_examples = len(x_train)

    for i in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)

        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x_train[offset: end], y_train[offset: end]
            sess.run(train_step, feed_dict={
                     x: batch_x, y: batch_y, keep_prob: 0.5})

        train_accuracy = evaluate(x_train_norm, y_train)
        val_accuracy = evaluate(x_valid_norm, y_valid)
        print("EPOCH {}...".format(i+1))
        print('Train accuracy = {:.3f} Validation accuracy = {:.3f}'.format(
            train_accuracy, val_accuracy))

        checkpoint.save(sess, 'LeNET')
        print("Model saved")


# Testing the model
