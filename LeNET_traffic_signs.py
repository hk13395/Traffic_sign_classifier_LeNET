# importing the files required for training
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
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
x_valid_norm = preprocess(x_valid)

image_datagen = ImageDataGenerator(
    rotation_range=15.0, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1)

# leNET architecture model design

epochs = 30
batch_size = 128
batch_per_epoch = 5000


def LeNET(x, n_Classes):
    mu = 0
    sigma = 0.1

    # Convolutional Layer 1
    conv1_w = tf.Variable(tf.random.truncated_normal(
        shape=(3, 3, 1, 64), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.constant(0.1, shape=(64,)))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[
                         1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')

    drop1 = tf.nn.dropout(conv1_pool, rate=keep_prob)

    # Convolutional Layer 2
    conv2_w = tf.Variable(tf.random.truncated_normal(
        shape=(3, 3, 64, 128), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.constant(0.1, shape=(128,)))
    conv2 = tf.nn.conv2d(drop1, conv2_w, strides=[
                         1, 1, 1, 1], padding='SAME') + conv2_b
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')

    drop2 = tf.nn.dropout(conv2_pool, rate=keep_prob)

    # Fully Connected layer 1
    fc_0 = tf.concat([tf.compat.v1.layers.flatten(
        drop1), tf.compat.v1.layers.flatten(drop2)], 1)

    fc1_w = tf.Variable(tf.random.truncated_normal(
        shape=(fc_0.shape[1], 64), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.constant(0.1, shape=(64,)))
    fc1 = tf.matmul(fc_0, fc1_w) + fc1_b

    drop_fc1 = tf.nn.dropout(fc1, rate=keep_prob)

    # Fully Connected layer 2
    fc2_w = tf.Variable(tf.random.truncated_normal(
        shape=(drop_fc1.shape[1], n_classes), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.constant(0.1, shape=(n_classes,)))
    logits = tf.matmul(drop_fc1, fc2_w) + fc2_b

    return logits


# Palceholders
x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.compat.v1.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.compat.v1.placeholder(tf.float32)

# training pipeline
learn_rate = 0.001
logits = LeNET(x, n_classes)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate)
train_step = optimizer.minimize(loss_function)


# Model evaluation
correct_prediction = tf.equal(
    tf.argmax(logits, 1), tf.argmax(tf.cast(y, tf.int64)))
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

    for epoch in range(epochs):
        print("Epoch{}....".format(epoch+1))
        batch_counter = 0

        for batch_x, batch_y in image_datagen.flow(x_train_norm, y_train, batch_size=batch_size):
            batch_counter += 1
            sess.run(train_step, feed_dict={
                     x: batch_x, y: batch_y, keep_prob: 0.5})

            if batch_counter == batch_per_epoch:
                break

        train_accuracy = evaluate(x_train_norm, y_train)
        val_accuracy = evaluate(x_valid_norm, y_valid)
        print('Train accuracy = {:.3f} Validation accuracy = {:.3f}'.format(
            train_accuracy, val_accuracy))

        checkpoint.save(
            sess, save_path='../checkpoint/traffic_sign_model.ckpt', global_step=epoch)

# Testing the model
