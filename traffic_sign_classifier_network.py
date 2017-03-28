import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import cv2
import pickle
import matplotlib.pyplot as plt

# Constants
EPOCHS = 1
BATCH_SIZE = 128
NORMALIZED_MEAN = 0
NORMALIZED_ST_DEV = 0.1
NUM_FULLY_CONNECTED_NODES = 120
NUM_FULLY_CONNECTED2_NODES = 84
LEARNING_RATE = 0.006


def pre_process(array):
    processed = []
    for i in range(0, len(array)):
        img = array[i]
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        normalized_grey_img = grey_img / 255. - 0.5
        processed.append(normalized_grey_img)
    return np.expand_dims(np.array(processed), 3)


def conv_relu_pool(conv_input, shape, strides, padding, ksize, pool_strides, pool_padding):
    depth = shape[3]

    # Convolutional + Maconv_input pool
    weights = tf.Variable(tf.truncated_normal(shape=shape, mean=NORMALIZED_MEAN, stddev=NORMALIZED_ST_DEV))
    biases = tf.Variable(tf.zeros(depth))
    conv_input = tf.nn.conv2d(conv_input, weights, strides=strides, padding=padding, name="conv2d") + biases
    conv_input = tf.nn.relu(conv_input)
    conv_input = tf.nn.max_pool(conv_input, ksize=ksize, strides=pool_strides, padding=pool_padding)
    return tf.contrib.layers.batch_norm(conv_input, center=True, scale=True, is_training=True)


def fully_connected(flat_input, num_input, num_output, activate=False):
    weights = tf.Variable(tf.truncated_normal(shape=(num_input, num_output), mean=NORMALIZED_MEAN, stddev=NORMALIZED_ST_DEV))
    bias = tf.Variable(tf.zeros(num_output))
    flat_input = tf.matmul(flat_input, weights) + bias
    if activate:
        flat_input = tf.nn.relu(flat_input)
    return flat_input


def TrafficSignClassifier(data, num_labels):
    initial_data_depth = data.get_shape().as_list()[3]
    data = conv_relu_pool(data, shape=(5, 5, initial_data_depth, 6), strides=[1, 1, 1, 1], padding='VALID', ksize=[1, 2, 2, 1], pool_strides=[1, 2, 2, 1], pool_padding='VALID')
    data = conv_relu_pool(data, shape=(5, 5, 6, 16), strides=[1, 1, 1, 1], padding='VALID', ksize=[1, 2, 2, 1], pool_strides=[1, 2, 2, 1], pool_padding='VALID')

    data = flatten(data)
    input_size = data.get_shape().as_list()[1]

    data = fully_connected(data, input_size, NUM_FULLY_CONNECTED_NODES, True)
    data = fully_connected(data, NUM_FULLY_CONNECTED_NODES, NUM_FULLY_CONNECTED2_NODES, True)
    data = fully_connected(data, NUM_FULLY_CONNECTED2_NODES, num_labels)
    return data


def evaluate(x_tensor, y_tensor, X_data, y_data, accuracy_op):
    num_eval = len(X_data)
    total_accuracy = 0.
    sess = tf.get_default_session()
    for oset in range(0, num_eval, BATCH_SIZE):
        b_x, b_y = X_data[oset:oset+BATCH_SIZE], y_data[oset:oset+BATCH_SIZE]
        accuracy = sess.run(accuracy_op, feed_dict={x_tensor: b_x, y_tensor: b_y})
        total_accuracy += (accuracy * len(b_x))
    return total_accuracy / num_eval

def outputFeatureMap(image_input, tf_activation, sess, x, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

def run_network(X_test, y_test, process_test=False, test_validation=False):
    # Open files
    training_file = 'train.p'
    validation_file = 'valid.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    # Set data sets and label sets to variables
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']

    # Collect some basic info about the data set and display it
    n_train = len(X_train)
    n_test = len(X_test)
    image_shape = X_train[0].shape
    n_classes = len(np.unique(y_train))
    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    X_train = pre_process(X_train)

    # Pre-process
    X_train, y_train = shuffle(X_train, y_train)
    X_valid = pre_process(X_valid)

    # Setup TensorFlow variables
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    # Calculate logits and setup optimizer
    logits = TrafficSignClassifier(x, n_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    training_operation = optimizer.minimize(loss_operation)

    # setup accuracy calculator and saver
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        validation_accuracy_sum = 0
        for i in range(EPOCHS):
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(x, y, X_valid, y_valid, accuracy_operation)
            validation_accuracy_sum += validation_accuracy

        print("Mean Validation Percentage = {:.3f}".format(validation_accuracy_sum / EPOCHS))
        print()
        saver.save(sess, './classifier')
        print("Model saved")

    if test_validation:
        with tf.Session() as sess:
            if process_test:
                X_test = pre_process(X_test)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('.'))

            test_accuracy = evaluate(x, y, X_test, y_test, accuracy_operation)
            print("Test Accuracy = {:.3f}".format(test_accuracy))

