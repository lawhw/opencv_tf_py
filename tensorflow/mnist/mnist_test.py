# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import cv2
import mnist
import argparse



def readData(filename):
    images = np.empty((1, 28 * 28), dtype=np.uint8)
    # img = cv2.imread(filename)
    # img = cv2.resize(img, (28, 28))
    # img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    # images = np.empty((1, 28*28), dtype=np.uint8)
    # arr = np.asarray(img, dtype=np.uint8)
    # images[0, :] = arr.flatten()
    label = np.empty((1,), dtype='uint8')
    label[0] = 0

    img = cv2.imread(filename)
    img = cv2.resize(img, (28, 28))
    temp = "temp.jpg"
    cv2.imwrite(temp, img)
    im = Image.open(temp)  # 读取的图片所在路径，注意是28*28像素

    im = im.convert('L')
    tv = list(im.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    images[0, :] = tva
    return (label,images)

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):

    feed_dict = {
        images_pl: data_set[1],
        labels_pl: data_set[0],
    }
    return feed_dict

def test(data_set):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)
        logits = mnist.inference(images_placeholder,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "./tmp/tensorflow/mnist/logs/fully_connected_feed/model.ckpt")
            prediction = tf.argmax(logits, 1)
            feed_dict = fill_feed_dict(data_set,
                                       images_placeholder,
                                       labels_placeholder)
            predint = prediction.eval(feed_dict=feed_dict, session=sess)

            print('识别结果:',predint[0])

def main():
   # print(test(readData("./tmp/tensorflow/mnist/test/3_1.jpg")))
   # print(test(readData("./tmp/tensorflow/mnist/test/2_1.png")))
   print(test(readData("./tmp/tensorflow/mnist/test/3_2.jpg")))
   print(test(readData("./tmp/tensorflow/mnist/test/5_1.png")))
   pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='./tmp/tensorflow/mnist/input_data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./tmp/tensorflow/mnist/logs/fully_connected_feed',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
