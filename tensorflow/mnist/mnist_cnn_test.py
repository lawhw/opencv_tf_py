from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def imageprepare(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (28, 28))
    temp = "temp.jpg"
    cv2.imwrite(temp, img)
    cv2.imwrite(filename, img)
    im = Image.open(temp) #读取的图片所在路径，注意是28*28像素
    # plt.imshow(im)  #显示需要识别的图片
    # plt.show()
    im = im.convert('L')
    tv = list(im.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv]
    return tva

def init():
    x = tf.placeholder(tf.float32, [None, 784])

    y_ = tf.placeholder(tf.float32, [None, 10])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./tmp/tensorflow/mnist/model/mnist_cnn/model.ckpt")  # 使用模型，参数和之前的代码保持一致

    prediction = tf.argmax(y_conv, 1)
    return (prediction, sess, keep_prob, x)




def test(result,prediction,sess,keep_prob,x):
    predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)
    print('识别结果:', predint[0])



def main():
    prediction,sess,keep_prob,x = init()
    test(imageprepare("./tmp/tensorflow/mnist/test/3_2.jpg"),prediction,sess,keep_prob,x)
    test(imageprepare("./tmp/tensorflow/mnist/test/3_1.jpg"),prediction,sess,keep_prob,x)
    test(imageprepare("./tmp/tensorflow/mnist/test/2_1.png"),prediction,sess,keep_prob,x)
    test(imageprepare("./tmp/tensorflow/mnist/test/5_1.png"),prediction,sess,keep_prob,x)
    pass

if __name__ == '__main__':
    main()
