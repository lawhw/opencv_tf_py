#参考Tensorflow技术解析与实战代码
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('./../../datas/mnist/', one_hot=True)

learning_rate = 0.01 # 学习率
training_epochs = 20 # 训练的轮数
batch_size = 256 # 每次训练的数据多少
display_step = 1 # 每隔多少轮显示一次训练结

examples_to_show = 10

#网络参数
n_hidden_1 = 256 # 第一个隐藏层神经元个数，也是特征值个数
n_hidden_2 = 128 # 第二个隐藏层神经元个数，也是特征值个数
n_input = 784 # 输入数据的特征值个数：28×

X= tf.placeholder("float", [None, n_input])

weights = {
  'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
  'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
  'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
  'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
  'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
  'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
  'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
  'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# 定义压缩函数
def encoder(x):
  # Encoder Hidden layer with sigmoid activation #1
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                         biases['encoder_b1']))
  # Decoder Hidden layer with sigmoid activation #2
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                         biases['encoder_b2']))
  return layer_2

# 定义解压函数
def decoder(x):
  # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                         biases['decoder_b1']))
  # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                         biases['decoder_b2']))
    return layer_2
# 构建模型
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# 得出预测值
y_pred = decoder_op
# 得出真实值，即输入值
y_true = X

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  total_batch = int(mnist.train.num_examples/batch_size)
  # 开始训练
  for epoch in range(training_epochs):

    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      # Run optimization op (backprop) and cost op (to get loss value)
      _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # 每一轮，打印出一次损失值
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

  print("Optimization Finished!")

  # 对测试集应用训练好的自动编码网络
  encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
  # 比较测试集原始图片和自动编码网络的重建结果
  f, a = plt.subplots(2, 10, figsize=(10, 2))
  for i in range(examples_to_show):
      a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  # 测试集
      a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  # 重建结果
  f.show()
  plt.draw()
  plt.show()







