import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector




def variable_summarise(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)#平均值
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev", stddev)#标准差
        tf.summary.scalar("max", tf.reduce_max(var))#最大值
        tf.summary.scalar("min", tf.reduce_min(var))#最小值
        tf.summary.histogram("histogram",mean)#直方图

def mnist_tensorboard():
    #载入数据
    mnist = input_data.read_data_sets("./../../datas/mnist/",one_hot=True)




    #每个批次的大小
    batch_size = 100
    #计算一共有多少个批次
    n_batch = mnist.train.num_examples // batch_size

    # 定义命名空间
    with tf.name_scope("input"):
        #定义两个
        x = tf.placeholder(tf.float32,[None,784],name="x-input")
        y = tf.placeholder(tf.float32,[None,10],name="y-input")

    #创建神经网络
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.zeros([784,10]),name="W")
            variable_summarise(W)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([10]),name="b")
            variable_summarise(b)
        with tf.name_scope("wx_plus_b"):
            wx_plus_b = tf.matmul(x,W)+b
        with tf.name_scope("softmax"):
            prediction = tf.nn.softmax(wx_plus_b)

    # 二次代价函数
    # loss = tf.reduce_mean(tf.square(y - prediction))
    #交叉熵
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
        tf.summary.scalar("loss", loss)
    # 梯度下降训练函数
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    # 初始化变量
    init = tf.global_variables_initializer()

    #定义准确率
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大值所在的位置
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar("accuracy", accuracy)


    #合并所有的summary
    merged = tf.summary.merge_all()

    # 开始训练
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./../../datas/test/",sess.graph)
        for epoch in range(11):
            for batch in range(n_batch):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
            writer.add_summary(summary,epoch)
            acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print(epoch,acc)

    pass

def tensorboard():
    # 载入数据集
    mnist = input_data.read_data_sets("./../../datas/mnist/",one_hot=True)
    # 运行次数
    max_steps = 3001
    # 图片数量
    image_num = 3000
    # 文件路径
    DIR = "./../../datas/logs/t5/"

    # 定义会话
    sess = tf.Session()

    # 载入图片
    embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

    # 参数概要
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)  # 平均值
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)  # 标准差
            tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
            tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
            tf.summary.histogram('histogram', var)  # 直方图

    # 命名空间
    with tf.name_scope('input'):
        # 这里的none表示第一个维度可以是任意的长度
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        # 正确的标签
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 显示图片
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('layer'):
        # 创建一个简单神经网络
        with tf.name_scope('weights'):
            W = tf.Variable(tf.zeros([784, 10]), name='W')
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([10]), name='b')
            variable_summaries(b)
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(x, W) + b
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(wx_plus_b)

    with tf.name_scope('loss'):
        # 交叉熵代价函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        # 使用梯度下降法
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 结果存放在一个布尔型列表中
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
        with tf.name_scope('accuracy'):
            # 求准确率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型
            tf.summary.scalar('accuracy', accuracy)

    # 产生metadata文件
    if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
        tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
    with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
        labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
        for i in range(image_num):
            f.write(str(labels[i]) + '\n')

            # 合并所有的summary
    merged = tf.summary.merge_all()

    projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = 'metadata.tsv'
    embed.sprite.image_path = 'mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    for i in range(max_steps):
        # 每个批次100个样本
        batch_xs, batch_ys = mnist.train.next_batch(100)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                              run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        projector_writer.add_summary(summary, i)

        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(acc))

    saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
    projector_writer.close()
    sess.close()

def tensorboard_op():
    # 载入数据集
    mnist = input_data.read_data_sets("./../../datas/mnist/",one_hot=True)
    # 运行次数
    max_steps = 8001
    # 图片数量
    image_num = 3000
    # 文件路径
    DIR = "./../../datas/logs/t5/"

    # 定义会话
    sess = tf.Session()

    # 载入图片
    embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

    # 参数概要
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)  # 平均值
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)  # 标准差
            tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
            tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
            tf.summary.histogram('histogram', var)  # 直方图

    # 命名空间
    with tf.name_scope('input'):
        # 这里的none表示第一个维度可以是任意的长度
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        # 正确的标签
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    keep_prob = tf.placeholder(tf.float32)
    lr = tf.Variable(0.001, dtype=tf.float32)

    # 显示图片
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('layer1'):
        # 创建一个简单神经网络
        with tf.name_scope('weights1'):
            W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1), name='W1')
            variable_summaries(W1)
        with tf.name_scope('biases1'):
            b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
            variable_summaries(b1)
        with tf.name_scope('wx_plus_b1'):
            wx_plus_b1 = tf.matmul(x, W1) + b1
        with tf.name_scope('tanh1'):
            L1 = tf.nn.tanh(wx_plus_b1)
        with tf.name_scope('dropout1'):
            L1_drop = tf.nn.dropout(L1, keep_prob)

    with tf.name_scope('layer2'):
        # 创建一个简单神经网络
        with tf.name_scope('weights2'):
            W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='W2')
            variable_summaries(W2)
        with tf.name_scope('biases2'):
            b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
            variable_summaries(b2)
        with tf.name_scope('wx_plus_b2'):
            wx_plus_b2 = tf.matmul(L1_drop, W2) + b2
        with tf.name_scope('tanh2'):
            L2 = tf.nn.tanh(wx_plus_b2)
        with tf.name_scope('dropout2'):
            L2_drop = tf.nn.dropout(L2, keep_prob)

    with tf.name_scope('layer3'):
        # 创建一个简单神经网络
        with tf.name_scope('weights3'):
            W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name='W3')
            variable_summaries(W3)
        with tf.name_scope('biases3'):
            b3 = tf.Variable(tf.zeros([10]) + 0.1, name='b3')
            variable_summaries(b3)
        with tf.name_scope('wx_plus_b3'):
            wx_plus_b3 = tf.matmul(L2_drop, W3) + b3
        with tf.name_scope('softmax'):
            prediction = tf.nn.softmax(wx_plus_b3)



    with tf.name_scope('loss'):
        # 交叉熵代价函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        # 使用梯度下降法
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
        #train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 结果存放在一个布尔型列表中
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
        with tf.name_scope('accuracy'):
            # 求准确率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型
            tf.summary.scalar('accuracy', accuracy)

    # 产生metadata文件
    if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
        tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
    with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
        labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
        for i in range(image_num):
            f.write(str(labels[i]) + '\n')

            # 合并所有的summary
    merged = tf.summary.merge_all()

    projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = 'metadata.tsv'
    embed.sprite.image_path = 'mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)
    count = 0
    for i in range(max_steps):
        # 每个批次100个样本
        # if i % 20 == 0 and count < 50:
        #     count += 1
        #     sess.run(tf.assign(lr, 0.001 * (0.95 ** i)))
        batch_xs, batch_ys = mnist.train.next_batch(100)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys,keep_prob:1.0}, options=run_options,
                              run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        projector_writer.add_summary(summary, i)

        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(acc))

    saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
    projector_writer.close()
    sess.close()


if __name__ == '__main__':
    tensorboard_op()
    pass