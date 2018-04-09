#TensorFlow简单应用－SoftMax函数手写数字识别
#1.导入相应包
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#2.读入数据
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True) #读入数据,one_hot向量只有一个1，其余全部为0
n = 100
#3.查看数据信息
# print(mnist.train.images.shape,mnist.train.images[1:100,1:10])
x_data = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#注意，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
#b的形状是[10]，所以我们可以直接把它加到输出上面。
#===下面这行代码实现函数映射y=softmax(W*x+b)==========
y = tf.nn.softmax(tf.matmul(x_data,W)+b)

#===下面2行代码定义损失函数:交叉熵（cross-entropy）========
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1])) #计算交叉熵
optimizer = tf.train.GradientDescentOptimizer(0.5)
train_step = optimizer.minimize(cross_entropy)

#===开始训练,每次随机抽100个样本作为输入=====
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for i in range(3000):
    x_batch,y_batch = mnist.train.next_batch(n)
    sess.run(train_step,feed_dict={x_data:x_batch,y_:y_batch})
    if i % 50 ==0:
        x_batch, y_batch = mnist.test.next_batch(n)
        preCorrect = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        pre = sess.run(preCorrect,feed_dict={x_data:x_batch,y_:y_batch})
        print(sum(pre)/n)
# 首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
#===统计准确率======
x_batch, y_batch = mnist.test.next_batch(1000)
pre = sess.run(preCorrect,feed_dict={x_data:x_batch,y_:y_batch})
print('pre: ',sum(pre)/n)
sess.close()
