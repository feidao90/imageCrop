import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.image as img
import matplotlib.pyplot as plt

sess = tf.Session()

diag = tf.diag([1,1,1,1])
truncated = tf.truncated_normal([2,3])
fill = tf.fill([2,3],5.0)
uniform = tf.random_uniform([3,2])
convert_tensor = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
truncatedTwo = tf.truncated_normal([3,4],mean=0.0,stddev=1.0)
input_data = tf.constant([[1.,2.,3.],[1.,5.,3.],[1.,2.,7.],[6.,2.,3.],[8.,2.,3.]])
shuffle = tf.random_shuffle(input_data)
crop = tf.random_crop(input_data,[1,1])

# 指定尺寸,图片随机裁剪
image = img.imread('./resources/test.jpg')
plt.imshow(image)
plt.show()
reshaped_image = tf.cast(image,tf.float32)
size = tf.cast(tf.shape(reshaped_image),tf.int32)
height = sess.run(size[0]//2)
width = sess.run(size[1]//2)
distorted_image = tf.random_crop(reshaped_image,[height,width,3])
plt.imshow(sess.run(tf.cast(distorted_image,tf.uint8)))
plt.show()

for i in range(9):
    a = tf.random_crop(reshaped_image,[height,width,3])
    plt.imshow(sess.run(tf.cast(a, tf.uint8)))
    plt.show()


print("test")