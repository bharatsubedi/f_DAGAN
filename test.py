import os
from os.path import join
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from src.utils import config
from src.nets import nn
directory='o'
if not os.path.exists(directory):
    os.makedirs(directory)

if __name__ == '__main__':
    weight_init = tf.initializers.orthogonal()
    model = nn.Generator(weight_init)
    model.load_weights(join('weights','g_model.tf'))
    print('model load successfully')
    for i in range(6000):
        noise = tf.random.truncated_normal([1, config.num_cont_noise])
        output_img = model(noise)[0]
        output_img = np.asarray(output_img)
        output_img = output_img.reshape(28,28,3)
        #print(output_img.shape)
        print('saving images.............')
        cv2.imwrite(directory+'/'+'{}.jpg'.format(i), output_img*255)


