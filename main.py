# -*- coding: utf-8 -*-
"""
@author: Sahil
"""


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import numpy as np
import cv2
import semver 
from tensorflow import __version__ as tf_ver

from config import *
from image_utils import load_images, img_reshape_and_normalize, create_noise_image, show_image, save_image
from tf_VGG19 import load_VGG19_model
from costs import compute_content_cost, compute_style_cost, compute_total_cost, get_cost
from tf_graph import tf_graph


# TensorFlow version check
#check whether using tensorflow 1 or 2 and determine which functions to use 
tf_version = semver.VersionInfo.parse(tf_ver)
if tf_version.major == 1:
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()




# MACROS
current_dir = os.getcwd()
content_dir = os.path.join(current_dir, 'content')
style_dir = os.path.join(current_dir, 'styles')
out_dir = os.path.join(current_dir, 'generated_images')

content_path = os.path.join(content_dir, content_filename)
style_path = os.path.join(style_dir, style_filename)
out_path = os.path.join(out_dir, out_filename)

mean_offset = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

# to check if I am using GPU acceleration in tensorflow
#conf = tf.ConfigProto(log_device_placement=True)
#conf.gpu_options.allow_growth = True




# MAIN
tf.reset_default_graph()
sess = tf.InteractiveSession() #config=conf to check if using GPU acceleration

content_orig, style_orig = load_images(content_path, style_path, image_height, image_width)
content, style = img_reshape_and_normalize(content_orig, style_orig)
noisy_img = create_noise_image(content, style, noise_ratio, image_height, image_width, num_channels, option)

print('\n\n\nLoading VGG19 model...\n\n\n')
model = load_VGG19_model(pretrained_weights_path, [image_height, image_width], num_channels)

print('Computing cost...\n')
content_cost, style_cost, total_cost = get_cost(sess, model, content, style, 
                                                content_layers, style_layers, 
                                                content_coeffs, style_coeffs, 
                                                alpha, beta, func)

if tv_reg is True:
    total_cost += tf.reduce_sum(tf.image.total_variation(tf.convert_to_tensor(noisy_img, np.float32)))


print('\n\n\nGenerating the image...\n\n\n')
genImg = tf_graph(noisy_img, total_cost, sess, model, learn_rate, 
                  adam_beta1, adam_beta2, adam_epsilon, num_epochs, 
                  mean_offset, out_dir, print_rate)

if show_img == True:
    show_image(genImg)


# finally close the tensorflow session
sess.close()


print('Saving Image\n')
# save/write image to output destination
save_image(genImg, out_path)

print('Done\n')






