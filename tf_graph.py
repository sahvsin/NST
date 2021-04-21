# -*- coding: utf-8 -*-
"""
@author: Sahil
"""


import os
import numpy as np
import semver 
from tensorflow import __version__ as tf_ver
import cv2
from image_utils import save_image



#check whether using tensorflow 1 or 2 and determine which functions to use 
tf_version = semver.VersionInfo.parse(tf_ver)
if tf_version.major == 1:
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf



def tf_graph(noisy_img, total_cost, sess, model, learn_rate, beta1, beta2, epsilon, num_epochs, offset, out_path, print_rate):
    
    """
    initializes the variables for the Tensorflow graph and processes/updates the generated image as per the (updating) cost
    
    
    Inputs:
    noisy_img -- the initialized generated image to process through the network (1, image height, image width, number of channels)
    total_cost -- the initial computed total cost (style + content) to minimize
    sess -- the current tensorflow session (unecessary when using an interactive session)
    model -- the dictionary used as the VGG19 model
    learn_rate -- the learning rate used for the optimizer
    beta1 -- Adam Optimization hyperparameter
    beta2 -- Adam Optimization hyperparameter
    epsilon -- Adam Optimization hyperparameter
    num_epochs -- number of times the entire image is processed (forward and backwards) through he neural network
    offset -- RGB mean offset array
    out_path -- path to save output image
    print_rate -- rate at which to save intermediate images during the training
    
    Output:
    genImg -- final output (generated image) after every epoch of processing (1, image height, image width, number of channels)
    """
    
    # define the tensorflow optimizer (using Adam Optimizaer)
    optimizer = tf.train.AdamOptimizer(learn_rate, beta1, beta2, epsilon).minimize(total_cost)
    
    # initialize tensorflow variables 
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # set the initial generated image as the current input to the VGG19 model, then run the it through the model in the session 
    sess.run(model['input'].assign(noisy_img))
    
    for i in range(num_epochs):
        # run the session on the optimizer to minimize the total cost for the current epoch
        sess.run(optimizer)
        
        # compute the current epoch's output (generated image) by running the session through the model
        genImg = sess.run(model['input'])
        
        if print_rate != 0:
            if i%print_rate == 0:
            
                j = sess.run(total_cost)
                print("cost at epoch[{}]: {} ".format(str(i), j))
            
                # intermediately save the image as it updates (monitor the stylize updates)
                output_path = os.path.join(out_path, os.path.join('iters', str(i) + '.jpg'))
                save_image(genImg, output_path)
    
    return genImg
    
    
