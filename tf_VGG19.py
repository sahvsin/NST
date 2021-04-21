# -*- coding: utf-8 -*-
"""
@author: Sahil

Credits to Andrew Ng and the MatConvNet Team (module modeled after Deep Neural Network Specialization assignment)
"""

import numpy as np
import scipy.io
import semver 
from tensorflow import __version__ as tf_ver



#check whether using tensorflow 1 or 2 and determine which functions to use 
tf_version = semver.VersionInfo.parse(tf_ver)
if tf_version.major == 1:
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf




def load_VGG19_model(weights_path, img_dims, num_ch=3):
    
    """
    An average pooling variant of the pretrained VGG19 model (without the top FC layers) recreated in Tensorflow 
    Weights are saved in a .mat file courtesy of Andrew Ng and the MatConvNet Team
    
    Inputs:
    weights_path -- filepath of the pretrained weights
    imh_dims -- list of (height and width, in pixels) dimensions of the image
    num_channels -- number of color channels (0 - B/W, 3 - RGB)
    
    Outputs:
    graph-- dictionary of the VGG19 model (without FC layers), maps layer names to associating layer operation
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """
    
    vgg = scipy.io.loadmat(weights_path)
    vgg_layers = vgg['layers']
    
    
    def tf_compute_params(layer_num):
    
        """
        gets the parameters (weights and bias) of the given layer (which were loaded from the .mat file)
        
        Inputs:
        layer_num -- index of the layer to process (following format of the .mat file)
        
        Outputs:
        weights -- array of weights for the current layer
        bias -- bias vector for the current layer 
        """
    
        params = vgg_layers[0][layer_num][0][0][2]
        weights = params[0][0]
        bias = params[0][1]
        
        return weights, bias


    def tf_conv2d(prev_layer, layer_num, layer_name):
    
        """
        computes the 2D-convolution of the given layer using Same padding and 1x1 strides
        
        Inputs:
        prev_layer -- the previous layer's activation 
        layer_num -- the index of the current layer
        layer_name -- the name of the current layer
        """
        
        # get the parameters for the current layer
        weights, bias = tf_compute_params(layer_num)
        weights = tf.constant(weights)
        bias = tf.constant(np.reshape(bias, (bias.size)))
        
        # use Tensorflow's conv2d function to compute the 2d-convolution for the current layer
        return tf.nn.conv2d(prev_layer, filter=weights, strides=[1,1,1,1], padding='SAME') + bias
        
        
    graph = dict()
    img_height, img_width = img_dims
    
    graph['input'] = tf.Variable(np.zeros((1, img_height, img_width, num_ch)), dtype='float32')
    graph['block1_conv1'] = tf.nn.relu(tf_conv2d(graph['input'], 0, 'block1_conv1'))
    graph['block1_conv2'] = tf.nn.relu(tf_conv2d(graph['block1_conv1'], 2, 'block1_conv2'))
    graph['block1_pool'] = tf.nn.avg_pool2d(graph['block1_conv2'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    graph['block2_conv1'] = tf.nn.relu(tf_conv2d(graph['block1_pool'], 5, 'block2_conv1'))
    graph['block2_conv2'] = tf.nn.relu(tf_conv2d(graph['block2_conv1'], 7, 'block2_conv2'))
    graph['block2_pool'] = tf.nn.avg_pool2d(graph['block2_conv2'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    graph['block3_conv1'] = tf.nn.relu(tf_conv2d(graph['block2_pool'], 10, 'block3_conv1'))
    graph['block3_conv2'] = tf.nn.relu(tf_conv2d(graph['block3_conv1'], 12, 'block3_conv2'))
    graph['block3_conv3'] = tf.nn.relu(tf_conv2d(graph['block3_conv2'], 14, 'block3_conv3'))
    graph['block3_conv4'] = tf.nn.relu(tf_conv2d(graph['block3_conv3'], 16, 'block3_conv4'))
    graph['block3_pool'] = tf.nn.avg_pool2d(graph['block3_conv4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    graph['block4_conv1'] = tf.nn.relu(tf_conv2d(graph['block3_pool'], 19, 'block4_conv1'))
    graph['block4_conv2'] = tf.nn.relu(tf_conv2d(graph['block4_conv1'], 21, 'block4_conv2'))
    graph['block4_conv3'] = tf.nn.relu(tf_conv2d(graph['block4_conv2'], 23, 'block4_conv3'))
    graph['block4_conv4'] = tf.nn.relu(tf_conv2d(graph['block4_conv3'], 25, 'block4_conv4'))
    graph['block4_pool'] = tf.nn.avg_pool2d(graph['block4_conv4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    graph['block5_conv1'] = tf.nn.relu(tf_conv2d(graph['block4_pool'], 28, 'block5_conv1'))
    graph['block5_conv2'] = tf.nn.relu(tf_conv2d(graph['block5_conv1'], 30, 'block5_conv2'))
    graph['block5_conv3'] = tf.nn.relu(tf_conv2d(graph['block5_conv2'], 32, 'block5_conv3'))
    graph['block5_conv4'] = tf.nn.relu(tf_conv2d(graph['block5_conv3'], 34, 'block5_conv4'))
    graph['block5_pool'] = tf.nn.avg_pool2d(graph['block5_conv4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    return graph
    
    
    