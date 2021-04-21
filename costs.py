# -*- coding: utf-8 -*-
"""
@author: Sahil
"""


import numpy as np
import semver 
from tensorflow import __version__ as tf_ver 


#check whether using tensorflow 1 or 2 and determine which functions to use 
tf_version = semver.VersionInfo.parse(tf_ver)
if tf_version.major == 1:
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()




def unroll_tensors(tensor, img_h, img_w, num_c):
    
    """
    reshapes layer array/tensor to compute content and style costs
    
    Inputs:
    tensor -- layer array/tensor to modify
    img_h -- height of the layer array/tensor
    img_w -- width of the layer array/tensor
    num_c -- number of channels layer 
    """
    
    return tf.transpose(tf.reshape(tensor, [img_h*img_w, num_c]))




def compute_layer_content_cost(content_tensor, genImg_tensor, fn):
    
    """
    compute the content cost (regression) between the content image and (to be) generated image for ONLY the given layer
    
    Inputs:
    content_tensor -- layer activation for the content of the content image (1, height, width, number of channels)
    genImg_tensor -- layer activation for the content of the generated image (1, height, width, number of channels)
    fn -- cost function (either L1 or L2)
    """
    
    # get the static dimensions of the generated image tensor (number of examples is negligible/ignored)
    _, height, width, num_filter = genImg_tensor.get_shape().as_list()
    
    # unroll the content and generated image tensors into 2D arrays
    unrolled_content = unroll_tensors(content_tensor, height, width, num_filter)
    unrolled_genImg = unroll_tensors(genImg_tensor, height, width, num_filter)
    
    if(fn == "L1"):
        return tf.reduce_sum(tf.math.abs(tf.math.subtract(unrolled_content, unrolled_genImg)))* 1/(4*height*width*num_filter)
    else:
        return tf.reduce_sum(tf.math.squared_difference(unrolled_content, unrolled_genImg))* 1/(4*height*width*num_filter)



def compute_content_cost(vgg_model, content_layers, content_coeff, sess, fn):
    
    """
    compute the content cost (regression) between the content image and (to be) generated image for ALL of the given layers
    
    Inputs:
    vgg_model -- the dictionary used as the VGG19 model
    content_layers -- list of the names of the chosen content layer 
    content_coeff -- coefficients used to compute the content cost for its respective layer
    sess -- the current Tensorflow session (not necessary if using interactive session)
    fn -- cost function (either L1 or L2)
    
    Outputs:
    content_cost -- the accumulated content cost of every content layer cost 
    """
    
    # initialize/reset the style cost accumulator
    content_cost = 0
    for l in content_layers:
        # save the output of the current elected content layer
        layer_output = vgg_model[l]
        
        # run the session on the output of the elected content layer and save the activation of that layer
        content_tensor = sess.run(layer_output)
        
        # save this layer's activation (placeholder for now, to be updated when processing the generated image through the model)
        genImg_tensor = layer_output
        
        # compute this layer's style cost and add it (weighted by the coefficient) to the accumulator
        content_cost += content_coeff * compute_layer_content_cost(content_tensor, genImg_tensor, fn)    
    
    return content_cost





def compute_gram_matrix(X):
    
    """
    computes the Gram Matrix of the input: X (number of channels, height x width)
    """
    
    return tf.matmul(X, X, transpose_b = True)




def compute_layer_style_cost(style_tensor, genImg_tensor, fn):
    
    """
    compute the style cost (regression) between the style image and (to be) generated image for ONLY the given layer

    Inputs: 
    style_tensor -- layer activation for the style of the style image (1, height, width, number of channels)
    genImg_tensor -- layer activation for the style of the generated image (1, height, width, number of channels)
    fn -- cost function (either L1 or L2)
    """
    
    # get the static dimensions of the generated image tensor 
    _, height, width, num_filter = genImg_tensor.get_shape().as_list()
    
    # unroll the content and generated image tensors into 2D matrices (necessary for Gram Matrix computation)
    unrolled_style = unroll_tensors(style_tensor, height, width, num_filter)
    unrolled_genImg = unroll_tensors(genImg_tensor, height, width, num_filter)
    
    # compute the Gram Matrices of the unrolled matrices
    gram_style = compute_gram_matrix(unrolled_style)
    gram_generated = compute_gram_matrix(unrolled_genImg)

    if(fn == "L1"):
        return tf.reduce_sum(tf.math.abs(tf.math.subtract(gram_style, gram_generated))) * 1/(4*((height*width)**2)*(num_filter**2))
    else:
        return tf.reduce_sum(tf.math.squared_difference(gram_style, gram_generated)) * 1/(4*((height*width)**2)*(num_filter**2))



def compute_style_cost(vgg_model, style_layers, style_coeff, sess, fn):
    
    """
    compute the style cost (regression) between the style image and (to be) generated image for ALL of the given layers
    
    Inputs:
    vgg_model -- the dictionary used as the VGG19 model
    style_layers -- list of the names of the chosen style layer 
    style_coeff -- coefficients used to compute the style cost for its respective layer
    sess -- the current Tensorflow session (not necessary if using interactive session)
    fn -- cost function (either L1 or L2)
    
    Outputs:
    style_cost -- the accumulated style cost of every style layer cost 
    """
    
    # initialize/reset the style cost accumulator
    style_cost = 0
    for l in style_layers:
        # save the output of the current elected style layer
        layer_output = vgg_model[l]
        
        # run the session on the output of the elected style layer and save the activation of that layer
        style_tensor = sess.run(layer_output)
        
        # save this layer's activation (placeholder for now, to be updated when processing the generated image through the model)
        genImg_tensor = layer_output
        
        # compute this layer's style cost and add it (weighted by the coefficient) to the accumulator
        style_cost += style_coeff * compute_layer_style_cost(style_tensor, genImg_tensor, fn)    
    
    return style_cost
    




def compute_total_cost(content_cost, style_cost, alpha, beta):
    
    """
    computes the combined weighted style and content costs
    
    Inputs:
    content_cost -- the conetent cost
    style_cost -- the style cost
    alpha -- hyperparameter for the weight for the content cost (higher relative to beta -> more influence from content layer)
    beta -- hyperparameter for the weight for the style cost (higher relative to alpha -> more influence from style layers)
    """
    
    return alpha*content_cost + beta*style_cost




def get_cost(sess, model, content, style, content_layers, style_layers, content_coeffs, style_coeffs, alpha, beta, fn="L2"):
    
    """
    computes and returns the three costs: content, style, and total
    
    Inputs:
    sess -- the current Tensorflow session (not necessary if using interactive session)
    model -- the dictionary used as the VGG19 model
    content -- normalized and reshaped content image array (assumed RGB)
    content_layers -- chosen content layer
    style_coeff -- coefficients used to compute the content cost for its respective layer
    style -- normalized and reshaped style image array (assumed RGB)
    style_layers -- list of the names of the chosen style layer 
    style_coeff -- coefficients used to compute the style cost for its respective layer
    alpha -- hyperparameter for the weight for the content cost (higher relative to beta -> more influence from content layer)
    beta -- hyperparameter for the weight for the style cost (higher relative to alpha -> more influence from style layers)
    fn -- cost function (either L1 or L2)
    
    Outputs:
    content_cost -- the conetent cost
    style_cost -- the style cost
    total_cost -- the combined weighted style and content costs
    """
    
    # set the content image as the current input to the VGG19 model, then run the it through the model in the session 
    print('compute content cost...')
    sess.run(model['input'].assign(content))
    
    # compute the style cost for the given style layers with the given style coefficients and current state of the session (style as input)
    content_cost = compute_content_cost(model, content_layers, content_coeffs, sess, fn)
    
    # set the style image as the current input to the VGG19 model, then run the it through the model in the session 
    print('compute style cost...')
    sess.run(model['input'].assign(style))
    
    # compute the style cost for the given style layers with the given style coefficients and current state of the session (style as input)
    style_cost = compute_style_cost(model, style_layers, style_coeffs, sess, fn)
    
    # compute the total cost 
    print('compute total cost...')
    total_cost = compute_total_cost(content_cost, style_cost, alpha, beta)

    return content_cost, style_cost, total_cost
    
    







