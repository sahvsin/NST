# -*- coding: utf-8 -*-
"""
@author: Sahil
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


mean_offset = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


def add_bias(image):
    return image - mean_offset


def remove_bias(image):
    return image + mean_offset


def load_images(content_filepath, style_filepath, image_height, image_width):

    """
    loads style and content images from specified filepaths and converts to RGB with specified dimensions

    Inputs:
    content_filepath -- path of the content image(relative or absolute)
    style_filepath -- path of the style image (relative or absolute)
    image_height -- height to resize images to
    image_width -- width to resize images to

    Outputs:
    content -- resized RGB read content image
    style -- resized RGB read style image
    """

    # read the images (openCV reads in BGR)
    content_BGR = cv2.imread(content_filepath)
    style_BGR = cv2.imread(style_filepath)

    # convert the image from BGR to RGB
    content_RGB = cv2.cvtColor(content_BGR, cv2.COLOR_BGR2RGB)
    style_RGB = cv2.cvtColor(style_BGR, cv2.COLOR_BGR2RGB)
    
    # resize the images
    content = cv2.resize(content_BGR, (image_width, image_height))
    style = cv2.resize(style_RGB, (image_width, image_height))

    return content, style




def img_reshape_and_normalize(content_arr, style_arr):

    """
    reshapes the image data and adds the NST normalization offset

    Inputs:
    content_arr -- read content image matrix (height, width, 3)
    style_arr -- read style image matrix (height, width, 3)
    
    Outputs:
    content -- reshaped and normalized content image array 
    style -- reshaped and normalized style image array
    """
    
    # convert data to numpy array and cast to floats
    np_content = np.asarray(content_arr, dtype = 'float32')
    np_style = np.asanyarray(style_arr, dtype = 'float32')
    
    # reshape data to shape (4-dimensions) expected by the VGG-19 model
    content = np.expand_dims(np_content, axis = 0)
    style = np.expand_dims(np_style, axis = 0)
    
    # apply mean offset
    content = add_bias(content)
    style = add_bias(style)
    
    return content, style




def create_noise_image(content_arr, style_arr, noise_ratio, image_height, image_width, num_channels, option):
    
    """
    creates the noisy image (initialized generated image) by adding random noise to the content image
    
    Inputs:
    content_arr -- content image matrix (1, height, width, num_channels)
    style_arr -- style image matrix (1, height, width, num_channels)
    noise_ration -- hyperparameter tuning the noisiness of the initialized generated image
    image_height -- height of the image
    image_width -- width of the image
    num_channels -- number of color channels to use (0 - black/gray/white, 3 - color)
    option -- choice of generated image base (uniform noise, noisy content, noisy style)
    """
    
    #noise = np.random.uniform(-20, 20, (1, image_height, image_width, num_channels)).astype('float32')
    noise = .256*np.random.normal(size=(1, image_height, image_width, num_channels), scale=np.std(content_arr)*.1).astype('float32')

    if option is 'content':
        return noise_ratio*noise + (1-noise_ratio)*content_arr
    elif option is 'style':
        return noise_ratio*noise + (1-noise_ratio)*style_arr
    return noise



def show_image(image):
    
    """
    displays the image (array) in a new window
    
    Inputs:
    image -- the image array to write
    """

    # remove the offset
    image = remove_bias(image)
    
    # clip the image, bind the values of the image between 0 and 255 (pixel values), and cast to ints
    out_img = np.clip(image[0], 0, 255).astype('uint8')
   
    cv2.imshow("NST Image", out_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.waitKey(1)


def save_image(image, output_path):
    
    """
    saves (writes) the image (array) to the output path
    
    Inputs:
    image -- the image array to write
    output_path -- the destination
    """

    # remove the offset
    image = remove_bias(image)
    
    # clip the image, bind the values of the image between 0 and 255 (pixel values), and cast to ints
    out_img = np.clip(image[0], 0, 255).astype('uint8')
   
    cv2.imwrite(output_path, out_img)
