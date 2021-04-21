# image and pre-trained weights filenames
content_filename = 'GoldenGateBridge.jpg'
style_filename = '/home/sahil/projects/ML/TF/nst/styles/beautiful_dance.jpg'
pretrained_weights_path = 'imagenet-vgg-verydeep-19.mat'
out_filename = 'nst_image.jpg'

# generated image dimensions
image_height = 400
image_width = 400
num_channels = 3

# noisy image generation parameters
option = 'content'      #noisy -> generate uniform noise, content -> noise+content image, style -> noise+style image
noise_ratio = .6        #noisiness WRT to content/style image (if latter 2 options chosen)


# layers elected for content and style feature extraction
content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

func = "L2"		#cost function (L1 or L2)

content_coeffs = 1.0/float(len(content_layers))      #coefficients to compute content cost
style_coeffs = 1.0/float(len(style_layers))          #coefficients to compute style cost

alpha = 1e1           #content cost weight when computing total cost
beta = 1e10          #style cost weight when computing total cost

# boolean determining whether or not to use total variation regularization (denoising/smoothing filter)
tv_reg = True

# optimization parameters
learn_rate = 10
num_epochs = 150

adam_beta1 = .9
adam_beta2 = .999
adam_epsilon = 1e-8

# rate at which to print epoch cost (i.e. every 5 epochs)
print_rate = 1

# want to show image after finishing NST
show_img = True