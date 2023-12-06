# Neural Style Transfer
Neural Style Transfer using TensorFlow

## Style Transfer

Neural Style Transfer is an artistic technique that uses deep learning to combine the content of one image with the style of another image, creating a new image that retains the content structure but adopts the artistic characteristics of the reference style image. This process is based on the representation of images by deep neural networks, particularly convolutional neural networks (CNNs).


This repository contains a Python script for performing Neural Style Transfer using TensorFlow. The script employs the VGG19 model and the Streamlit library to create a straightforward web application for applying artistic styles to images.

## Overview

Here's a brief overview of the key components involved in Neural Style Transfer:

Content Image: The content image is the source image whose general structure and content you want to retain in the final stylized image.

Style Image: The style image is the reference image whose artistic style (textures, colors, patterns) you want to apply to the content image.

Feature Extraction: Deep neural networks, often pre-trained on large datasets, are used to extract features from both the content and style images. Convolutional layers in these networks capture hierarchical representations of features, such as edges, textures, and higher-level patterns.

Content and Style Representations: Specific layers in the neural network are chosen to represent the content and style features. The content representation typically comes from a higher layer where more abstract features are captured, while the style representation comes from multiple layers capturing different levels of abstraction.

Loss Functions: Two types of loss functions are defined to optimize the generated image: content loss and style loss.

Content Loss: Measures the difference between the content features of the generated image and the content image. The goal is to ensure that the generated image retains the content of the original image.

Style Loss: Compares the Gram matrices of the features from the generated image and the style image. The Gram matrix encodes information about the style by capturing correlations between different features. The goal is to replicate the style of the reference image.

Optimization: The generated image is iteratively updated to minimize the total loss, which is a combination of content loss, style loss, and possibly other terms like total variation loss. Optimization methods such as gradient descent are used for this purpose.

The iterative optimization process continues until the generated image achieves a good balance between content preservation and style transfer. Neural Style Transfer has been widely used for creating visually appealing and artistic images, and it has applications in various domains, including art, design, and computer graphics.


## Results
The script will display intermediate stylized images during the style transfer process. The final stylized image will be saved to the specified file path.

## Acknowledgments
The script uses a pre-trained VGG19 model from TensorFlow's Keras applications for feature extraction. The technique is inspired by the original paper on Neural Style Transfer by Gatys et al.

## Author
Sai Kumar Diguvapatnam
