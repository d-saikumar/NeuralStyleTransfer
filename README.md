# Neural_Style_Transfer
Neural Style Transfer using TensorFlow

This repository contains a Python script for performing Neural Style Transfer using TensorFlow. The script employs the VGG19 model and the Streamlit library to create a straightforward web application for applying artistic styles to images.

## Usage
Overview
Neural Style Transfer is a technique that combines the content of one image with the style of another image to create visually appealing and artistic results. The script implements this technique through the following steps:

Loading and Preprocessing Images: Content and style images are loaded and preprocessed using the VGG19 model's preprocessing functions.

Defining Content and Style Layers: Specific layers in the VGG19 model are chosen to extract content and style features.

Building the VGG19 Model: A modified VGG19 model is created to access intermediate layers for both style and content.

Computing Content Loss: A function is defined to compute the content loss between the base content and the target content.

Computing Gram Matrix and Style Loss: Functions are defined to compute the gram matrix and style loss between the base style and the target style.

Total Variation Loss: A function is defined to compute the total variation loss to promote spatial smoothness in the stylized image.

Getting Style and Content Features: The script defines a function to get style and content features from the model.

Computing Total Loss: A function is defined to compute the total loss using style and content weights.

Computing Gradients: A function is defined to compute gradients for optimization.

Running Style Transfer: The main function runs the style transfer process using the defined functions and parameters.

Displaying Results: Intermediate and final stylized images are displayed during the style transfer process.

Saving the Final Stylized Image: The final stylized image is saved to a specified path.

## Results
The script will display intermediate stylized images during the style transfer process. The final stylized image will be saved to the specified file path.

## Acknowledgments
The script uses a pre-trained VGG19 model from TensorFlow's Keras applications for feature extraction. The technique is inspired by the original paper on Neural Style Transfer by Gatys et al.

## Author
Sai Kumar Diguvapatnam
