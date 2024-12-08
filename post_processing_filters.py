import cv2
import numpy as np
'''
post_processing_filters.py

This script defines all the filter methods used to enhance the masked and remainder images.
'''

def adjust_brightness(image, brightness): #returns pixel data with brightness adjusted by the provided brightness scaling factor
    return np.clip(image * brightness, 0, 255).astype(np.uint8)



def adjust_contrast(image, contrast_factor): 
    #adjusts contrast based on the provided contrast scaling factor
  
    contrast_factor = np.clip(contrast_factor, 0, 3)   

    mean = np.mean(image, axis=(0, 1), dtype=np.float32)  #calculate mean of the image
    adjusted_image = (image - mean) * contrast_factor + mean  #adjustment is based on the contrast scaling factor and the mean of the image

   
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image

def adjust_saturation(image, saturation_scale):
    #uses the provided scaling factor to adjust image saturation
    if image is None:
        raise ValueError("Error: invalid image input. Check the main function!")

    if saturation_scale < 0:
        raise ValueError("Error: the saturation scale must be non-negative. Check profile map in main.py for negative saturation values")

    #converting the image to HSV format for saturation adjustment, as this provides a specific channel for saturation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #splitting the image into the three channels
    h, s, v = cv2.split(hsv_image)

    #scaling saturation channel
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)

    #merging modified channel back into image
    hsv_image = cv2.merge([h, s, v])
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return saturated_image



def convert_to_grayscale(image): #returns image data in grayscale
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_filter_to_patch(patch, filter): #uses matrix sum method to return results of applying a filter to an image patc
    return np.sum(patch * filter)

def apply_filter_to_image(image, filter):

    if len(image.shape) == 3 and image.shape[2] == 3: #checking if the image has three colour channels

        output_image = np.zeros_like(image) #"blank" image

        for channel in range(3): #iterates over each channel, applying the filter to that channel independently
            output_image[..., channel] = apply_filter_to_channel(image[..., channel], filter)
        return output_image

    else:
        return apply_filter_to_channel(image, filter)

def apply_filter_to_channel(channel, filter):
    #applies a filter to an entire channel of an image using the apply_filter_to_patch method

    #using filter shape property to define the dimensions
    filter_height, filter_width = filter.shape
    half_height = filter_height // 2
    half_width = filter_width // 2
    
    #defines a "blank" output channel
    output_channel = np.zeros_like(channel)

    #iterates over the image channel to apply the filter
    for i in range(half_height, channel.shape[0] - half_height):

        for j in range(half_width, channel.shape[1] - half_width):

            patch = channel[i - half_height:i + half_height + 1, j - half_width:j + half_width + 1]
            output_channel[i, j] = apply_filter_to_patch(patch, filter)
    
    return output_channel

def apply_numpy_convolution(image, filter): #applies convolution using filter method
    return cv2.filter2D(image, -1, filter)



def create_gaussian_filter(half_size): 
    #uses a provided kernel half size to create a gaussian filter
    size = 2 * half_size + 1

    sigma = half_size / 3.0 #sigma value is defined as a third of the half size

    ax = np.arange(-half_size, half_size + 1)
    xx, yy = np.meshgrid(ax, ax)

    #the kernel is defined based on the gaussian equation provided in class
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def sharpen_image(image):

    #sharpen_image is used to improve definition of image details

    #a laplacian is used as the kernel, where the average of the pixels surrounding the target is 
    #subtracted and the center pixel is given a greater weight, which creates a better-defined edge
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]
                       ])
    
    sharpened_image = cv2.filter2D(image, -1, kernel) #the laplacian kernel is then convolved with the imaged
    
    return sharpened_image
