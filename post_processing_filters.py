import cv2
import numpy as np

def adjust_brightness(image, brightness): 
    return np.clip(image * brightness, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(image, brightness=0, contrast=1): 
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)



def adjust_contrast(image, contrast_factor): 
  
    contrast_factor = np.clip(contrast_factor, 0, 3)  

    mean = np.mean(image, axis=(0, 1), dtype=np.float32)  
    adjusted_image = (image - mean) * contrast_factor + mean  

   
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image

def adjust_saturation(image, saturation_scale):
    
    if image is None:
        raise ValueError("Invalid image input.")

    if saturation_scale < 0:
        raise ValueError("Saturation scale must be non-negative.")


    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    h, s, v = cv2.split(hsv_image)


    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)


    hsv_image = cv2.merge([h, s, v])


    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return saturated_image



def convert_to_grayscale(image): 

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_filter_to_patch(patch, filter):

    return np.sum(patch * filter)

def apply_filter_to_image(image, filter):

    if len(image.shape) == 3 and image.shape[2] == 3:
        output_image = np.zeros_like(image)
        for channel in range(3):
            output_image[..., channel] = apply_filter_to_channel(image[..., channel], filter)
        return output_image
    else:
        return apply_filter_to_channel(image, filter)

def apply_filter_to_channel(channel, filter):

    filter_height, filter_width = filter.shape
    half_height = filter_height // 2
    half_width = filter_width // 2
    
    output_channel = np.zeros_like(channel)
    

    for i in range(half_height, channel.shape[0] - half_height):
        for j in range(half_width, channel.shape[1] - half_width):
            patch = channel[i - half_height:i + half_height + 1, j - half_width:j + half_width + 1]
            output_channel[i, j] = apply_filter_to_patch(patch, filter)
    
    return output_channel

def apply_numpy_convolution(image, filter):

    return cv2.filter2D(image, -1, filter)



def create_gaussian_filter(half_size):
    size = 2 * half_size + 1
    sigma = half_size / 3.0
    ax = np.arange(-half_size, half_size + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    return sharpened_image
