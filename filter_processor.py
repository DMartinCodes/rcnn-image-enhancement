import cv2
import os
from post_processing_filters import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    convert_to_grayscale,
    apply_numpy_convolution,
    create_gaussian_filter,
    sharpen_image
)

'''
filter_processor.py

This script is used to apply postprocessing filters to the content/noncontent images. 
The main function calls the two methods defined in this script, and passes the filter values selected by the user
to these methods. The process_remaining_content() and process_mask_content() methods then use these arguments
to apply filters of varying intensities to their respective images.
'''


#postprocesses the remaining content image
def process_remaining_content(image_path: str, brightness, contrast, saturation, blur_size, output_dir: str = "./processed_images/"):


    image = cv2.imread(image_path) 
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if blur_size != 0: #conditional just skips application of the gaussian filter if the blur size is 0

        gaussian_filter = create_gaussian_filter(blur_size) #calls up the function from post_processing_filters.py

        image = apply_numpy_convolution(image, gaussian_filter)

   #brightness of the image is adjusted by the user's input
    image = adjust_brightness(image, brightness)


    #similarly, contrast is modified by passing the profile value through to the filter method
    image = adjust_contrast(image, contrast)


    #saturation adjustment
    image = adjust_saturation(image, saturation)

    os.makedirs(output_dir, exist_ok=True)
    #writes out the modified remainder image to the processed images folder
    output_path = os.path.join(output_dir, "processed_aggressive.jpg")
    cv2.imwrite(output_path, image)

    return output_path

'''this method works very similarly to the process_remaining_content method, 
    so I will not repeat my comments for most of the function calls, but please note the
    addition of the sharpening method
'''

def process_mask_content(image_path: str, brightness, contrast, saturation, blur_size, sharpening, output_dir: str = "./processed_images/"):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")


    image = adjust_brightness(image, brightness)
    image = adjust_contrast(image, contrast)
    image = adjust_saturation(image, saturation)


    '''
    The user may note that the processing method for the focus object has a sharpening function call,
    instead of a blurring method call.

    This change was made after testing out the program showed that, to reduce the abruptness of the transition
    between the focus object and background in the merged image, gaussian blurring could be used to feather the
    transitionary areas. The problem introduced then, by that solution, was that the focus object was blurred!

    To resolve the issue and preserve details, I added the call to the sharpen_image() method, which clarifies
    focus object details by sharpening them.
    '''
    if sharpening != 0:

        image = sharpen_image(image)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "processed_mask_content.jpg")
    cv2.imwrite(output_path, image)

    return output_path
