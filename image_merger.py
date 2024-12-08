import cv2
import os
import numpy as np

'''
image_merger.py

This script contains the method which is used to combine the postprocessed object/background images into one cohesive 
filtered output image.
'''

def merge_images(mask_content_path: str, modified_remaining_content_path: str, output_path):
    
    #checking for the images being 3-channel reduces the risk of having the program fail due to images being merged 
    #with incompatible dimensions
    mask_content = cv2.imread(mask_content_path, cv2.IMREAD_COLOR) #checks the mask image to ensure it is 3-channel
    modified_remaining_content = cv2.imread(modified_remaining_content_path, cv2.IMREAD_COLOR) #same check for the remaining content image

    #error reporting in case of incompatible dimensions
    if mask_content is None:
        raise FileNotFoundError(f"Error: the mask content image was not found: {mask_content_path}")
    if modified_remaining_content is None:
        raise FileNotFoundError(f"Error: the remaining content image was not found: {modified_remaining_content_path}")

    if mask_content.shape[:2] != modified_remaining_content.shape[:2]:
        raise ValueError("Error: the dimensions of the focus and remainder images must be the same. Check the input to merge_images()!")


    #the grayscale and alpha masks are used to create the blurred mask,which is then used to blend the two images together
    grayscale_mask = cv2.cvtColor(mask_content, cv2.COLOR_BGR2GRAY)
    alpha_mask = grayscale_mask / 255.0

    blurred_alpha_mask = cv2.GaussianBlur(alpha_mask, (51, 51), 0)

    #the new blended image is created by multiplying the masked image with the blurred alpha mask and adding it to 
    #the remaining content image
    blended_image = (
        mask_content * blurred_alpha_mask[:, :, np.newaxis] + 
        modified_remaining_content * (1 - blurred_alpha_mask[:, :, np.newaxis])  
    ).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, blended_image)

    return output_path
