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

def process_remaining_content(image_path: str, brightness, contrast, saturation, blur_size, output_dir: str = "./processed_images/"):

    """
    Applies aggressive versions of all filters to the remaining_content image.

    Args:
        image_path (str): Path to the `remaining_content` image.
        output_dir (str): Directory to save the processed image.

    Returns:
        str: Path to the processed image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if blur_size != 0:

        gaussian_filter = create_gaussian_filter(blur_size)

        image = apply_numpy_convolution(image, gaussian_filter)

   
    image = adjust_brightness(image, brightness)


    
    image = adjust_contrast(image, contrast)


  
    image = adjust_saturation(image, saturation)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "processed_aggressive.jpg")
    cv2.imwrite(output_path, image)

    return output_path


def process_mask_content(image_path: str, brightness, contrast, saturation, blur_size, sharpening, output_dir: str = "./processed_images/"):

    """
    Applies aggressive versions of filters to the mask_content image.

    Args:
        image_path (str): Path to the `mask_content` image.
        output_dir (str): Directory to save the processed image.

    Returns:
        str: Path to the processed image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")



  
    image = adjust_brightness(image, brightness)


    image = adjust_contrast(image, contrast)


    image = adjust_saturation(image, saturation)

    if sharpening != 0:

        image = sharpen_image(image)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "processed_mask_content.jpg")
    cv2.imwrite(output_path, image)

    return output_path
