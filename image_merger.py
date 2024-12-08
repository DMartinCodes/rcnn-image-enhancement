import cv2
import os
import numpy as np

def merge_images(mask_content_path: str, modified_remaining_content_path: str, output_path):
    """
    Merges the mask content image and the modified remaining content image with a smooth transition.

    Args:
        mask_content_path (str): Path to the soft-mask content image.
        modified_remaining_content_path (str): Path to the modified remaining content image.
        output_path (str): Path to save the merged image.

    Returns:
        str: Path to the saved merged image.
    """
    mask_content = cv2.imread(mask_content_path, cv2.IMREAD_COLOR)  # Ensure 3-channel
    modified_remaining_content = cv2.imread(modified_remaining_content_path, cv2.IMREAD_COLOR)  # Ensure 3-channel

    if mask_content is None:
        raise FileNotFoundError(f"Mask content image not found: {mask_content_path}")
    if modified_remaining_content is None:
        raise FileNotFoundError(f"Modified remaining content image not found: {modified_remaining_content_path}")

    if mask_content.shape[:2] != modified_remaining_content.shape[:2]:
        raise ValueError("Mask content and remaining content images must have the same dimensions.")

    grayscale_mask = cv2.cvtColor(mask_content, cv2.COLOR_BGR2GRAY)

    alpha_mask = grayscale_mask / 255.0

    blurred_alpha_mask = cv2.GaussianBlur(alpha_mask, (51, 51), 0)

    blended_image = (
        mask_content * blurred_alpha_mask[:, :, np.newaxis] + 
        modified_remaining_content * (1 - blurred_alpha_mask[:, :, np.newaxis])  
    ).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, blended_image)

    return output_path
