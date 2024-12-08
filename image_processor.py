import cv2
import os



def merge_images(mask_content_path: str, modified_remaining_content_path: str, output_path: str = "./final_output/merged_image.jpg"):
  
    mask_content = cv2.imread(mask_content_path, cv2.IMREAD_UNCHANGED)
    modified_remaining_content = cv2.imread(modified_remaining_content_path, cv2.IMREAD_UNCHANGED)

    if mask_content is None:
        raise FileNotFoundError(f"Mask content image not found: {mask_content_path}")
    if modified_remaining_content is None:
        raise FileNotFoundError(f"Modified remaining content image not found: {modified_remaining_content_path}")

    if mask_content.shape[:2] != modified_remaining_content.shape[:2]:
        raise ValueError("Mask content and remaining content images must have the same dimensions.")

    if len(mask_content.shape) == 2:  
        mask_content = cv2.cvtColor(mask_content, cv2.COLOR_GRAY2BGR)
    if len(modified_remaining_content.shape) == 2: 
        modified_remaining_content = cv2.cvtColor(modified_remaining_content, cv2.COLOR_GRAY2BGR)

    merged_image = cv2.addWeighted(mask_content, 1.0, modified_remaining_content, 0.5, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, merged_image)

    return output_path
