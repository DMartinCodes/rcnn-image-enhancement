import os
import shutil

def store_image(image_path: str, input_dir: str = "input_images/") -> str:

    os.makedirs(input_dir, exist_ok=True)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file '{image_path}' does not exist.")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    _, file_extension = os.path.splitext(image_path)
    if file_extension.lower() not in valid_extensions:
        raise ValueError(f"The file '{image_path}' is not a valid image.")

    destination_path = os.path.join(input_dir, os.path.basename(image_path))

    try:
        shutil.copy(image_path, destination_path)
    except Exception as e:
        raise OSError(f"Error copying file to '{destination_path}': {e}")

    return destination_path
