import os
import shutil

'''
image_retriever.py

This is a fairly simple helper file, it's functional purpose is to copy an image file from 
the path provided by the user when executing the main.py script to the input directory of this project.

'''
def store_image(image_path: str, input_dir: str = "input_images/") -> str:

    os.makedirs(input_dir, exist_ok=True) #confirms existence of the input directory

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file '{image_path}' does not exist.")

    valid_extensions = {".jpg", ".jpeg", ".png"} #extensions which will can be processed
    _, file_extension = os.path.splitext(image_path)

    #error reporting if the user submits an image with an incorrect extension
    if file_extension.lower() not in valid_extensions:
        raise ValueError(f"Error: the provided file at '{image_path}' is not a valid image. The valid extensions are {valid_extensions}")

    destination_path = os.path.join(input_dir, os.path.basename(image_path))

    #attempts to copy the image file from the user provided path to the input directory
    '''IMPORTANT! This program was developed on and for use with Ubuntu 24.04. I cannot confirm 
        whether or not this works properly on Windows/MacOS, but functionality should be identical
        with any reasonably modern Debian/Ubuntu OS.
    '''
    try:
        shutil.copy(image_path, destination_path)
    except Exception as e:
        raise OSError(f"Operating System Error: Could not copy file to '{destination_path}': {e}")

    return destination_path
