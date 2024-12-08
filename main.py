import os
import argparse
from image_retriever import store_image
from mask_selector import process_image
from filter_processor import process_remaining_content, process_mask_content
from image_merger import merge_images
'''
main.py

This script is the primary driver code for the RCNN Image Enhancement project. In this file,
you will find two primary elements of interest:

1 - get_processing_intensity()

    This method was defined as a way to use user input to define the type of image enhancement that would 
    be applied to the image passed as a program argument. The user is presented with a set of enhancement 
    profiles, and they select one in turn. This selection corresponds to an option in the intensity map below.

    The intensity map predefines values for image brightness, contrast, saturation, and blurring intensity.
    These values are associated with their respective options in the intensity map.

    Presently, I have defined three profiles.

    Profile 1 (Object Isolation) is used mainly as a demonstration tool. It isolates the focus object from the
    provided image, converts all other image data to grayscale, and applies a heavy blur to the non-object content.

    Profile 2 (Object Identification) is the default option, and is used to identify the object of interest in an image.
    It applies a mild enhancement to the masked object to enhance it's visual presence.

    Profile 3 (Object Clarification) applies filters more aggressively to enhance the focus object content, creating
    a greater contrast with other image data to produce a new image which has a heavy focus on the focus content.
'''

def get_processing_intensity():
    print("Select a processing intensity option:")
    print("1. Object Isolation (Grayscale non-object content with heavy blur)")
    print("2. Object Identification (Mild Gaussian blur to non-object content, retaining similar brightness/contrast)")
    print("3. Object Clarification (Increased contrast and saturation of the object content)")
    choice = input("Enter 1, 2, or 3: ").strip()

    intensity_map = {
        "1": {
            "object": {
                "brightness": 1.1,
                "contrast": 1.2,
                "saturation": 1.1, 
                "blur_size": 0  
            },
            "non_object": {
                "brightness": 0.7,  
                "contrast": 0.7,    
                "saturation": 0.0,  
                "blur_size": 20     
            }
        },
        "2": {
            "object": {
                "brightness": 1.2,
                "contrast": 1.1,
                "saturation": 1.1,
                "blur_size": 0  
            },
            "non_object": {
                "brightness": 1.0,
                "contrast": 1.0,
                "saturation": 1.0,
                "blur_size": 5  
            }
        },
        "3": {
            "object": {
                "brightness": 1.1,
                "contrast": 1.3,
                "saturation": 1.5,
                "blur_size": 0  
            },
            "non_object": {
                "brightness": 0.95,
                "contrast": 0.8,
                "saturation": 0.8,
                "blur_size": 10  
            }
        }
    }

    if choice not in intensity_map:
        print("Invalid choice. Defaulting to Object Identification.")
        return intensity_map["2"]

    return intensity_map[choice]

'''
2 - Main function
    This function drives the program, and is the script which is initially called.

    I have added comments throughout this function to provide more context on the workflow.
'''


def main():

    #the parser object is used to interpret the input image (as a path)
    parser = argparse.ArgumentParser(description="Enhance an image using R-CNN detection and filtering.")
    parser.add_argument("image_path", help="Path to the input image.")

    #as well as the optional output path variable, which allows the user to override the default output path
    parser.add_argument(
        "--output_path", 
        help="Path to save the final processed image (default is ./final_output/).",
        default="./final_output/"
    )
    args = parser.parse_args()

    #program workflow is run in a try/catch block to handle errors (errors can be thrown by incorrect package versions - see readme)
    try:

        #the image path provided by the user is first used to copy the target image into the working directory
        input_dir = "./mrcnn/input_images/"
        stored_image_path = store_image(args.image_path, input_dir)
        print(f"Image stored successfully at: {stored_image_path}")

        #the enahncement profile is stores in the intensity value
        intensity = get_processing_intensity()
        print(f"Selected intensity: {intensity}")

        #after defining the output directory, the mrcnn code is called by the process_image() method
        #to split the provided image into a focus object image, and another image containing remaining content
        output_dir = "./mrcnn/output_images/"
        mask_content_path, remaining_content_path = process_image(stored_image_path, output_dir)
        print(f"Soft mask content saved at: {mask_content_path}")
        print(f"Remaining content saved at: {remaining_content_path}")


        #non-focus data is postprocessed according to the selected filter profile
        filtered_remaining_content_path = process_remaining_content(
            remaining_content_path,
            brightness=intensity["non_object"]["brightness"],
            contrast=intensity["non_object"]["contrast"],
            saturation=intensity["non_object"]["saturation"],
            blur_size=intensity["non_object"]["blur_size"]
        )
        print(f"Filtered remaining content saved at: {filtered_remaining_content_path}")

        #the masked object image is postprocessed using the filter profile selected by the user
        filtered_mask_content_path = process_mask_content(
            mask_content_path,
            brightness=intensity["object"]["brightness"],
            contrast=intensity["object"]["contrast"],
            saturation=intensity["object"]["saturation"],
            blur_size=0,  
            sharpening=1 #here's a note if you (the user) want to tweak the visual look of the output images - set this value to 0 to eliminate sharpening
        )
        print(f"Filtered mask content saved at: {filtered_mask_content_path}")

        #saving the image is done by merging the filtered mask content and remaining content images
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path) 
        final_image_path = os.path.join(args.output_path, "filtered_image.jpg")
        final_image = merge_images(filtered_mask_content_path, filtered_remaining_content_path, final_image_path)
        
        print(f"Final image saved at: {final_image_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
