import os
from image_retriever import store_image
from mask_selector import process_image
from filter_processor import process_remaining_content, process_mask_content
from image_merger import merge_images

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process an image using Mask R-CNN.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--output_path", 
        help="Path to save the final processed image (default is ./final_output/).",
        default="./final_output/"
    )
    args = parser.parse_args()

    try:
        input_dir = "./mrcnn/input_images/"
        stored_image_path = store_image(args.image_path, input_dir)
        print(f"Image stored successfully at: {stored_image_path}")

        intensity = get_processing_intensity()
        print(f"Selected intensity: {intensity}")

        output_dir = "./mrcnn/output_images/"
        mask_content_path, remaining_content_path = process_image(stored_image_path, output_dir)
        print(f"Soft mask content saved at: {mask_content_path}")
        print(f"Remaining content saved at: {remaining_content_path}")

        filtered_remaining_content_path = process_remaining_content(
            remaining_content_path,
            brightness=intensity["non_object"]["brightness"],
            contrast=intensity["non_object"]["contrast"],
            saturation=intensity["non_object"]["saturation"],
            blur_size=intensity["non_object"]["blur_size"]
        )
        print(f"Filtered remaining content saved at: {filtered_remaining_content_path}")

        filtered_mask_content_path = process_mask_content(
            mask_content_path,
            brightness=intensity["object"]["brightness"],
            contrast=intensity["object"]["contrast"],
            saturation=intensity["object"]["saturation"],
            blur_size=0,  
            sharpening=1
        )
        print(f"Filtered mask content saved at: {filtered_mask_content_path}")

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path) 
        final_image_path = os.path.join(args.output_path, "filtered_image.jpg")
        final_image = merge_images(filtered_mask_content_path, filtered_remaining_content_path, final_image_path)
        
        print(f"Final image saved at: {final_image_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
