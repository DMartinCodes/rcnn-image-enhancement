

import mrcnn.config
import mrcnn.model
import cv2
import numpy as np
import os

'''
mask_selector.py

This script is responsible for using the mrcnn model.py script to bisect the provided input image into two
images:
    1. An image containing soft-masked data of the focus object (this typically corresponds to one of the classes
    used in training the mask-rcn-coco.h5 model).
    2. An image containing all input data except the soft mask data.

IMPORTANT - I did not write the code in the mrcnn folder. This code is provided under the MIT License by the author
of the original Matterport Mask-RCNN repository. The original repository can be found at: https://github.com/matterport/Mask_RCNN

I make no claim that the R-CNN code is my own. I am simply using it as a component of this project (much like a python library such as openCV)
and have not modified the code in any way. 

'''

#the class names which apply to the mask_rcnn_coco.h5's training
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
               'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#the SimpleConfig class is used to configure the behaviour of the mrcnn model
class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference" 
    GPU_COUNT = 1 #my program generally defaults to using the CPU
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)
    DETECTION_MIN_CONFIDENCE = 0.9  #min confidence value has been modified to create a more selective soft mask
#defines the model to be using the inference mode
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(), 
                             model_dir=os.getcwd())

#loads weights (make sure, if you didn't download the weights from my repo, that you obtain them from the 
#original R-CNN repo I provided the link to above!)
model.load_weights(filepath="mask_rcnn_coco.h5", by_name=True)


#process image method splits the provided image based on mask/nonmask content
def process_image(image_path: str, output_dir: str = "output_images/"):
 
    #reads in the image daata
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    #converts the image to RGB format for intepretation by the R-CNN model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #the resulting detection is then defined in results
    results = model.detect([image_rgb], verbose=0)[0]

    #the target classes are used to determine whether or not the image contains a focus object
    #IE a blank image should not cause the program to break because there is no focus object to enhance
    target_classes = CLASS_NAMES
    valid_indices = [i for i, cid in enumerate(results['class_ids']) if CLASS_NAMES[cid] in target_classes]

    if not valid_indices:
        raise ValueError("No target objects detected in the image.")

    #the mask is then defined as the mask with the largest area
    #this is an assumption I am making to reduce computational complexity that would be inherent in using a more
    #complex method of focus detection. Generally, the largest mask in the results set will correspond to the mask 
    #covering the object which is the focus of the image (IE a selfie will have the largest mask being of the person taking the photo)
    masks = results['masks']
    
    #computing the areas of the masks to determine which is the largest
    mask_areas = [np.sum(masks[:, :, i]) for i in valid_indices]
    main_mask_index = valid_indices[np.argmax(mask_areas)] #uses argmax to select the largest (focus) mask
    main_mask = masks[:, :, main_mask_index]

    #the soft mask is then defined as the main_mask multiplied across the 255 colour channels
    soft_mask = (main_mask * 255).astype(np.uint8)

    '''
    Mask postprocessing - initially, there were issues where the soft mask would capture too much background data.
    To reduce this, the script now uses cv2.erode() to perform an erosion on the soft mask using the kernel. This 
    slides the kernel over the softmask image component, and converts any pixel which has proximity to non-mask areas to 
    itself be removed from the mask. 
    '''
    kernel = np.ones((10, 10), np.uint8)  
    soft_mask = cv2.erode(soft_mask, kernel, iterations=1)

    #AND operation is then used to isolate the soft mask content from the original image
    soft_mask_content = cv2.bitwise_and(image, image, mask=soft_mask)

    #AND operation is used with the mask data inverse to define the non-mask image
    inverse_mask = cv2.bitwise_not(soft_mask)
    remaining_content = cv2.bitwise_and(image, image, mask=inverse_mask)
 

    os.makedirs(output_dir, exist_ok=True)

    #exporting the new images for postprocessing
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    mask_content_path = os.path.join(output_dir, f"{base_filename}_mask_content.jpg")
    remaining_content_path = os.path.join(output_dir, f"{base_filename}_remaining_content.jpg")
    cv2.imwrite(mask_content_path, soft_mask_content)
    cv2.imwrite(remaining_content_path, remaining_content)

    return mask_content_path, remaining_content_path