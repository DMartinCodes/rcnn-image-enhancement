

import mrcnn.config
import mrcnn.model
import cv2
import numpy as np
import os

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

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)
    DETECTION_MIN_CONFIDENCE = 0.9  

model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(), 
                             model_dir=os.getcwd())

model.load_weights(filepath="mask_rcnn_coco.h5", by_name=True)

def process_image(image_path: str, output_dir: str = "output_images/"):
 
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model.detect([image_rgb], verbose=0)[0]

    target_classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
               'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']  
    valid_indices = [i for i, cid in enumerate(results['class_ids']) 
                     if CLASS_NAMES[cid] in target_classes]

    if not valid_indices:
        raise ValueError("No target objects detected in the image.")

    masks = results['masks']
    mask_areas = [np.sum(masks[:, :, i]) for i in valid_indices]
    main_mask_index = valid_indices[np.argmax(mask_areas)]
    main_mask = masks[:, :, main_mask_index]

    soft_mask = (main_mask * 255).astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)  
    soft_mask = cv2.erode(soft_mask, kernel, iterations=1)

    soft_mask_content = cv2.bitwise_and(image, image, mask=soft_mask)

    inverse_mask = cv2.bitwise_not(soft_mask)
    remaining_content = cv2.bitwise_and(image, image, mask=inverse_mask)
 

    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    mask_content_path = os.path.join(output_dir, f"{base_filename}_mask_content.jpg")
    remaining_content_path = os.path.join(output_dir, f"{base_filename}_remaining_content.jpg")
    cv2.imwrite(mask_content_path, soft_mask_content)
    cv2.imwrite(remaining_content_path, remaining_content)

    return mask_content_path, remaining_content_path