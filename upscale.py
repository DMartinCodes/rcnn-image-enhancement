import cv2
import numpy as np


'''
upscale.py

This file contains the methods necessary to upscale the image by a user-defined factor

'''



def upscale_image(image, scale_factor, model_path=None, model_name=None): #driver method which passes the args to correct function based on sFactor
   
    scale_factor = int(scale_factor)

    
    model_path = "FSRCNN_x3.pb"
    model_name = 'fsrcnn'
    sr = cv2.dnn_superres.DnnSuperResImpl_create() #creates model instance
    sr.readModel(model_path)
    sr.setModel(model_name, scale_factor) #model is configured with the scaling factor
    return sr.upsample(image)