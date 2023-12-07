import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from utils.io import get_image_label_pairs, list_files

# Define your data directory
data_root = "data"
label_to_idx_map = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Proliferate_DR': 3, 'Severe': 4}
idx_to_label_map = {idx: label for label, idx in label_to_idx_map.items()}




def label_transforms(label) -> int:
    label_to_idx_map = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Proliferate_DR': 3, 'Severe': 4}

    return label_to_idx_map[label]

def read_image(self, file_path, mode, resize=(256, 256), grayscale=False):
    print("Trying to open:", file_path)

    # Implement your image reading logic here
    image = Image.open(file_path)

    if grayscale:
        image = image.convert("L")

    image = Image.open(file_path)
    # print(image.size)
    height,width=image.size
    if height==width:
        pass
    else:
        
        if mode=='zoom':
            # left=0
            # right=0
            # upper=0
            # lower=0
            if height<width:
                diff=width-height
                left=diff//2
                right=width-diff//2
                upper=0
                lower=height
            elif width<height:
                diff=height-width
                left=0
                right=width
                upper=diff//2
                lower=height-upper


            new_image=image.crop((left,upper,right,lower))
        elif mode=='padding':
        
            image=ImageOps.pad(image, size=(256,256), centering=(0.5, 0.5))
            
    new_image=image.resize(resize)
    img_array = np.asarray(new_image)
    return img_array
# Transforms
def image_transforms(file_name, label) -> np.ndarray:
    """ this function extract the exact image from the image file name and label provided
    file_name:string,label:string
    """
    file_path = os.path.join(data_root, label, file_name)
    array = read_image(file_path, "padding", grayscale=True)
    flatten_image = array.flatten()
    return flatten_image
            

def label_to_idx(label:str):
    """convert label name to index
	for example: if my dataset consists of three labels (aom, csom, myringosclerosis,normal)
	this function should return 0 for aom, 1 for csom, 2 for myringosclerosis,normal
	"""
    """label represent class name in otitis media and index represent the corresponding class

    Raises:
        KeyError: _description_

    Returns:
        _type_: _description_
    """
    if label not in label_to_idx_map:
        raise KeyError(f"label not define . Defined label are:{label_to_idx_map.keys()}")
    return label_to_idx_map[label]
    
def idx_to_label(idx:int):
        """ similiar as label_to_idx but opposite I.e. take the index and return the string label """
        try:
            return idx_to_label_map[idx]
        except KeyError:
            raise KeyError(f"Label not found. Try one of these: {idx_to_label_map.keys()}")
    


