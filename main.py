#from sklearn.model_selection import train_test_split
#import pandas as pd
import numpy as np
from PIL import ImageTk, Image
from pathlib import Path
import os
#import cv2
#import numpy as np

if __name__ == '__main__':

#Directorios del Dataset original
    directory_img = 'dataset_original/images/'
    directory_mask = 'dataset_original/masks/'

#Directorios a ser creados
    new_path_img = 'fdataset_normal_imgFNLung/'
    new_path_mask = 'fdataset_normal_maskLung/'
    new_path_imgC = 'fdataset_cancer_imgFNLung/'
    new_path_maskC = 'fdataset_cancer_maskLung/'


    if not Path(new_path_img).exists():
        Path(new_path_img).mkdir()
    if not Path(new_path_mask).exists():
        Path(new_path_mask).mkdir()
    if not Path(new_path_imgC).exists():
            Path(new_path_imgC).mkdir()
    if not Path(new_path_maskC).exists():
            Path(new_path_maskC).mkdir()

#Navegación por el directorio inicial, función para guardar imagenes en los nuevos directorios
    for pic_num in os.listdir(directory_img):
        if pic_num.__contains__('_0.'):
            img = Image.open(directory_img + pic_num)
            img.save(new_path_img + pic_num)

        if pic_num.__contains__('_1.'):
            img = Image.open(directory_img + pic_num)
            img.save(new_path_imgC + pic_num)

    for mask_num in os.listdir(directory_mask):
        if mask_num.__contains__('_0.') or mask_num.__contains__('_0_'):
            img = Image.open(directory_mask + mask_num)
            img.save(new_path_mask + mask_num)

        if mask_num.__contains__('_1.') or mask_num.__contains__('_1_'):
            img = Image.open(directory_mask + mask_num)
            img.save(new_path_maskC + mask_num)

