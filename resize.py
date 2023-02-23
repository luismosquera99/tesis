# from sklearn.model_selection import train_test_split
# import pandas as pd
from PIL import ImageTk, Image
from pathlib import Path
import os
import cv2
import numpy as np
import sys
import albumentations as A

#Directorios
directory_img = 'fdataset_cancer_imgFNLung/'
new_directory_img = 'fdataset_cancer_imgFNLungFinal/'

#Crear los directorios nuevos para guardar las nuevas imagenes
if not Path(new_directory_img).exists():
    Path(new_directory_img).mkdir()

directory_mask = 'fdataset_cancer_maskLung/'
new_directory_mask = 'fdataset_cancer_maskLungFinal/'

if not Path(new_directory_mask).exists():
    Path(new_directory_mask).mkdir()

directory_normal = 'fdataset_normal_imgFNLung/'
new_directory_normal = 'fdataset_normal_imgFNLungFinal/'

if not Path(new_directory_normal).exists():
    Path(new_directory_normal).mkdir()

directory_normal_mask = 'fdataset_normal_maskLung/'
new_directory_normal_mask = 'fdataset_normal_maskLungFinal/'

if not Path(new_directory_normal_mask).exists():
    Path(new_directory_normal_mask).mkdir()

if __name__ == '__main__':

#Navegar por los directorios, transformar el tamaño de las imágenes
    for i, j in zip(os.listdir(directory_img), os.listdir(directory_mask)):
        transform = A.Compose([
            A.Resize(height=256, width=256)
        ])

        image_cancer = np.array(Image.open(directory_img + i))
        mask_cancer = np.array(Image.open(directory_mask + j))

        aug = transform(image=image_cancer, mask=mask_cancer)

        image_cancer = aug['image']
        mask_cancer = aug['mask']

        image = Image.fromarray(image_cancer)
        mask = Image.fromarray(mask_cancer)

        image.save(new_directory_img + i)
        mask.save(new_directory_mask + j)


    # Para resize de la img normales

for h in (os.listdir(directory_normal)):
    transform = A.Compose([
        A.Resize(height=256, width=256)
    ])

    image_normal = np.array(Image.open(directory_normal + h))

    aug = transform(image=image_normal)

    image_normal = aug['image']

    image = Image.fromarray(image_normal)

    image.save(new_directory_normal + h)

    # Para resize de la mask normales

for k in (os.listdir(directory_normal_mask)):
    transform = A.Compose([
        A.Resize(height=256, width=256)
    ])

    image_normal = np.array(Image.open(directory_normal_mask + k))

    aug = transform(image=image_normal)

    image_normal = aug['image']

    image = Image.fromarray(image_normal)

    image.save(new_directory_normal_mask + k)
