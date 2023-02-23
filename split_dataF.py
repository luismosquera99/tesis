from PIL import ImageTk, Image
from pathlib import Path
import os
import csv

#Crear el archivo csv que permitira realizar 10-fold cross-validation
if __name__ == '__main__':
    new_path = 'df_even.csv'
    f = open(new_path, 'w')
    columns = ["image_fname", "cancer"]
    writer = csv.writer(f)
    writer.writerow(columns)

    w = len(os.listdir('fdataset_cancer_imgFNLungFinal'))
    x = len(os.listdir('fdataset_cancer_maskLungFinal'))
    y = len(os.listdir('fdataset_normal_imgFNLungFinal'))
    z = len(os.listdir('fdataset_normal_maskLungFinal'))

    train_path = Path('train_all')
    image_path = train_path / 'img'
    label_path = train_path / 'mask'
    if not train_path.exists():
        train_path.mkdir()
        image_path.mkdir()
        label_path.mkdir()

    test_path = Path('test')
    image_path = test_path / 'img'
    label_path = test_path / 'mask'
    if not test_path.exists():
        test_path.mkdir()
        image_path.mkdir()
        label_path.mkdir()

    vec = os.listdir('fdataset_cancer_imgFNLungFinal')
    vec2 = os.listdir('fdataset_normal_imgFNLungFinal')

#Crear los nuevos directorios con las imagenes divididas para realizar el training y test
    for i in range(w):
        if i == '.DS_Store':
            continue

        image = Image.open('fdataset_cancer_imgFNLungFinal/' + vec[i])
        if vec[i].__contains__("CHNC"):
            mask = Image.open('fdataset_cancer_maskLungFinal/' + vec[i].replace(".png", "") + "_mask.png")
        else:
            mask = Image.open('fdataset_cancer_maskLungFinal/' + vec[i])
        if i < x * 0.1:
            image.save('test/img/' + str(i) + '.png')
            mask.save('test/mask/' + str(i) + '.png')
        else:
            image.save('train_all/img/' + str(i) + '.png')
            mask.save('train_all/mask/' + str(i) + '.png')
            writer.writerow([str(i) + '.png', str(0)])

    for i in range(y):
        if i == '.DS_Store':
          continue

        image = Image.open('fdataset_normal_imgFNLungFinal/' + vec2[i])
        if vec2[i].__contains__("CHNC"):
            mask = Image.open('fdataset_normal_maskLungFinal/' + vec2[i].replace(".png", "") + "_mask.png")
        else:
            mask = Image.open('fdataset_normal_maskLungFinal/' + vec2[i])
        if i < y * 0.1:
            image.save('test/img/' + str(w+i) + '.png')
            mask.save('test/mask/' + str(w+i) + '.png')
        else:
            image.save('train_all/img/' + str(w+i) + '.png')
            mask.save('train_all/mask/' + str(w+i) + '.png')
            writer.writerow([str(w+i) + '.png', str(1)])

    f.close()

