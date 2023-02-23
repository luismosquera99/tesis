import os
from unet_model import UNet, RedUnet
from munet import ImageDataGenerator, adjust_data, dice_coef_loss, iou, dice_coef
import PIL.Image as Image
from pathlib import Path as Path
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as trans
from keras.optimizers import *

#definimos las dimensiones de la imagen
img_height = 256
img_width = 256
img_size = (img_height, img_width)
BATCH_SIZE = 1
test_path = "test"
test_path_img = "test/img"
test_num = len(os.listdir('test/img'))

#Estipulamos las caracteristicas de nuestro train
def train_generator(
    aug_dict,
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode='grayscale',
    mask_color_mode='grayscale'
):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1
    )
    train_g = zip(image_generator, mask_generator)
    for (img, mask) in train_g:
        img, mask = adjust_data(img, mask)
        yield img, mask

#Guardamos resultados
def save_results(
    npyfile,
):
    #Realizamos un Overlap
    Img = []
    Msk = []
    new_images = []
    predicted_names = []
    names = []
    l_img = os.listdir("test/img")
    l_msk = os.listdir("test/mask")

    #Recorremos nuestros directorios, para proceder a realizar un apppend
    for img in l_img:
        img1 = Image.open("test/img/" + img)
        names.append(img)
        img1 = img1.copy()
        Img.append(img1)
    for msk in l_msk:
        msk1 = Image.open("test/mask/" + msk)
        msk1 = msk1.copy()
        Msk.append(msk1)

    #Estipulamos los paths donde se guardaran
    cont = 0
    predicted_path = Path("predicted_masks")
    first_overlap_path = predicted_path / "first_overlap"
    only_predicted_path = predicted_path / "only_predicted"
    final_path = predicted_path / "final"
    if not predicted_path.exists():
        predicted_path.mkdir()
        first_overlap_path.mkdir()
        only_predicted_path.mkdir()
        final_path.mkdir()
    print(names)
    for img, msk, name in zip(Img, Msk, names):
        new_img = Image.blend(img.convert("L"), msk, 0.5)
        new_images.append(new_img)
        new_img.save(first_overlap_path / name)
        cont += 1
    #normalizamos la imagen y mantenmos el formato .png
    for i, (item, name) in enumerate(zip(npyfile, l_img)):
        img = normalize_mask(item)
        img = (img * 255).astype('uint8')
        name = f'{l_img[i].strip(".png")}_predict.png'
        predicted_names.append(name)
        io.imsave(os.path.join(only_predicted_path, name), img)
    #Indicamos que la imagen esta tipo RGBA
    for img, name in zip(Img, predicted_names):
        predicted_mask = Image.open(only_predicted_path / name).convert("RGBA")
        data = np.array(predicted_mask)
        red, green, blue, alpha = data.T

        white_areas = (red == 255) & (blue == 255) & (green == 255)
        data[..., :-1][white_areas.T] = (255, 0, 0)
        im2 = Image.fromarray(data).convert("RGBA")
        new_img = Image.blend(img.convert("RGBA"), im2, 0.6)
        new_img.save(final_path / name)

#Colocamos las caracteristicas de nuestro Test
def test_generator(
    test_path,
    target_size,
    as_gray=True
):
    l = os.listdir(test_path)
    for i in l:
        img = io.imread(os.path.join(test_path, i), as_gray=as_gray)
        img = square_image(img)
        img = reshape_image(img, target_size)
        yield img

def square_image(img, random=None):
    size = max(img.shape[0], img.shape[1])
    new_img = np.zeros((size, size),np.float32)
    ax, ay = (size - img.shape[1])//2, (size - img.shape[0])//2

    if random and not ax == 0:
        ax = int(ax * random)
    elif random and not ay == 0:
        ay = int(ay * random)

    new_img[ay:img.shape[0] + ay, ax:ax+img.shape[1]] = img
    return new_img

#Realizamos una función para normalizar la mascara
def normalize_mask(mask):
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

def reshape_image(img, target_size):
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img

#Indicamos la imagen
def show_image(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    unet = RedUnet(
        (img_width, img_height, 1),
    )

    unet.load_weights("checkpoints/unet_red/10weightsUNETRED.100--0.96.hdf5")
    learning_rate = 1e-4
    EPOCHS = 1000

    decay_rate = learning_rate / EPOCHS
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    unet.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef])
    unet.summary()

    test_gen = train_generator(
        aug_dict=dict(),
        batch_size=BATCH_SIZE,
        train_path=test_path,
        image_folder='img',
        mask_folder='mask',
        target_size=img_size
    )

    steps = test_num//BATCH_SIZE
    unet.evaluate(test_gen, steps=steps)
    test_gen = test_generator(test_path_img, img_size)
    results = unet.predict_generator(test_gen, test_num, verbose=1)
    save_results(results)