from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import os
from unet_model import UNet, RedUnet
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import pandas as pd
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

#Caracteristicas que aplicara nuestro modelo, tamaño de imagen, bath_size, directorios de las imagenes y csv
img_height = 256
img_width = 256
img_size = (img_height, img_width)
train_path = 'train'
val_path = 'valid'
test_path = 'test'
save_path = Path('results')
version = 'base'
model_name = 'unet_model.hdf5'
model_weights_name = 'unet_weight_model.hdf5'
test_num = len(os.listdir('test/img'))
BATCH_SIZE = 16  # 16
image_dir = "train_all"
train_data = pd.read_csv("df_even.csv")
skf = StratifiedKFold(n_splits=10, shuffle=True)
Y = train_data['cancer']


def df_train_generator(
    aug_dict,
    batch_size,
    dataframe,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode='grayscale',
    mask_color_mode='grayscale'
):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_dataframe(
        dataframe,
        directory=image_folder,
        x_col="image_fname",
        y_col="cancer",
        class_mode=None,
        shuffle=True,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=image_color_mode,
        seed=1
    )
    mask_generator = mask_datagen.flow_from_dataframe(
        dataframe,
        directory=mask_folder,
        x_col="image_fname",
        y_col="cancer",
        class_mode=None,
        shuffle=True,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=mask_color_mode,
        seed=1
    )

    train_g = zip(image_generator, mask_generator)
    for (img, mask) in train_g:
        img, mask = adjust_data(img, mask)
        yield img, mask

#normalización de las imágenes
def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)

#función para el calculo del DICE
def dice_coef(y_true, y_pred):
    smooth = 1
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return (2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)

#función para el calculo del loss basada en el DICE
def dice_coef_loss(y_true, y_pred):
    smooth = 1
    return 1-dice_coef(y_true, y_pred)

#función para el calculo del IoU
def iou(y_true, y_pred):
    smooth = 1
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


if __name__ == "__main__":

    train_generator_args = dict(rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')

    fold = 1

    learning_rate = 1e-4  #Basado en la literatura investigada

    for train_idx, val_idx in skf.split(train_data, Y):
        EPOCHS = 1000
        BATCH_SIZE = 16
        learning_rate = 1e-4

        model = UNet(input_size=(img_height, img_width, 1))

        #Adam optimizer
        decay_rate = learning_rate / EPOCHS
        opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
        model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef])

        #Indica el modelo y el fold en el que se encuentra
        print(f"Currently training on Model N1, fold {fold}")
        training_data = train_data.iloc[train_idx]
        train_num = len(training_data)
        validation_data = train_data.iloc[val_idx]
        val_num = len(validation_data)
        steps_per_epoch = train_num // BATCH_SIZE
        steps_val = val_num // BATCH_SIZE
        train_gen = df_train_generator(
            aug_dict=train_generator_args,
            batch_size=BATCH_SIZE,
            dataframe=training_data,
            image_folder='train_all/img',
            mask_folder='train_all/mask',
            target_size=img_size
        )

        val_gen = df_train_generator(
            aug_dict=dict(),
            batch_size=BATCH_SIZE,
            dataframe=validation_data,
            image_folder='train_al/img',
            mask_folder='train_all/mask',
            target_size=img_size
        )

        filepath = str(fold) + 'weightsUNET.{epoch:02d}-{val_loss:.2f}.hdf5'
        #Callback para que los "mejores pesos" sean guardados para su posterior uso
        callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                     save_best_only=False, save_weights_only=False,
                                     mode='auto', period=100)]

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=steps_val
        )

        histDF = pd.DataFrame.from_dict(history.history)
        histDF.to_csv('results/historyCSVUNET_fold_' + str(fold) + '.csv')

        model.save_weights(str(fold) + 'unet_weight_model.hdf5')
        model.save(str(fold) + 'UNetTest.h5')

        fold += 1
