import os
import pandas as pd
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, \
    MaxPooling2D, Dropout, concatenate, UpSampling2D, add
from keras.callbacks import ModelCheckpoint
from keras.models import Model


img_height = 256
img_width = 256
img_size = (img_height, img_width)
train_path = '../../Downloads/train'
val_path = 'valid'
test_path = 'test'
save_path = 'results'
version = 'base'
model_name = 'unet_model.hdf5'
model_weights_name = 'unet_weight_model.hdf5'
train_num = len(os.listdir('train/img'))
val_num = len(os.listdir('valid/img'))
test_num = len(os.listdir('test/img'))
BATCH_SIZE = 32  # 16
flag = False
smooth = 1

#Realizamos una función para normalizar la imagen
def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)

#Realizamos una función para calcular el Dice Coeffiecent
def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return (2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)

#Realizamos una función para calcular el loss en función al DICE
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

#Realizamos una función para calcular el IoU
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


if __name__ == "__main__":

    def augment_image_and_mask(augmentation, image, mask):
        aug_image_dict = augmentation(image=image, mask=mask)
        image_matrix = aug_image_dict['image']
        mask_matrix = aug_image_dict['mask']
        return image_matrix, mask_matrix

    train_generator_args = dict(rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')

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
        """ Image Data Generator
        Function that generates batches of data (img, mask) for training
        from specified folder. Returns images with specified pixel size
        Does preprocessing (normalization to 0-1)
        """
        # no augmentation, only rescaling
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

    # generates training set
    train_gen = train_generator(
        aug_dict=train_generator_args,
        batch_size=BATCH_SIZE,
        train_path=train_path,
        image_folder='img',
        mask_folder='mask',
        target_size=img_size
    )

    val_gen = train_generator(
        aug_dict=dict(),
        batch_size=BATCH_SIZE,
        train_path=val_path,
        image_folder='img',
        mask_folder='mask',
        target_size=img_size
    )


    #Definimos nuestro modelo Unet
    def unet(input_size=(256, 256, 3)):
        inputs = Input(input_size)

        conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
        bn1 = Activation('relu')(conv1)
        conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
        bn1 = BatchNormalization(axis=3)(conv1)
        bn1 = Activation('relu')(bn1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

        conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
        bn2 = Activation('relu')(conv2)
        conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
        bn2 = BatchNormalization(axis=3)(conv2)
        bn2 = Activation('relu')(bn2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

        conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
        bn3 = Activation('relu')(conv3)
        conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
        bn3 = BatchNormalization(axis=3)(conv3)
        bn3 = Activation('relu')(bn3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

        conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
        bn4 = Activation('relu')(conv4)
        conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
        bn4 = BatchNormalization(axis=3)(conv4)
        bn4 = Activation('relu')(bn4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

        conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
        bn5 = Activation('relu')(conv5)
        conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
        bn5 = BatchNormalization(axis=3)(conv5)
        bn5 = Activation('relu')(bn5)

        up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), padding='same')(up6)
        bn6 = Activation('relu')(conv6)
        conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
        bn6 = BatchNormalization(axis=3)(conv6)
        bn6 = Activation('relu')(bn6)

        up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), padding='same')(up7)
        bn7 = Activation('relu')(conv7)
        conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
        bn7 = BatchNormalization(axis=3)(conv7)
        bn7 = Activation('relu')(bn7)

        up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), padding='same')(up8)
        bn8 = Activation('relu')(conv8)
        conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
        bn8 = BatchNormalization(axis=3)(conv8)
        bn8 = Activation('relu')(bn8)

        up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), padding='same')(up9)
        bn9 = Activation('relu')(conv9)
        conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
        bn9 = BatchNormalization(axis=3)(conv9)
        bn9 = Activation('relu')(bn9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

        return Model(inputs=[inputs], outputs=[conv10])


    EPOCHS = 800
    BATCH_SIZE = 32
    learning_rate = 1e-4


    model = unet(input_size=(img_height, img_width, 3))

    decay_rate = learning_rate / EPOCHS
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef])

    tempModelName = 'unet' + version + '.hdf5'
    callbacks = [ModelCheckpoint(tempModelName, verbose=1, save_best_only=True)]

    history = model.fit(train_gen,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=val_num // BATCH_SIZE)

    model.save_weights(model_weights_name)
    histDF = pd.DataFrame.from_dict(history.history)
    histDF.to_csv('historyCSV')
    model.save('UNet1Test.h5')

    plt.plot(history.history["loss"], label='Training')
    plt.plot(history.history["val_loss"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("loss_fig.png")

    plt.clf()

    plt.plot(history.history["dice_coef"], label='Training')
    plt.plot(history.history["val_dice_coef"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("dice_fig.png")

    plt.clf()

    plt.plot(history.history["iou"], label='Training')
    plt.plot(history.history["val_iou"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("iou_fig.png")