import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
from statistics import mean, stdev
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold


def plot_models(history, path):
    plt.plot(history["loss"], label='Training')
    plt.plot(history["val_loss"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.axvline(x=100, color='g', linestyle=(0, (5, 1)), label='best model')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "loss_fig.png")

    plt.clf()

    plt.plot(history["dice_coef"], label='Training')
    plt.plot(history["val_dice_coef"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation DICE')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "dice_fig.png")

    plt.clf()

    plt.plot(history["iou"], label='Training')
    plt.plot(history["val_iou"], label='Validation')

    # Add in a title and axes labels
    plt.title('Training and Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')

    # Display the plot
    plt.legend(loc='best')
    plt.savefig(path + "iou_fig.png")

    plt.clf()


def compress(path, new_path):
    DF = []
    for cv in os.listdir(path):
        if cv == '.DS_Store':
            continue
        DF.append(pd.read_csv(path + cv))
    cont = 1
    f = open(new_path, 'w')
    epochs = 1000
    columns = ["Epochs", "loss", "std_loss", "iou", "std_iou", "dice_coef", "std_dice",
               "val_loss", "std_val_loss", "val_iou", "std_val_iou", "val_dice_coef", "std_val_dice"]
    writer = csv.writer(f)
    writer.writerow(columns)
    for i in range(epochs):
        I = []
        D = []
        L = []
        vI = []
        vD = []
        vL = []
        for d in DF:
            L.append(d.iloc[i]['loss'])
            I.append(d.iloc[i]['iou'])
            D.append(d.iloc[i]['dice_coef'])
            vL.append(d.iloc[i]['val_loss'])
            vI.append(d.iloc[i]['val_iou'])
            vD.append(d.iloc[i]['val_dice_coef'])
        avgL = mean(L)
        avgI = mean(I)
        avgD = mean(D)
        avgvL = mean(vL)
        avgvI = mean(vI)
        avgvD = mean(vD)

        sL = stdev(L)
        sI = stdev(I)
        sD = stdev(D)
        svL = stdev(vL)
        svI = stdev(vI)
        svD = stdev(vD)

        r = [str(cont), str(avgL), str(sL), str(avgI), str(sI), str(avgD), str(sD),
             str(avgvL), str(svL), str(avgvI), str(svI), str(avgvD), str(svD)]
        writer.writerow(r)
        cont += 1
    f.close()

def analyse(df, new_write):
    columns = ["Checkpoint (epoch)", "Learning rate", "Batch size", " train dice mean",
               "train dice std", "train iou mean", "train iou std", "train loss mean",
               "train loss std", " val dice mean", "val dice std", "val iou mean",
               "val iou std", "val loss mean", "val loss std"]
    # open the file in the write mode
    f = open(new_write, 'w')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(columns)

    cont = 0
    I = []
    D = []
    L = []
    vI = []
    vD = []
    vL = []
    for row in df.iterrows():
        L.append(float(row[1][1]))
        I.append(float(row[1][2]))
        D.append(float(row[1][3]))
        vL.append(float(row[1][4]))
        vI.append(float(row[1][5]))
        vD.append(float(row[1][6]))

        cont += 1
        if cont % 100 == 0:
            avgL = mean(L)
            avgI = mean(I)
            avgD = mean(D)
            avgvL = mean(vL)
            avgvI = mean(vI)
            avgvD = mean(vD)

            sL = stdev(L)
            sI = stdev(I)
            sD = stdev(D)
            svL = stdev(vL)
            svI = stdev(vI)
            svD = stdev(vD)

            r = [str(cont), "1e-4", "32", str(avgD), str(sD), str(avgI), str(sI), str(avgL), str(sL),
                 str(avgvD), str(svD), str(avgvI), str(svI), str(avgvL), str(svL)]

            writer.writerow(r)

    f.close()


if __name__ == "__main__":

    path = 'results/unet_normal/'
    visual_path = 'results/unet_normal_visual/'
    new_path_1 = 'results/unet_normal_visual/mean_history.csv'
    new_write = 'results/unet_normal_visual/table.csv'

    compress(path, new_path_1)
    plot_models(pd.read_csv(new_path_1, encoding='utf-8',  engine='python'), visual_path)

    path = 'results/unet_red/'
    visual_path = 'results/unet_red_visual/'
    new_path_2 = 'results/unet_red_visual/mean_history.csv'
    new_write = 'results/unet_red_visual/table.csv'

    compress(path, new_path_2)
    plot_models(pd.read_csv(new_path_2, encoding='utf-8',  engine='python'), visual_path)