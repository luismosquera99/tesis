import os

from PIL import Image

#Función para realizar un overlap
test_path = "test"

for img in os.listdir("test/img"):
    img1 = Image.open("test/img/" + img)
    img1 = img1.copy()
    mask1 = Image.open("test/mask/" + img)
    mask1 = mask1.copy()

    imgt = Image.blend(img1, mask1, 0.5)

    imgt.save("overlap_inicial/" + img)