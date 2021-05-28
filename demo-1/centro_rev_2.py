from PIL import Image, ImageFilter, ImageDraw
import turtle
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageFilter import (BLUR, CONTOUR, DETAIL, EDGE_ENHANCE,
                             EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES,
                             SMOOTH, SMOOTH_MORE, SHARPEN)

################# Funciones
def draw_rct(dato1, dato2, imagen):
    draw = ImageDraw.Draw(imagen)
    draw.rectangle((dato1, dato2, (dato1*2), (dato2*2)),
                   fill=None,
                   outline=(0, 175, 120))
    return imagen

def cut_mid(dato1, dato2, im):
    corte=im.crop((dato1, dato2, (dato1*2), (dato2*2)))
    corte.save('Corte.bmp', "BMP", quality=300)
    return corte

def center(centro):
    acumula=[]
    color=(51,255,251)
    color2=(111,100,251)
    orig=centro
    dato=centro.getextrema()
    dato2=dato[0]
    dato3=dato2[1]
    print(dato3)
    for m in range(centro.height):
        for n in range(centro.width):  # extrae los valores de todos los pixel
            d1=centro.getpixel((n,m))
            d2=d1[0]
            if d2 > (dato3-100):
                centro.putpixel((n,m), color)
                acumula.append((n,m))
                     
    centro.save('circ_central.bmp', "BMP", quality=300)        
    return centro 
#############################

imorig = Image.open("Ag_Nano1.bmp")
im = Image.open("Ag_Nano1.bmp") #Ag_Nano1.bmp Ag_Nano2.bmp  Au_Nano_1.bmp  Au_Nano_2.bmp  Fe3O4_Nano1.bmp
print('imagen 1',im.size, im.mode, im.format)
px = im.load()

w= im.size[0]
h= im.size[1]
wid=round(w/3)
hei=round(h/3)
print(wid,hei)

######################### dibujo seccion central
im=draw_rct(wid,hei,im)
plt.imshow(im)
plt.show()
####################### corte seccion revisada
corte= cut_mid(wid, hei, im)
plt.imshow(corte)
plt.show()
##########################  pinta lo que identifico mas intenso en blanco
centro= Image.open("Corte.bmp")
#pix = centro.load()
pix_central=center(centro)
plt.imshow(pix_central)
plt.show()

###################################### GRÁFICO 
rows = 2
columns = 2
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(imorig)
plt.axis('off')
plt.title("First")
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
# showing image
plt.imshow(im)
plt.axis('off')
plt.title("Second")
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
# showing image
plt.imshow(corte)
plt.axis('off')
plt.title("Third")
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
# showing image
plt.imshow(centro)
plt.axis('off')
plt.title("Fourth")
plt.show()
