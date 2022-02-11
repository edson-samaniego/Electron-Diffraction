import cv2
import random
import numpy as np
from math import sqrt, fabs, ceil, floor
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cen_det
import red_resolucion
import spots
import gradiente
import scale
from time import time
import array 

img = cv2.imread('Ag_Nano1.bmp')#Ag_Nano1.bmp Ag_Nano2.bmp Au_Nano_1.bmp
                                 #Au_Nano_2.bmp  Fe3O4_Nano1.bmp
plt.imshow(img)
plt.show()
h=img.shape[0]
w=img.shape[1]
####################### diagonal ###############################
muestras=100
img2=img.copy()
img3=img.copy()
img4=img.copy()
img5=img.copy()
esquina =(500,0)
centro =(round(h/2),round(w/2))
diagonal=[]
posit=[]
img2 = cv2.line(img2, esquina, centro, (0,255,0), 1)
for m in range(round(h/2)+5):
    for n in range(round(w/2)+5):
        pix= img2[m,n]
        if (pix[0],pix[1],pix[2]) == (0,255,0):
            diagonal.append(img[m,n][0])
diferencia=(len(diagonal)-muestras)
mitad=round(diferencia/2)
diagonal=diagonal[mitad+(round(mitad/2)):]
diagonal=diagonal[:muestras]
poscnt=diagonal
mt1=[]
mt2=[]
##########################################################
for saltos in 1, 2, 3, 4:
    rangos= random.sample(poscnt,round(len(poscnt)/saltos))
    #print(rangos)
    print("############", len(rangos), "Muestras analizadas ##############")
    antes = time()
    C_haz1= cen_det.center(img, rangos)
    mt1.append(C_haz1)
    print(C_haz1,"centro sin modificacion de imagen (cen_det)")
    Tiempo= (time() - antes)
    print("Tiempo de funcion cen_det: ", Tiempo ,"segundos")
    #######################################################
    #antes = time()
    #C_haz2= red_resolucion.center(img, rangos)
    #mt2.append(C_haz2)
    #print(C_haz2,"centro reduciendo al 50% (red_resolucion)")
    #Tiempo= (time() - antes)
    #print("Tiempo de funcion red_resolucion: ", Tiempo,"segundos")

#####################################################################
antes = time()
gradiente= gradiente.resta(img3,mt1[0])
Tiempo= (time() - antes)
print("Tiempo de gradiente: ", Tiempo,"segundos")
grad2 = cv2.bitwise_not(gradiente)
plt.imshow(grad2), plt.title('restando gradiente')
plt.show()
#####################################################################
antes = time()
gradiente = cv2.circle(gradiente, (mt1[0][1], mt1[0][0]), 130, (0,0,0), -1)
contornos, posiciones = spots.center(gradiente)
Tiempo= (time() - antes)
print("Tiempo deteccion spots: ", Tiempo,"segundos")
plt.imshow(contornos), plt.title('contornos')
plt.show()
#####################################################################
distancias=[]
diametros=[]
for s in posiciones:
    ECL=sqrt((s[0]-mt1[0][0])**2+(s[1]-mt1[0][1])**2)
    ECL2=(sqrt((s[0]-mt1[0][0])**2+(s[1]-mt1[0][1])**2))*2
    distancias.append(ECL)
    diametros.append(ECL2)
    img5 = cv2.line(img5, (s[1],s[0]), (mt1[0][1],mt1[0][0]), (0,255,0), 1)
    img6 = cv2.line(contornos, (s[1],s[0]), (mt1[0][1],mt1[0][0]), (0,255,0), 1)
print(diametros)
plt.subplot(121), plt.imshow(img5), plt.title('dist original')
plt.subplot(122), plt.imshow(img6), plt.title('dist sin ruido')
plt.show()
for r in distancias:
    img7 = cv2.circle(img5, (mt1[0][1], mt1[0][0]), round(r), (0,0,255), 1)
plt.subplot(121), plt.imshow(img6), plt.title('dist sin ruido')
plt.subplot(122), plt.imshow(img7), plt.title('con anillos')
plt.show()
#####################################################################
antes = time()
escala, conversion= scale.conversion(img4)
Tiempo= (time() - antes)
print("Escala:",escala,"Conversion:",conversion)
print("Tiempo escala y conversion: ", Tiempo,"segundos")
#####################################################################
print("##### TERMINO! #####")



############# Esta seccion para utilizar el histograma #########
#gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.hist(gray.ravel(), 256, [0,256])
#counts, bins, bars = plt.hist(gray.ravel(), 256, [0,256])
#plt.show()
######### Elimina las intensidad que no hay ##############
#counts = counts.tolist()
#poscnt=[]
#for s in range(len(counts)):
 #   L= counts[s]
  #  if L != 0:
   #     poscnt.append(s)
#if len(poscnt)>= 220:
 #   medio=round(len(poscnt)/2)
  #  poscnt= poscnt[(medio-110):(medio+10)]
#print(poscnt)
