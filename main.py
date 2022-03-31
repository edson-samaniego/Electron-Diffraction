import cv2
import random
import numpy as np
import pandas as pd
from math import sqrt, fabs, ceil, floor, atan2, degrees
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cen_det
import red_resolucion
import spots
import gradiente
import scale
from time import time
import re
import array 

img = cv2.imread('Ag_Nano1.bmp')#Ag_Nano1.bmp Ag_Nano2.bmp Au_Nano_1.bmp
                                 #Au_Nano_2.bmp  Fe3O4_Nano1.bmp
#plt.imshow(img)
#plt.show()
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
#plt.imshow(grad2), plt.title('restando gradiente')
#plt.show()
#####################################################################
antes = time()
gradiente = cv2.circle(gradiente, (mt1[0][1], mt1[0][0]), 130, (0,0,0), -1)
contornos, posiciones = spots.center(gradiente)
Tiempo= (time() - antes)
print("Tiempo deteccion spots: ", Tiempo,"segundos")
#plt.imshow(contornos), plt.title('contornos')
#plt.show()
#####################################################################
distancias=[]
diametros=[]
angulos=[]
Y, X = [], []
img5_2=img5.copy()
img5_3= img5.copy()
for s in posiciones:
    Y.append(s[0])
    X.append(s[1])
    (dy, dx)=(s[0]-mt1[0][0], s[1]-mt1[0][1])
    angle =  degrees(atan2(float(dy), float(dx)))
    if angle < 0:
            angle += 180
    angulos.append(angle)
    ECL=sqrt((s[0]-mt1[0][0])**2+(s[1]-mt1[0][1])**2)##radios
    ECL2=(sqrt((s[0]-mt1[0][0])**2+(s[1]-mt1[0][1])**2))*2#diametros
    distancias.append(ECL)
    diametros.append(ECL2)
    
    img5_3 = cv2.line(img5_3, (s[1],s[0]), (mt1[0][1],mt1[0][0]), (0,255,0), 1)
    
    img5 = cv2.line(img5, (s[1],s[0]), (mt1[0][1],mt1[0][0]), (0,255,0), 1)
    st = ','.join(map(str, (s[0],s[1])))
    img5 = cv2.putText(img5, (st), (s[1],s[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 1, cv2.LINE_AA)
    img6 = cv2.line(contornos, (s[1],s[0]), (mt1[0][1],mt1[0][0]), (0,255,0), 1)
    
plt.imshow(img5_3)
plt.show()
plt.subplot(121), plt.imshow(img5), plt.title('dist original')
plt.subplot(122), plt.imshow(img6), plt.title('dist sin ruido')
plt.show()

for r in distancias:#para trazar circulos se redonde distancia(no basar tanto)
    img7 = cv2.circle(img5, (mt1[0][1], mt1[0][0]), round(r), (0,0,255), 1)
#trazar un anillo o diametro y leer su histograma para ver picos 
#plt.subplot(121), plt.imshow(img6), plt.title('dist sin ruido')
#plt.subplot(122), plt.imshow(img7), plt.title('con anillos')
#plt.show()
    
print("#####################################################################")
print("distancias",distancias)
antes = time()
escala, conversion= scale.conversion(img4)
Tiempo= (time() - antes)
print("Escala:",escala,"Conversion:",conversion)
print("Tiempo escala y conversion: ", Tiempo,"segundos")

radio_escala=[r / escala for r in distancias]
#print("Nuevos radios con la escala de imagen",radio_escala)

dividido= [2*(1/(i/2)) for i in radio_escala]
dividido2= [2 / s for s in radio_escala]
#print("Primer dato ya dividido",dividido)
#print("Primer dato ya dividido",dividido2)

#print("tipo de datos:",type(conversion))
#print("cantidad de datos:",len(conversion))
#print("primer dato que es",(type(conversion[0])))

conversion= str(conversion)
p= re.findall('[0-9]+',conversion)
#print(p)
#print("despues de findall numeros tipo:",type(p))
#print("largo de p:",len(p))

q= re.findall('[a-z]+',conversion)
#print(q)
#print("despues de findall letras tipo:",type(q))
#print("largo de q:",len(q))

r= re.findall('/',conversion)
#print(r[0])
#print("despues de findall diagonal tipo:",type(r))
#print("largo de r:",len(r))
h, k, l=[],[],[]
#print("np",(np.arange(dividido)))
#print("x",list(dividido))

#plt.scatter((np.arange(len(dividido))),dividido, color='red',label='radio= 2(1/(radio/2))')
#plt.scatter((np.arange(len(dividido2))),dividido2, color='blue',label='radio= 2/radio')
#plt.show()
#####################################################################
#datos={'h': '',
 #      'k':'',
  #     'l':'',
   #    'Radio px-px': distancias,
    #   'Angulo': angulos,
     #  'Diametros px-px': diametros,
      # 'radio escala': radio_escala,
       #'radio= 2(1/(radio/2))': dividido,
      # 'radio= 2/radio': dividido2}
#df = pd.DataFrame(datos)
#df = df.sort_values(by='Radio px-px')
#print(df)
#from openpyxl.workbook import Workbook
#file_name = 'spots.xlsx'
#df.to_excel(file_name)
#####################################################################
#####################################################################
datos={'h': '',
       'k':'',
       'l':'',
       'Radio px-px': distancias,
       'X': X,
       'Y':Y,
       'Angulo': angulos,
       'radio escala': radio_escala,
       'radio= 2/radio': dividido2}
df = pd.DataFrame(datos)
df = df.sort_values(by='Radio px-px')
print(df)
from openpyxl.workbook import Workbook
file_name = 'spots.xlsx'
df.to_excel(file_name)
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
