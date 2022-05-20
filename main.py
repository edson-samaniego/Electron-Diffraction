import cv2
import random
import numpy as np
import pandas as pd
from math import sqrt, fabs, ceil, floor, atan2, degrees
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import rc
import cen_det
import spots
import gradiente
import scale
import read_cards
from time import time
import re
import array 

img = cv2.imread('Au_Nano_2.bmp')
h=img.shape[0]
w=img.shape[1]
####################### diagonal ###############################
muestras=200
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
for saltos in 1, 2:#, 4:
    rangos= random.sample(poscnt,round(len(poscnt)/saltos))
    #print(rangos)
    print("############", len(rangos), "Muestras analizadas ##############")
    antes = time()
    C_haz1= cen_det.center(img, rangos)
    mt1.append(C_haz1)
    print(C_haz1,"centro sin modificacion de imagen (cen_det)")
    Tiempo= (time() - antes)
    print("Tiempo de funcion cen_det: ", Tiempo ,"segundos")
antes = time()
gradiente= gradiente.resta(img3,mt1[0])
Tiempo= (time() - antes)
print("Tiempo de gradiente: ", Tiempo,"segundos")
grad2 = cv2.bitwise_not(gradiente)
#####################################################################
antes = time()
gradiente = cv2.circle(gradiente, (mt1[0][1], mt1[0][0]), 130, (0,0,0), -1)
contornos, posiciones, intensidades, secciones = spots.center(gradiente)
Tiempo= (time() - antes)
print("Tiempo deteccion spots: ", Tiempo,"segundos")
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
cv2.imwrite('dist_spots.png',img5_3)    
plt.imshow(img5_3)
plt.show()
for r in distancias:#para trazar circulos se redonde distancia(no basar tanto)
    img7 = cv2.circle(img5, (mt1[0][1], mt1[0][0]), round(r), (0,0,255), 1)
antes = time()
escala, conversion= scale.conversion(img4)
Tiempo= (time() - antes)
print("Escala:",escala,"Conversion:",conversion)
print("Tiempo escala y conversion: ", Tiempo,"segundos")
radio_escala=[r / escala for r in distancias]
dividido= [round((2 / s),4) for s in radio_escala]
conversion= str(conversion)
p= re.findall('[0-9]+',conversion)
q= re.findall('[a-z]+',conversion)
r= re.findall('/',conversion)
h, k, l=[],[],[]
#####################################################################
datos={'h': '',
       'k':'',
       'l':'',
       'X': X,
       'Y':Y,
       'dA': dividido,
       'I': intensidades}
df = pd.DataFrame(datos)
from openpyxl.workbook import Workbook
file_name = 'spots.xlsx'
df.to_excel(file_name)
tarjetas=['Ag.rtf']
pk_list_DRX= read_cards.DRX(tarjetas)
pk_list_PTC= read_cards.PTC_Lab('Ag')
data={'h': pk_list_DRX.h,
      'k': pk_list_DRX.k,
      'l': pk_list_DRX.l,
      'dA': pk_list_DRX.dA,
      'I': pk_list_DRX.I,
      'h': pk_list_PTC.h,
      'k': pk_list_PTC.k,
      'l': pk_list_PTC.l,
      'dA': pk_list_PTC.dA,
      'I': pk_list_PTC.I}
df_2 = pd.DataFrame(data)
file = 'Tarjeta.xlsx'
df_2.to_excel(file)
lista=df_2['dA']
indexar=[]
dif=[]
for numero in df['dA']: # ciclo recorre los spots encontrados 'dA'
    res=min(lista, key=lambda x:abs(x-numero))# el mas cercano en lista reportada
    diferencia=abs(numero-res)# diferencia entre encontrado y reportado
    if diferencia <= 0.007:# histeresis diferencia para que guarde en lista
        indexar.append((numero,res))# guarda el encontrado y el reportado
        dif.append(diferencia)
fila=[]
d_enc, d_rep=[], []
for e,r in indexar:
    d_enc.append(e)
    d_rep.append(r)
    value=r
    hkl=df_2.loc[df_2['dA'] == value]# busca el valor en la lista extrae fila
    H,K,L=hkl.iloc[0:,0],hkl.iloc[0:,1],hkl.iloc[0:,2]#de fila extrae hkl
    value2= e
    hkl2=df.loc[df['dA'] == value2]
    idx=hkl2.index.values #se encuentra como array [valor]
    idx=int(idx)
    fila.append(idx)
    if len(H) ==1:
        df.loc[idx, 'h']= int(H)
        df.loc[idx, 'k']= int(K)
        df.loc[idx, 'l']= int(L)
    else:
        H=H.tolist()
        K=K.tolist()
        L=L.tolist()
        df.loc[idx, 'h']= int(H[0])
        df.loc[idx, 'k']= int(K[0])
        df.loc[idx, 'l']= int(L[0])
###############################################################
PD=[]
i=0
while(True):
    PD.append(((df.iloc[i,3]),(df.iloc[i,4])))
    i=i+1
    if len(df.index)==i:
        break
H, K, L=[], [], []
H2, K2, L2=[], [], []
for s in df['h']:
    H.append(str(s))
    if s != '':
        H2.append(str(s))
for s in df['k']:
    K.append(str(s))
    if s != '':
        K2.append(str(s))
for s in df['l']:
    L.append(str(s))
    if s != '':
        L2.append(str(s))
        
miller= list(zip(H, K, L))
print(miller)
assert len(PD)==len(miller)

cnt=0
for u, v in PD:
    d=''.join(miller[cnt])
    if d != '':
        img= cv2.rectangle(img, (u-15,v-15), (u+15,v+15), (255,0,0), 1)
    plt.text(u-10, v-10, ''.join(miller[cnt]), color='lime', fontsize=7)
    cnt=cnt+1
data2={'h':H2,
       'k':K2,
       'l':L2,
       'dA-Python':d_enc,
       'dA-DRX-PTCLab':d_rep,
       'Diferencia':dif}
df_final= pd.DataFrame(data2)
with open('tabla_final.tex','w') as tf:
    tf.write(df_final.to_latex())
print("##### TERMINO! #####")











