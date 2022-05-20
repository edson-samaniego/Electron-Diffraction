from PIL import Image, ImageFilter, ImageDraw
import turtle
from math import sqrt, fabs, ceil, floor
from cv2 import cv2
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageFilter import (BLUR, CONTOUR, DETAIL, EDGE_ENHANCE,
                             EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES,
                             SMOOTH, SMOOTH_MORE, SHARPEN)

def ran_inten(imorig,h,w,rango):
    imorig=np.array(imorig)
    im_copy=imorig.copy()
    blanco=(255,255,255)
    negro=(0,0,0)
    
    black_pixels_mask = np.all(imorig <= rango, axis=-1)
    non_black_pixels_mask = np.any(imorig > rango, axis=-1)

    im_copy[black_pixels_mask] = [255, 255, 255]
    im_copy[non_black_pixels_mask] = [0, 0, 0]
    return (im_copy)

def cont_solo(new,verde):
    white = np.any(new != verde, axis=-1)
    new[white] = [0, 0, 0]
    indices = np.where(np.all(new == verde, axis=-1))
    pyborde = zip(indices[0])
    pxborde = zip(indices[1])
    return(new,list(pxborde),list(pyborde))
#######################################
def center(muestra, rangos):
    CX=[]
    CY=[]
    h, w, c = muestra.shape
    implot= muestra.copy()
    muestra= cv2.rectangle(muestra, (0,900), (300,w-1), (0,0,0), -1)
    for S in rangos:
        antes_ciclo = time()
        antes = time()
        h, w, c = muestra.shape
        imorig= cv2.rectangle(muestra, (0,0), (w-1,h-1), (0,0,0), 1)
        Tiempo= (time() - antes)
########### Pinta segun el rango de intensidad ########## 
        antes = time()
        blanco=(255,255,255)
        negro=(0,0,0)
        rango=(S,S,S)
        imorig= Image.fromarray(imorig)
        imorig= ran_inten(imorig,h,w,rango)
        Tiempo= (time() - antes)
############### negativo ##########################
        antes = time()
        imorig=np.array(imorig)
        negativo = cv2.bitwise_not(imorig)
        Tiempo= (time() - antes)
############### escala de grises #######################
        antes = time()
        gray= cv2.cvtColor(negativo, cv2.COLOR_BGR2GRAY)
        Tiempo= (time() - antes)
################# MEDIAN BLUR #################################
        antes = time()
        median = cv2.medianBlur(gray, 9)
        Tiempo= (time() - antes)
########### ruido gaussiano #################################
        antes = time()
        blur = cv2.GaussianBlur(median,(11,11), 0)
        Tiempo= (time() - antes)
############## canny contorno ##############################
        antes = time()   
        canny= cv2.Canny(blur,30,150,3)
        Tiempo= (time() - antes)
#################### DILATED ##################################
        antes = time() 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        dilated = cv2.dilate(canny, kernel)
        Tiempo= (time() - antes)
################## Detección contornos ########################
        antes = time()
        contornos=[]
        verde=(0,255,0)
        (cnt, heirarchy)= cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb,cnt, -1, verde, 2)
        Tiempo= (time() - antes)
################## contorno mayor    ###########################
        antes = time() 
        (cnt, heirarchy)= cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
        for i in range(len(cnt)):
            contornos.append([len(cnt[i]),i])
        if len(contornos) == 0:
            break
        mayor=max(contornos)
        borde=mayor[0]
        pos=mayor[1]
        cv2.drawContours(rgb,cnt, pos, verde, 2)
        Tiempo= (time() - antes)
##################### Borra todo menos el contorno ########################
        new= Image.fromarray(rgb)
        antes = time()
        new=np.array(new)
        new= cont_solo(new,verde)
        Tiempo= (time() - antes)
        xmin=(min(new[1]))[0]
        xmax=(max(new[1]))[0]
        ymin=(min(new[2]))[0]
        ymax=(max(new[2]))[0]
        new=new[0]
#################Limites y median blur para hough############################
        antes = time()
        X=xmax-xmin
        Y=ymax-ymin
        diametros=X,Y
        rmax=(ceil(max(diametros)/2))
        rmin=(floor(min(diametros)/2))
        cen_circ=cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        cen_circ = cv2.medianBlur(cen_circ,5)
        Tiempo= (time() - antes)
###################### HOUGH CIRCLE ###########################################
        try:
            antes = time()   
            circles = cv2.HoughCircles(cen_circ,cv2.HOUGH_GRADIENT,1,1200,
                    param1=100,param2=10,minRadius=rmin,maxRadius=rmax)#80,20
            circles = np.uint16(np.around(circles))
            rojo=(255,0,0)
            azul=(0,0,255)
            cimg = cv2.cvtColor(cen_circ,cv2.COLOR_GRAY2BGR)
            for i in circles[0,:]: 
                cv2.circle(cimg,(i[0],i[1]),i[2],verde,1)
                cv2.circle(cimg,(i[0],i[1]),2,rojo,-1)        
        
            cen_x=circles[0][0][0]
            cen_y=circles[0][0][1]
            pix_cen= (cen_y, cen_x)
            cimg[pix_cen]= azul
            Tiempo= (time() - antes)
        except TypeError:
            pix_cen=(0,0)
    
        if pix_cen != (0,0):
            CX.append(pix_cen[0])
            CY.append(pix_cen[1])
        Tiempo_ciclo= (time() - antes_ciclo)
###################### caja bigote #############################
    antes = time()
    X_data=[]
    Y_data=[]
    X_data.append(CX)
    Y_data.append(CY)
    for M in range(3):
        
        CX2= CX[:]
        Qx1=np.quantile(CX,.25)
        Qx3=np.quantile(CX,.75)
        IQRx= Qx3 - Qx1
        medianaX= np.median(CX)
        Vx_min = np.min(CX)
        Vx_max = np.max(CX)
        big_infx= (Qx1 - 1.5 * IQRx)
        big_supx= (Qx3 + 1.5 * IQRx)
        outliers_x= ([CX] < big_infx) | ([CX] > big_supx)
        for i in range(len(CX)):
            if CX[i] < big_infx:
                nuevos_x= CX2.remove(CX[i])
            if CX[i] > big_supx:
                nuevos_x= CX2.remove(CX[i])

        CY2= CY[:]
        Qy1=np.quantile(CY,.25)
        Qy3=np.quantile(CY,.75)
        IQRy= Qy3 - Qy1
        medianaY= np.median(CY)
        Vy_min = np.min(CY)
        Vy_max = np.max(CY)
        big_infy= (Qy1 - 1.5 * IQRy)
        big_supy= (Qy3 + 1.5 * IQRy)
        outliers_y= ([CY] < big_infy) | ([CY] > big_supy)
        for i in range(len(CY)):
            if CY[i] < big_infy:
                nuevos_y= CY2.remove(CY[i])
            if CY[i] > big_supy:
                nuevos_y= CY2.remove(CY[i])
        CX=CX2
        CY=CY2
        X_data.append(CX)
        Y_data.append(CY)
    Tiempo= (time() - antes)
    X= ((sum(CX))/(len(CX)))
    Y= ((sum(CY))/(len(CY)))
    pix_cen=(round(X), round(Y))
    return(pix_cen)



