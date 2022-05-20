import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, fabs, ceil, floor
import cv2
from matplotlib.pyplot import * 
from math import sqrt, fabs
from PIL import Image, ImageChops
from scipy.signal import find_peaks

def spot_center(img, minx, maxx, miny, maxy):
    crop_img = img[miny:maxy, minx:maxx]
    crop_img2=crop_img.copy()
    h, w, c = crop_img.shape
    esquina =(0,0)
    pmayor=[]
    for y in range(h):
        for x in range(w):
            px=crop_img[y][x]
            pmayor.append(((px[0]),y,x))
    maximo=max(pmayor)
    centro =(maximo[1],maximo[2])
    spot = cv2.line(crop_img2, esquina, centro, (0,255,0), 1)
    diagonal=[]
    for m in range(h):
        for n in range(w):
            pix= spot[m,n]
            if (pix[0],pix[1],pix[2]) == (0,255,0):
                diagonal.append(crop_img[m,n][0])             
    diagonal=diagonal[-10:]
    imorig=np.array(crop_img)
    im_copy=imorig.copy()
    blanco=(255,255,255)
    negro=(0,0,0)
    n_centro=[]
    n2_centro=[]
    n3_centro=[]
    for s in diagonal:
        rango=(s,s,s)
        black_pixels_mask = np.all(imorig <= rango, axis=-1)
        non_black_pixels_mask = np.any(imorig > rango, axis=-1)
        im_copy[black_pixels_mask] = [255, 255, 255]
        im_copy[non_black_pixels_mask] = [0, 0, 0]    
        
        imorig2=np.array(im_copy)
        negativo = cv2.bitwise_not(imorig2)
        gray= cv2.cvtColor(negativo, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 9)
        blur = cv2.GaussianBlur(median,(11,11), 0)
        canny= cv2.Canny(blur,30,150,3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        dilated = cv2.dilate(canny, kernel)
        contornos=[]
        verde=(0,255,0)
        (cnt, heirarchy)= cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb,cnt, -1, verde, 2)

        white = np.any(rgb != verde, axis=-1)
        rgb[white] = [0, 0, 0]
        indices = np.where(np.all(rgb == verde, axis=-1))
        cen_circ=cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        cen_circ = cv2.medianBlur(cen_circ,5)
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = .01
        params.filterByCircularity = True
        params.minCircularity = 0.01#0.6
        params.filterByConvexity = True
        params.minConvexity = 0.3
        params.filterByInertia = True
        params.minInertiaRatio = 0.001
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(cen_circ)
        for keyPoint in keypoints:
            x = round(keyPoint.pt[0])
            y = round(keyPoint.pt[1])
            s = keyPoint.size
            if len(keypoints)== 1:
                n_centro.append((y,x))
            if len(keypoints)== 2:
                n2_centro.append((y,x))
                n_centro=[]
            if len(keypoints)== 3:
                n3_centro.append((y,x))
                n_centro=[]
                n2_centro=[]
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(cen_circ, keypoints, blank, (255,0,0),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    contador=0
    centro_spot=[]
    for s in n_centro, n2_centro, n3_centro:
        if len(s)>0:
            for L in range(0,contador+1):
                centro_spot.append(s[L])
        contador=contador+1
    return(centro_spot)
########################
def intensidad(img, minx, maxx, miny, maxy ):
    crop_img = img[miny:maxy, minx:maxx]
    crop_img2=crop_img.copy()
    h, w, c = crop_img.shape
    pixels=[]
    for y in range(h):
        for x in range(w):
            px=crop_img[y][x]
            pixels.append(px[0])
    minimo=min(pixels)
    maximo=max(pixels)
    minimo=minimo+5
    pixels=[i  for i in pixels if i > minimo]
    promedio= sum(pixels)/len(pixels)
    return(maximo)
###########################################################################
def center(muestra):
    h, w, c = muestra.shape
    muestra= cv2.rectangle(muestra, (0,0), (w-1,h-1), (0,0,0), 1)
    img=muestra.copy()
    im2=muestra.copy()
    im3=muestra.copy()
    ret, bw_img = cv2.threshold(muestra, 10, 255, cv2.THRESH_BINARY)

    gray= cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    blur = cv2.GaussianBlur(median,(11,11), 0)
    canny= cv2.Canny(blur,30,150,3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(canny, kernel)

    (cnt, heirarchy)= cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)

    cuadro=[]
    posiciones=[]
    seccion_cent=[]
    for t in cnt[:]: 
        cv2.drawContours(rgb,t, -1, (0,255,0), 1)
        t=t.tolist()
        q = [(item[0][1],item[0][0]) for item in t]
        minx=(min(t))[0][0]-4
        maxx=(max(t))[0][0]+4
        miny=(min(q))[0]-4
        maxy=(max(q))[0]+4
        esq_sup2=(minx,miny)
        esq_inf2=(maxx,maxy)
        rgb= cv2.rectangle(im2, esq_sup2, esq_inf2, (255,0,0), 1)
        brillo = intensidad(muestra, minx, maxx, miny, maxy)
        cent= spot_center(muestra, minx, maxx, miny, maxy)
        for s in cent:
            posiciones.append(((miny+s[0]),(minx+s[1])))
            seccion_cent.append(brillo)
            cuadro.append(((minx,miny),(maxx,maxy)))
            rgb[(miny+s[0]),(minx+s[1])]= (255,0,0)
    mayor=max(seccion_cent)
    porcentajes=[round(((f/mayor)*100),1) for f in seccion_cent]
    return(rgb, posiciones, porcentajes, cuadro)

