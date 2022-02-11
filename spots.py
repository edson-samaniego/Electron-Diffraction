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
    esquina =(round(h/2),0)
    centro =(round(h/2),round(w/2))
    spot = cv2.line(crop_img, esquina, centro, (0,255,0), 1)
    diagonal=[]
    for m in range(round(h/2)+3):
        for n in range(round(w/2)+3):
            pix= spot[m,n]
            if (pix[0],pix[1],pix[2]) == (0,255,0):
                diagonal.append(crop_img2[m,n][0])             
    diagonal=diagonal[3:]
    imorig=np.array(crop_img2)
    im_copy=imorig.copy()
    blanco=(255,255,255)
    negro=(0,0,0)
    n_centro=[]
    n2_centro=[]
    n3_centro=[]
    #print("############# muestras:",len(diagonal),"################")
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
        params.minArea = 1
        params.filterByCircularity = True
        params.minCircularity = 0.6
        params.filterByConvexity = True
        params.minConvexity = 0.2
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
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
        #plt.subplot(111), plt.imshow(blobs), plt.title('spot')
        #plt.show()
    contador=0
    centro_spot=[]
    for s in n_centro, n2_centro, n3_centro:
        if len(s)>0:
            for L in range(0,contador+1):
                centro_spot.append(s[L])
        contador=contador+1
    #print(centro_spot)
    return(centro_spot)
###########################################################################
def center(muestra):
    im2=muestra.copy()
    h, w, c = muestra.shape
    img= cv2.rectangle(muestra, (0,0), (w-1,h-1), (0,0,0), 1)
    ret, bw_img = cv2.threshold(muestra, 10, 255, cv2.THRESH_BINARY)

    gray= cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    blur = cv2.GaussianBlur(median,(11,11), 0)
    canny= cv2.Canny(blur,30,150,3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(canny, kernel)

    (cnt, heirarchy)= cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)

    esq_sup=[]
    esq_inf=[]
    posiciones=[]
    for t in cnt[:]:
        cv2.drawContours(rgb,t, -1, (0,255,0), 1)
        t=t.tolist()
        #print("minX:",min(t), "maxX:",max(t)) 
        q = [(item[0][1],item[0][0]) for item in t]
        #print("minY:",min(q), "maxY:",max(q))
        minx=(min(t))[0][0]-4
        maxx=(max(t))[0][0]+4
        miny=(min(q))[0]-4
        maxy=(max(q))[0]+4
        esq_sup.append((minx,miny))#guarda secciones
        esq_inf.append((maxx,maxy))#guarda secciones
        esq_sup2=(minx,miny)
        esq_inf2=(maxx,maxy)
        #rgb= cv2.rectangle(rgb, esq_sup2, esq_inf2, (255,0,0), 1)
        rgb= cv2.rectangle(im2, esq_sup2, esq_inf2, (255,0,0), 1)
        cent= spot_center(muestra, minx, maxx, miny, maxy)
        for s in cent:
            posiciones.append(((miny+s[0]),(minx+s[1])))
            rgb[(miny+s[0]),(minx+s[1])]= (255,0,0)
        #plt.subplot(111), plt.imshow(rgb), plt.title('deteccion')
        #plt.show()
    return(rgb, posiciones)

