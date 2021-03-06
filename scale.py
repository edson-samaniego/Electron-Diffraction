import pytesseract
import cv2
import numpy as np
from math import sqrt, fabs, ceil, floor
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt
##########################################################################

def vec_brd(im,coordenadas):
    verde=(0,255,0)
    e=coordenadas[0]
    f=coordenadas[1]
    acumulados=[]
    p0=im[e-1,f-1]
    p1=im[e-1,f]
    p2=im[e-1,f+1]
    p3=im[e,f-1]
    p5=im[e,f+1]
    p6=im[e+1,f-1]
    p7=im[e+1,f]
    p8=im[e+1,f+1]
    V=(e-1,f-1),(e-1,f), (e-1,f+1),(e,f-1),(e,f+1),(e+1,f-1),(e+1,f),(e+1,f+1)
    vecinos=((p0[0],p0[1],p0[2]),(p1[0],p1[1],p1[2]),(p2[0],p2[1],p2[2]),
            (p3[0],p3[1],p3[2]),(p5[0],p5[1],p5[2]),(p6[0],p6[1],p6[2]),
            (p7[0],p7[1],p7[2]),(p8[0],p8[1],p8[2]))
    if (all([u == (0,0,0) for u in vecinos]))== True:
        im[e,f]=(0,0,0)
    while((any([u == verde for u in vecinos]))== True):
        guarda=[]
        acumulados.append((e,f))
        for s in range(len(vecinos)):
            pos=vecinos[s]
            if (pos[0],pos[1],pos[2])==verde:
                guarda.append(V[s])
        
        for r in range(len(guarda)):
            M=guarda[r]
            if r > 1:
                im[M[0],M[1]]=(0,0,0)
                
        im[e,f]=(0,0,0)
        nuevo=guarda[0]
        e=nuevo[0]
        f=nuevo[1]
        p0=im[e-1,f-1]
        p1=im[e-1,f]
        p2=im[e-1,f+1]
        p3=im[e,f-1]
        p5=im[e,f+1]
        p6=im[e+1,f-1]
        p7=im[e+1,f]
        p8=im[e+1,f+1]
        V=(e-1,f-1),(e-1,f), (e-1,f+1),(e,f-1),(e,f+1),(e+1,f-1),(e+1,f),(e+1,f+1)
        vecinos=((p0[0],p0[1],p0[2]),(p1[0],p1[1],p1[2]),(p2[0],p2[1],p2[2]),
                (p3[0],p3[1],p3[2]),(p5[0],p5[1],p5[2]),(p6[0],p6[1],p6[2]),
                (p7[0],p7[1],p7[2]),(p8[0],p8[1],p8[2]))
    return(acumulados)

def conversion(muestra):
    im1= muestra
    h=im1.shape[0]
    w=im1.shape[1]
    verde=(0,255,0)
    crd1=(900,0)
    crd2=(1020,400)
    cv2.rectangle(im1,(crd1[1],crd1[0]),(crd2[1],crd2[0]),verde,1)#rectangulo es [x,y]
    imb=im1.copy()
############ se revisa solo la seccion y color negro el resto ################
    for a in range(h):
        for b in range(w):
            if (crd1[0]) < a <(crd2[0]) and (crd1[1]) < b < (crd2[1]):
                pos=im1[a,b]
                if pos[0] >= 140:
                    im1[a,b]=(255,255,255)
                else:
                    im1[a,b]=(0,0,0)
            else:
                im1[a,b]=(0,0,0)
    imb1=im1.copy()
############################## BORDES ########################################
    imb= im1.copy()
    rojo=(255,0,0)
    verde=(0,255,0)
    azul=(0,0,255)
    for c in range(crd1[0],crd2[0]+1):
        for d in range(crd1[1],crd2[1]+1):
            pos2=imb[c,d]
            if pos2[0]== (255):
                p1=imb[c-1,d-1]
                p2=imb[c-1,d]
                p3=imb[c-1,d+1]
                p4=imb[c,d-1]
                p6=imb[c,d+1]
                p7=imb[c+1,d-1]
                p8=imb[c+1,d]
                p9=imb[c+1,d+1]
                vecinos=(p2[0],p4[0],p6[0],p8[0])#conectividad 4
                if (any([u == 0 for u in vecinos]))== True:
                    im1[c,d]=verde
                else:
                    im1[c,d]=(0,0,0)
###################### elimina verdes por grupos #############################
    imb2=im1.copy()
    contorno=[]
    for e in range(crd1[0],crd2[0]+1):
        for f in range(crd1[1],crd2[1]+1):
            pos3=imb2[e,f]
            if (pos3[0],pos3[1],pos3[2]) == verde:
                pcentral=(e,f)
                cnt=vec_brd(imb2,pcentral)
                contorno.append(cnt)
                for h in range(len(cnt)):
                    imb2[cnt[h]]=(0,0,0)

    rectan=(max(contorno))
    for L in range(len(rectan)):
        imb2[rectan[L]]=(verde)
    
    Y=[]
    X=[]
    for S in range(len(rectan)):
        N=rectan[S]
        Y.append(N[0])
        X.append(N[1])

    Ymin=min(Y)
    Ymax=max(Y)
    Xmin=min(X)
    Xmax=max(X)
    DE= sqrt((Xmax-Xmin)**2+(Ymin-Ymin)**2)
    p1=(str(Xmin),str(Ymin))
    p2=(str(Xmax),str(Ymin))
    p3=str(DE)
    pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe' 
    imb1 = cv2.cvtColor(imb1,cv2.COLOR_BGR2RGB)

    himg,wimg,_=imb1.shape
    boxes=pytesseract.image_to_boxes(imb1)
    conversion=[]
    for b in boxes.splitlines():
        b= b.split(' ')
        conversion.append(b[0])
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(imb1,(x,himg-y),(w,himg-h),(0,0,255),1)
    return(DE,conversion)


