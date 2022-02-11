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
    muestra= cv2.rectangle(muestra, (0,900), (300,w-1), (0,0,0), -1)
    for S in rangos:
        #print("muestra",S)
        antes_ciclo = time() ####!!!!!!!TIEMPO
    #imagen de cv viene en un array
        antes = time() ####!!!!!!!TIEMPO
        h, w, c = muestra.shape
        #print(type(muestra))
        imorig= cv2.rectangle(muestra, (0,0), (w-1,h-1), (0,0,0), 1)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("0-Lectura imagen original:",Tiempo)
        #plt.title("P0: original")            
        #plt.imshow(imorig)              
        #plt.show()

########### Pinta segun el rango de intensidad ########## 
        antes = time() ####!!!!!!!TIEMPO
        blanco=(255,255,255)
        negro=(0,0,0)
        rango=(S,S,S)
        imorig= Image.fromarray(imorig)
        imorig= ran_inten(imorig,h,w,rango)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("1-Seccion de intensidad:",Tiempo)
        #plt.title("P1: pinta rango de intensidad(for)")            
        #plt.imshow(imorig)     
        #plt.show()    

############### negativo ##########################
        antes = time() ####!!!!!!!TIEMPO
        imorig=np.array(imorig)
        negativo = cv2.bitwise_not(imorig)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("2-Negativo:",Tiempo)
        #plt.title("P2:Se hace el negativo")
        #plt.imshow(negativo)
        #plt.show()

############### escala de grises #######################
        antes = time() ####!!!!!!!TIEMPO
        gray= cv2.cvtColor(negativo, cv2.COLOR_BGR2GRAY)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("3-Escala de grises:",Tiempo)
        #plt.imshow(gray,cmap='gray')
        #plt.title("P3: Escala de grises")
        #plt.show()
################# MEDIAN BLUR #################################
        antes = time() ####!!!!!!!TIEMPO
        median = cv2.medianBlur(gray, 9)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("4-Filtro de media:",Tiempo)
        #plt.imshow(median,cmap='gray')
        #plt.title("P4: Medianblur 5")
        #plt.show()

########### ruido gaussiano #################################
        antes = time() ####!!!!!!!TIEMPO
        blur = cv2.GaussianBlur(median,(11,11), 0)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("5-Ruido Gaussiano:",Tiempo)
        #plt.imshow(blur,cmap='gray')
        #plt.title("P5: Gaussian Blur")
        #plt.show()

############## canny contorno ##############################
        antes = time() ####!!!!!!!TIEMPO    
        canny= cv2.Canny(blur,30,150,3)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("6-Canny Bordes:",Tiempo)
        #plt.imshow(canny,cmap='gray')
        #plt.title('P6: Canny aplicado a blur')
        #plt.show()
               
#################### DILATED ##################################
        antes = time() ####!!!!!!!TIEMPO
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        dilated = cv2.dilate(canny, kernel)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("7-Operacion Dilatación:",Tiempo)
        #plt.imshow(dilated)
        #plt.title('P7: dilated')
        #plt.show()

################## Detección contornos ########################
        antes = time() ####!!!!!!!TIEMPO
        contornos=[]
        verde=(0,255,0)
        (cnt, heirarchy)= cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb,cnt, -1, verde, 2)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("8-Deteccion de contornos:",Tiempo)
        #plt.imshow(rgb)
        #plt.title('P8: Contornos')
        #plt.show()
        
################## contorno mayor    ###########################
        antes = time() ####!!!!!!!TIEMPO
        (cnt, heirarchy)= cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
        for i in range(len(cnt)):
            contornos.append([len(cnt[i]),i])  
        mayor=max(contornos)
        borde=mayor[0]
        pos=mayor[1]
        cv2.drawContours(rgb,cnt, pos, verde, 2)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("9-contorno mayor:",Tiempo)
        #plt.imshow(rgb)
        #plt.title('P9: Contorno mayor')
        #plt.show()

##################### Borra todo menos el contorno ########################
        #antes = time() ####!!!!!!!TIEMPO
        new= Image.fromarray(rgb)
        antes = time() ####!!!!!!!TIEMPO
        new=np.array(new)
        new= cont_solo(new,verde)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("10.1-Contorno solo numpy:",Tiempo)
        xmin=(min(new[1]))[0]
        xmax=(max(new[1]))[0]
        ymin=(min(new[2]))[0]
        ymax=(max(new[2]))[0]
        new=new[0]
        #print("10-Contorno único:",Tiempo)
        #plt.imshow(new)
        #plt.title('P10: Unifica el contorno')
        #plt.show()
    
#################Limites y median blur para hough############################
        antes = time() ####!!!!!!!TIEMPO
        X=xmax-xmin
        Y=ymax-ymin
        diametros=X,Y
        rmax=(ceil(max(diametros)/2))
        rmin=(floor(min(diametros)/2))

        cen_circ=cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        cen_circ = cv2.medianBlur(cen_circ,5)
        Tiempo= (time() - antes)####!!!!!!!TIEMPO
        #print("11-Media para Hough:",Tiempo)
        #plt.imshow(cen_circ)
        #plt.title('P11: Median blur para Hough')
        #plt.show()

###################### HOUGH CIRCLE ###########################################
        try:
            antes = time() ####!!!!!!!TIEMPO    
            circles = cv2.HoughCircles(cen_circ,cv2.HOUGH_GRADIENT,1,1200,
                    param1=100,param2=10,minRadius=rmin,maxRadius=rmax)#80,20
            circles = np.uint16(np.around(circles))#circles arroja(x,y,radio)
            #print("circles",circles)
            rojo=(255,0,0)
            azul=(0,0,255)
            cimg = cv2.cvtColor(cen_circ,cv2.COLOR_GRAY2BGR)
            for i in circles[0,:]: #dibuja la cantidad de circulos detectados
            #dibuja fuera del circulo
                cv2.circle(cimg,(i[0],i[1]),i[2],verde,1)
            #dibuja el centro del circulo
                cv2.circle(cimg,(i[0],i[1]),2,rojo,-1)        
        
            cen_x=circles[0][0][0]
            cen_y=circles[0][0][1]
            pix_cen= (cen_y, cen_x)
            cimg[pix_cen]= azul
            Tiempo= (time() - antes)####!!!!!!!TIEMPO
            #print("12-Circulo encontrado:",Tiempo)
            #plt.title('P12: Circulo encontrado con centro')
            #plt.imshow(cimg)    ###################
            #plt.show()          ####################
            #print("Si detecto")

        except TypeError:
            #print("No detecto")
            pix_cen=(0,0)
    
        if pix_cen != (0,0):
            CX.append(pix_cen[0])
            CY.append(pix_cen[1])
        #print(pix_cen)
        Tiempo_ciclo= (time() - antes_ciclo)####!!!!!!!TIEMPO
        #print(Tiempo_ciclo)

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
                #print("hubo uno inferior",CX[i])
                nuevos_x= CX2.remove(CX[i])
            if CX[i] > big_supx:
                #print("hubo uno superior",CX[i])
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
                #print("hubo uno inferior",CY[i])
                nuevos_y= CY2.remove(CY[i])
            if CY[i] > big_supy:
                #print("hubo uno superior",CY[i])
                nuevos_y= CY2.remove(CY[i])
                
        CX=CX2
        CY=CY2
        X_data.append(CX)
        Y_data.append(CY)

    #plt.subplot(2, 2, 1)
    #plt.xlabel('Muestras')
    #plt.ylabel('Píxel')
    #plt.plot(range(len(X_data[0])), X_data[0], color='r')
    #plt.scatter(range(len(X_data[0])), X_data[0], marker="x", color="r", s=25,label='Pixel X')
    #plt.plot(range(len(Y_data[0])), Y_data[0], color='b')
    #plt.scatter(range(len(Y_data[0])), Y_data[0], marker="1", color="b", s=30,label='Pixel Y')
    #plt.legend()
    #plt.legend(loc='best', title='Posicion del pixel')

    #plt.subplot(2, 2, 2)
    #plt.xlabel('Muestras')
    #plt.ylabel('Píxel')
    #plt.plot(range(len(X_data[1])), X_data[1], color='r')
    #plt.scatter(range(len(X_data[1])), X_data[1], marker="x", color="r", s=25,label='Pixel X')
    #plt.plot(range(len(Y_data[1])), Y_data[1], color='b')
    #plt.scatter(range(len(Y_data[1])), Y_data[1], marker="1", color="b", s=30,label='Pixel Y')
    #plt.legend()
    #plt.legend(loc='best', title='Posicion del pixel')

    #plt.subplot(2, 2, 3)
    #plt.xlabel('Muestras')
    #plt.ylabel('Píxel')
    #plt.plot(range(len(X_data[2])), X_data[2], color='r')
    #plt.scatter(range(len(X_data[2])), X_data[2], marker="x", color="r", s=25,label='Pixel X')
    #plt.plot(range(len(Y_data[2])), Y_data[2], color='b')
    #plt.scatter(range(len(Y_data[2])), Y_data[2], marker="1", color="b", s=30,label='Pixel Y')
    #plt.legend()
    #plt.legend(loc='best', title='Posicion del pixel')

    #plt.subplot(2, 2, 4)
    #plt.xlabel('Muestras')
    #plt.ylabel('Píxel')
    #plt.plot(range(len(X_data[3])), X_data[3], color='r')
    #plt.scatter(range(len(X_data[3])), X_data[3], marker="x", color="r", s=25,label='Pixel X')
    #plt.plot(range(len(Y_data[3])), Y_data[3], color='b')
    #plt.scatter(range(len(Y_data[3])), Y_data[3], marker="1", color="b", s=30,label='Pixel Y')
    #plt.legend()
    #plt.legend(loc='best', title='Posicion del pixel')
    #plt.show()
    
    #plt.subplot(2, 1, 1)
    #plt.title("Datos coordenada X")
    #plt.boxplot(X_data)
    #plt.ylabel('posiciones')
    #plt.subplot(2, 1, 2)
    #plt.title("Datos coordenada Y")
    #plt.boxplot(Y_data)
    #plt.ylabel('posiciones')
    #plt.show()
        
    Tiempo= (time() - antes)
    #print("Tiempo de analisis Graficos: ", Tiempo)    

    X= ((sum(CX))/(len(CX)))
    Y= ((sum(CY))/(len(CY)))
    pix_cen=(round(X), round(Y))
    print("centro promedio:",pix_cen)
    #muestra[pix_cen]= (0,255,0)
    #plt.title('P12: Circulo encontrado con centro')
    #plt.imshow(muestra)
    #plt.show()      
    return(pix_cen)



