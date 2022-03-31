import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import * 
from math import sqrt, fabs
from PIL import Image, ImageChops
import math
from scipy.signal import find_peaks


def resta(muestra,pixel_cen):
##################### Genera diagonal  ###############################
    h, w, c = muestra.shape
    muestra= cv2.rectangle(muestra, (0,900), (300,w-1), (0,0,0), -1)
    orig =muestra
    linea=orig.copy()
    Y, X = orig.shape[0], orig.shape[1]
    centro= pixel_cen
    esquina =(0,0) #(Y, X)
    E1=sqrt((esquina[0]-centro[0])**2+(esquina[1]-centro[1])**2), esquina
    verde=(0,255,0)
    end_point = (esquina)
    linea = cv2.line(linea,(centro[1],centro[0]), end_point, verde, 1)
    #plt.imshow(linea)
    #plt.show()

################# Guarda intensidades y euclidianas #####################
    diagonal=[]
    posit=[]
    for m in range(Y):
        for n in range(X):
            pix= linea[m,n]
            if (pix[0],pix[1],pix[2]) == verde:
                diagonal.append((orig[m,n][0]))
                eucl=sqrt((n-centro[1])**2+(m-centro[0])**2) 
                posit.append(eucl)

################ elimina bajada final si es que hay #######################
    diagonal.reverse()
    diagonal2=diagonal.copy()
    #plt.subplot(122), plt.plot((range(0,len(diagonal))),diagonal)
    #plt.subplot(121), plt.imshow(linea), plt.title('antes de recorte')
    #plt.show()
### agregar si el siguiente a el pixel leido es mayor entonces borrar actual
    for e in range(0,len(diagonal2)):
        diagonal.pop(0)
        if diagonal2[e] > 230:##### utilizo rango, ver como ajustar
            break
    diagonal.reverse()
    #plt.subplot(122), plt.plot((range(0,len(diagonal))),diagonal)
    #plt.subplot(121), plt.imshow(linea), plt.title('linea')
    #plt.show()

#################### Se aplica regresion polinomial ###########################
    pixeles_diag=len(diagonal)
    x=(range(0,len(diagonal)))
    y=diagonal
    inverso = [i * -1 for i in diagonal]
    pixeles_diag2=len(inverso)
    x2=(range(0,len(inverso)))
    y2=inverso

    mymodel = np.poly1d(np.polyfit(x, y, 6))
    myline = np.linspace(0, len(diagonal), pixeles_diag)
    mymodel2 = np.poly1d(np.polyfit(x2, y2, 6))
    myline2 = np.linspace(0, len(inverso), pixeles_diag2)
    #plt.plot(x, y)
    #plt.plot(myline, mymodel(myline))
    #plt.show()

###################### buscar picos de grafico ##################################
    data=(myline, mymodel(myline))
    peaks = find_peaks(diagonal, height= (data[1], 230),distance=10, prominence=15)
    height = peaks[1]['peak_heights'] 
    peak_pos = peaks[0]

    data2=(myline2, mymodel2(myline2))
    peaks2 = find_peaks(inverso, height= (data2[1], 230))#,distance=10, prominence=5)
    height2 = peaks2[1]['peak_heights'] 
    peak_pos2 = peaks2[0]
    real_height2 = height2 *-1

#plt.plot(data[1], "--", color="gray")
    #plt.plot(myline, mymodel(myline),color='gray')
    #plt.scatter(peak_pos, height, color= 'r', s=90, marker='x')
#plt.plot((range(0,len(diagonal))),diagonal, color='blue')
    #plt.scatter((range(0,len(diagonal))),diagonal, color='blue', s=8)
    #plt.show()

    #plt.plot(myline, mymodel(myline),color='gray')
    #plt.plot((range(0,len(diagonal))), diagonal, color= 'green')
    #plt.scatter(peak_pos, height, color= 'r', s=90, marker='x')
    #plt.scatter((range(0,len(diagonal))), diagonal, color='blue', s=8)
#plt.plot(myline2, mymodel2(myline2),color='gray')
#plt.plot((range(0,len(inverso))), inverso, color= 'green')
    #plt.scatter(peak_pos2, real_height2, color= 'r', s=90, marker='x')
#plt.scatter((range(0,len(inverso))),inverso, color='blue', s=8)
    #plt.show()

######################### elimina picos ###################################
    for s in peak_pos[0:]:
        #print("################## pico posicion:",s, " ##################")
        menores=[]
        mayores=[]
        contador=0
        for t in peak_pos2[0:]:
            if t < s:
                u=real_height2[contador]
                menores.append((t,u))
            else:
                u=real_height2[contador]
                mayores.append((t,u))
            contador=contador+1
        lim_men=max(menores)
        lim_may=min(mayores)   
        pt1=diagonal[:lim_men[0]]
        pt2=diagonal[lim_may[0]:]
        dif_hor=lim_may[0]-lim_men[0]
        dif_vert=lim_may[1]-lim_men[1]
        datavert=list(range(int(lim_men[1]),int(lim_may[1]+1)))
        divis=dif_vert/dif_hor
        remplazo=list(range(1,dif_hor+1))
        nuevo = [round(i * divis) for i in remplazo]
        nuevo2=[]
        ###### error aqui en otras muestras picos dobles no hay minimo y brinca
        #print("nuevo",nuevo)
        for s in nuevo:
        #    print(datavert[s])
            nuevo2.append(datavert[s])    
        diagonal=pt1 + nuevo2 + pt2
        
        #plt.plot(myline, mymodel(myline),color='gray')
        #plt.plot((range(0,len(diagonal))), diagonal, color= 'green')
        #plt.scatter(peak_pos, height, color= 'r', s=90, marker='x')
        #plt.scatter((range(0,len(diagonal))), diagonal, color='blue', s=8)
    #plt.plot(myline2, mymodel2(myline2),color='gray')
    #plt.plot((range(0,len(inverso))), inverso, color= 'green')
        #plt.scatter(peak_pos2, real_height2, color= 'r', s=90, marker='x')
    #plt.scatter((range(0,len(inverso))),inverso, color='blue', s=8)
        #plt.show()
    
############################## nuevo gradiente ###############################
    im = np.zeros((Y,X,3),np.uint8)
    for G in range(len(diagonal)):
        C=diagonal[G]
        radius = round(posit[G])
        color = (int(C), int(C), int(C))
        im10 = cv2.circle(im, (centro[1], centro[0]), radius, color, -1)
    suma=cv2.add(orig,im10)
    resta=cv2.subtract(orig,im10)

    #plt.subplot(121), plt.imshow(orig), plt.title('original')
    #plt.subplot(122),plt.imshow(im10), plt.title('gradiente')
    #plt.subplot(133), plt.imshow(resta), plt.title('resta')
    #plt.show()

    #plt.subplot(121), plt.imshow(orig), plt.title('original')
    #plt.subplot(122), plt.imshow(resta), plt.title('sin gradiente')
    #plt.show()
    return(resta)
